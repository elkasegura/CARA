# ====================================================================================
# MODELO CAE: "Concrete Autoencoders: Differentiable Feature Selection and Reconstruction"
# Authors: Abubakar Abid, Muhammed Fatih Balin, James Zou1
# ICML 2019
# ====================================================================================

#Parameters
# K_features: número de features a seleccionar
# output_function: función que mapea las features seleccionadas a la salida reconstruida
# num_epochs: número de épocas de entrenamiento
# batch_size: tamaño de batch (None = full-batch)
# learning_rate: tasa de aprendizaje del optimizador Adam
# start_temp: temperatura inicial para el truco Gumbel-Softmax, este se reduce gradualmente para que la selección sea más "dura"y salga un vetor one-hot
# min_temp: temperatura mínima (annealing), aqui se detiene la reducción de temperatura
# tryout_limit: número máximo de intentos para alcanzar convergencia    

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import math
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant, glorot_normal
from tensorflow.keras.optimizers import Adam


class ConcreteSelect(Layer):

    def __init__(self, output_dim, start_temp=10.0, min_temp=0.1, alpha=0.99999, **kwargs):
        self.output_dim = output_dim
        self.start_temp = start_temp
        self.min_temp = K.constant(min_temp)
        self.alpha = K.constant(alpha)
        super(ConcreteSelect, self).__init__(**kwargs)

    def build(self, input_shape):
        self.temp = self.add_weight(
            name='temp',
            shape=[],
            initializer=Constant(self.start_temp),
            trainable=False
        )
        self.logits = self.add_weight(
            name='logits',
            shape=[self.output_dim, input_shape[1]],
            initializer=glorot_normal(),
            trainable=True
        )
        super(ConcreteSelect, self).build(input_shape)

    def call(self, X, training=None):
        #uniform = K.random_uniform(self.logits.shape, K.epsilon(), 1.0)
        uniform = tf.random.uniform(tf.shape(self.logits), K.epsilon(), 1.0)
        #gumbel = -K.log(-K.log(uniform))
        #aqui mete ruido a los logits
        gumbel = -tf.math.log(-tf.math.log(uniform))

        # Keras 3 compat: reemplaza K.update
        #temp = K.update(self.temp, K.maximum(self.min_temp, self.temp * self.alpha))
        temp = self.temp.assign(K.maximum(self.min_temp, self.temp * self.alpha))

        noisy_logits = (self.logits + gumbel) / temp
        #samples = K.softmax(noisy_logits)
        samples = tf.nn.softmax(noisy_logits, axis=-1)  # (K, D)


        # selección discreta ( argmax por cada una de las K filas)
        #discrete_logits = K.one_hot(K.argmax(self.logits, axis=-1), self.logits.shape[1])
        #discrete_logits = K.one_hot(K.argmax(self.logits), self.logits.shape[1])

        discrete_logits = tf.one_hot(tf.argmax(self.logits, axis=-1),   # (K,)
                                      depth=tf.shape(self.logits)[1]     # D
                                      
        )  # (K, D)

        # Keras 3 compat: reemplaza K.in_train_phase 
        #self.selections = K.in_train_phase(samples, discrete_logits, training)
        #Y = K.dot(X, K.transpose(self.selections))
        if training is None:
            training = tf.keras.backend.learning_phase()

        self.selections = tf.cond(
            tf.cast(training, tf.bool),
            lambda: samples,
            lambda: discrete_logits
        )
        Y = tf.linalg.matmul(X, self.selections, transpose_b=True) # (batch, K)
        
        return Y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class StopperCallback(EarlyStopping):
    #Esta clase detiene el entrenamiento cuando la media de las probabilidades máximas
    #de selección de features alcanza un valor objetivo (mean_max_target).
    def __init__(self, mean_max_target=0.998):
        self.mean_max_target = mean_max_target
        super(StopperCallback, self).__init__(
            monitor='',
            patience=float('inf'),
            verbose=1,
            mode='max',
            baseline=self.mean_max_target
        )

    def on_epoch_begin(self, epoch, logs=None):
        print(
            'mean max of probabilities:',
            self.get_monitor_value(logs),
            '- temperature',
            K.get_value(self.model.get_layer('concrete_select').temp)
        )

    def get_monitor_value(self, logs):
        monitor_value = K.get_value(
            K.mean(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis=-1))
        )
        return monitor_value


class ConcreteAutoencoderFeatureSelector:

    def __init__(self, K_features, output_function,
                 num_epochs=300, batch_size=None, learning_rate=0.001,
                 start_temp=10.0, min_temp=0.1, tryout_limit=5):
        self.K = K_features
        self.output_function = output_function
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.tryout_limit = tryout_limit

    def fit(self, X, Y=None, val_X=None, val_Y=None):
        if Y is None:
            Y = X
        assert len(X) == len(Y)

        validation_data = None
        if val_X is not None and val_Y is not None:
            assert len(val_X) == len(val_Y)
            validation_data = (val_X, val_Y)

        if self.batch_size is None:
            self.batch_size = max(len(X) // 256, 16)

        num_epochs = self.num_epochs
        steps_per_epoch = (len(X) + self.batch_size - 1) // self.batch_size

        for _ in range(self.tryout_limit):

            # --- Keras 3 compat: K.set_learning_phase ya no existe ---
            # (no afecta el comportamiento de model.fit)
            inputs = Input(shape=X.shape[1:])

            alpha = math.exp(math.log(self.min_temp / self.start_temp) / (num_epochs * steps_per_epoch))

            self.concrete_select = ConcreteSelect(self.K, self.start_temp, self.min_temp, alpha, name='concrete_select')

            selected_features = self.concrete_select(inputs)
            outputs = self.output_function(selected_features)

            self.model = Model(inputs, outputs)
            self.model.compile(Adam(self.learning_rate), loss='mean_squared_error')

            print(self.model.summary())

            stopper_callback = StopperCallback()

            self.model.fit(
                X, Y,
                self.batch_size,
                num_epochs,
                verbose=1,
                callbacks=[stopper_callback],
                validation_data=validation_data
            )

            if K.get_value(K.mean(K.max(K.softmax(self.concrete_select.logits, axis=-1)))) >= stopper_callback.mean_max_target:
                break

            num_epochs *= 2

        self.probabilities = K.get_value(K.softmax(self.model.get_layer('concrete_select').logits))
        #self.indices = K.get_value(K.argmax(self.model.get_layer('concrete_select').logits, axis=-1))
        self.indices = K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))
        return self

    def get_indices(self):
        return K.get_value(K.argmax(self.model.get_layer('concrete_select').logits, axis=-1))

    def get_mask(self):
        logits = self.model.get_layer('concrete_select').logits
        d = logits.shape[1]
        return K.get_value(K.sum(K.one_hot(K.argmax(logits, axis=-1), d), axis=0))

    def transform(self, X):
        return X[:, self.get_indices()]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices=False):
        return self.get_indices() if indices else self.get_mask()

    def get_params(self):
        return self.model
