from abc import ABC, abstractmethod
import os
import pandas as pd
import numpy as np

class BaseDataPreparation(ABC):
    def __init__(self, input_folder, output_folder, label_column):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.label_column = label_column

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def process_data(self):
        pass