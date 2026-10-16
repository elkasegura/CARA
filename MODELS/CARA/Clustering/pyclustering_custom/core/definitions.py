"""!

@brief Common definition for CCORE.

@authors Andrei Novikov (pyclustering@yandex.ru)
@date 2014-2020
@copyright BSD-3-Clause

"""

# Rimuovi questa riga
# import pyclustering_custom.core as core

import os
import platform
import sys
from sys import platform as _platform

# Path to CCORE library - pyclustering core.
PATH_PYCLUSTERING_CCORE_LIBRARY = None

# Ottieni il percorso base per i file delle librerie
# Otteniamo il percorso assoluto del file corrente
current_file_path = os.path.abspath(__file__)
# Otteniamo la directory del core (risalendo di un livello dalla posizione di questo file)
core_dir = os.path.dirname(current_file_path)

core_architecture = None
if platform.architecture()[0] == "64bit":
    core_architecture = "64-bit"
else:
    core_architecture = "32-bit"

# Costruisci il percorso alla libreria in base alla piattaforma
if (_platform == "linux") or (_platform == "linux2"):
    PATH_PYCLUSTERING_CCORE_LIBRARY = os.path.join(core_dir, core_architecture, "linux", "libpyclustering.so")

elif _platform == "darwin":
    PATH_PYCLUSTERING_CCORE_LIBRARY = os.path.join(core_dir, core_architecture, "macos", "libpyclustering.so")

elif _platform == "win32":
    PATH_PYCLUSTERING_CCORE_LIBRARY = os.path.join(core_dir, core_architecture, "win", "pyclustering.dll")

elif _platform == "cygwin":
    PATH_PYCLUSTERING_CCORE_LIBRARY = os.path.join(core_dir, core_architecture, "win", "libpyclustering.so")

# Stampa informazioni utili per il debug
#print(f"CCORE Path: {PATH_PYCLUSTERING_CCORE_LIBRARY}")
#print(f"File exists: {os.path.exists(PATH_PYCLUSTERING_CCORE_LIBRARY) if PATH_PYCLUSTERING_CCORE_LIBRARY else False}")