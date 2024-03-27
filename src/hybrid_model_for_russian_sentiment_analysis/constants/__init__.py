import os
from pathlib import Path

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

CONFIG_FILE_PATH = Path(os.path.join(PACKAGE_DIR, "config/config.yaml"))
PARAMS_FILE_PATH = Path(os.path.join(PACKAGE_DIR, "params"))
WEIGHTS_FILE_PATH = Path(os.path.join(PACKAGE_DIR, "weights"))