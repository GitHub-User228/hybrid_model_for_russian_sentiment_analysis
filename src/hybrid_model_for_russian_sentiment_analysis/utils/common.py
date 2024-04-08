import os
import yaml
import joblib
from pathlib import Path

from box import ConfigBox
from box.exceptions import BoxValueError

import gc
from torch import cuda

from hybrid_model_for_russian_sentiment_analysis import logger



def read_yaml(path_to_yaml: Path, verbose: bool = True) -> ConfigBox:
    """
    Reads a yaml file, and returns a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the yaml file.
        verbose (bool): Whether to show logger's messages.

    Raises:
        ValueError: If the yaml file is empty.
        e: If any other exception occurs.

    Returns:
        ConfigBox: The yaml content as a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if verbose: logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        logger.info("Value exception: empty yaml file")
        raise ValueError("yaml file is empty")
    except Exception as e:
        logger.info(f"An exception {e} has occurred")
        raise e



def clear_vram():
    cuda.empty_cache()
    gc.collect()



def load_pkl(path_to_pkl: Path, verbose: bool=True):
    """
    Loads pkl file

    Parameters:
    - path (str): Path from where to load model
    - filename (str): Filename of pkl file.

    Returns:
    - model: Loaded pkl file
    """
    try:
        content = joblib.load(path_to_pkl)
        if verbose: logger.info(f"pkl file: {path_to_pkl} loaded successfully")
        return content
    except Exception as e:
        logger.info(f"An exception {e} has occurred")
        raise e
