import warnings

warnings.filterwarnings("ignore")

import os
import sys
import math
import numpy as np
from pathlib import Path

import torch
from torch import nn
from torch.nn import Module
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from hybrid_model_for_russian_sentiment_analysis import logger
from hybrid_model_for_russian_sentiment_analysis.constants import *
from hybrid_model_for_russian_sentiment_analysis.models.head import HeadModel
from hybrid_model_for_russian_sentiment_analysis.entity.dataset import CustomDataset
from hybrid_model_for_russian_sentiment_analysis.models.trio import MHSAParallelConvRecModel
from hybrid_model_for_russian_sentiment_analysis.utils.common import clear_vram, read_yaml, load_pkl


def str_to_class(classname: str):
    return getattr(sys.modules[__name__], classname)


class CustomHybridModel:
    """
    Class for making predictions using trained custom hybrid model.

    Attributes:
    - config (dict): Configuration settings.
    - params (list[dict]): List of dictionaries with parameters for the head models.
    - device (str): Device on which calculations will be performed
    - verbose (bool): Whether to print and save logs during calculation.
    """

    def __init__(self,
                 device: str = 'cuda',
                 verbose: bool = True):
        """
        Initializes CustomHybridModel.

        Args:
        - device (str): Device on which calculations will be performed
        - verbose (bool): Whether to print and save logs during calculation.
        """

        self.config = read_yaml(Path(CONFIG_FILE_PATH), verbose=verbose)
        self.params = {}
        for head_model in self.config.head_models:
            self.params[head_model] = read_yaml(Path(os.path.join(PARAMS_FILE_PATH, f"{head_model}.yaml")),
                                                verbose=verbose)
        self.device = device
        self.verbose = verbose

    def load_tokeniser(self) -> object:
        """
        Function to load the tokeniser

        Returns:
        - tokeniser (object): Tokeniser's object.
        """

        return AutoTokenizer.from_pretrained(self.config.model_checkpoint, **self.config.tokeniser_loader_parameters)

    def load_embedding_model(self) -> object:
        """
        Function to load an embedding model

        Returns:
        - model (object): Model's object.
        """

        return AutoModel.from_pretrained(self.config.model_checkpoint, num_labels=2).to(self.device)

    def load_head_model(self, head_model_name: str) -> Module:
        """
        Loads head model.

        Parameters:
        - head_model_name (str): Name of file with a head model config.

        Returns:
        - model (Module): Head model object.
        """

        model = HeadModel(main_model=str_to_class(self.params[head_model_name].main_model_class),
                          output_size=self.params[head_model_name].params['hidden_layers'][-1],
                          n_targets=2,
                          main_model_kwargs=self.params[head_model_name].params)

        model._modules['main_model'].load_state_dict(
            torch.load(os.path.join(WEIGHTS_FILE_PATH, f'{self.params[head_model_name].name}_main_model.pt'), map_location=self.device))
        model._modules['ffnn'].load_state_dict(
            torch.load(os.path.join(WEIGHTS_FILE_PATH, f'{self.params[head_model_name].name}_ffnn.pt'), map_location=self.device))

        model.eval()
        model = model.to(self.device)

        return model

    def load_second_level_model(self):
        """
        Loads second level model

        Returns:
        - model: Second level model
        """

        return load_pkl(Path(os.path.join(WEIGHTS_FILE_PATH, f'{self.config.second_level_model}.pkl')), verbose=self.verbose)

    def tokenise(self, data: list[str]) -> CustomDataset:
        """
        Function to tokenise input dataset

        Parameters:
        - data (list[str]): Dataset to be processed

        Returns:
        - output (CustomDataset): Tokenised data
        """

        tokeniser = self.load_tokeniser()
        output = tokeniser(data, **self.config.tokeniser_parameters)
        output = CustomDataset(input_ids=output['input_ids'], attention_mask=output['attention_mask'])

        if self.verbose: logger.info("Data has been tokenised")

        return output

    def predict_on_tokens(self,
                          data: CustomDataset,
                          return_probabilities: bool = False) -> np.ndarray:
        """
        Makes predictions via the hybrid model for the tokenised data.

        Parameters:
        - data (CustomDataset): Dataset to be processed

        Returns:
        - predictions (np.ndarray): Array with predicted labels or probabilities
        """

        # Loading the embedding model
        embedding_model = self.load_embedding_model()
        embedding_model.eval()

        # Loading the first level models (head models)
        head_models = [self.load_head_model(head_model_name=name) for name in self.config.head_models]

        # Loading the second level model
        second_level_model = self.load_second_level_model()

        # Creating batch generator and tqdm iterator
        batch_generator = torch.utils.data.DataLoader(dataset=data, batch_size=self.config.batch_size, shuffle=False)
        n_batches = math.ceil(len(data) / batch_generator.batch_size)
        if self.verbose:
            iterator = tqdm(enumerate(batch_generator), desc='batch', leave=True, total=n_batches)
        else:
            iterator = enumerate(batch_generator)

        # Calculating predictions
        with torch.no_grad():

            predictions = None

            for it, (batch_ids, batch_masks) in iterator:
                # Moving tensors to specified device
                batch_ids = batch_ids.to(self.device)
                batch_masks = batch_masks.to(self.device)

                # Calculating embeddings
                batch_output = embedding_model(input_ids=batch_ids, attention_mask=batch_masks).last_hidden_state.to(torch.float32)

                # Converting data to EmbeddingsDataset format
                # batch_output = EmbeddingsDataset(X=batch_output)

                # Calculating first level predictions (logits) for each head model
                logits = []
                for k in range(len(head_models)):
                    logits_ = head_models[k](batch_output)[:, 0].to('cpu')
                    logits.append(logits_.detach().tolist())

                # Transposing the array with logits
                logits = list(map(list, zip(*logits)))

                # Clearing VRAM cache
                if self.device == 'cuda': clear_vram()
                del batch_output

                # Calculating second level prediction
                if return_probabilities:
                    predictions_ = second_level_model.predict_proba(logits, verbose=-1)
                    # Merging outputs
                    predictions = predictions_ if predictions is None else np.vstack([predictions, predictions_])
                else:
                    predictions_ = second_level_model.predict(logits, verbose=-1)
                    # Merging outputs
                    predictions = predictions_ if predictions is None else np.hstack([predictions, predictions_])

        if self.verbose: logger.info("Predictions have been made")

        # Deleting loaded models
        del embedding_model
        del head_models
        del second_level_model

        return predictions


    def predict(self, data: list[str]) -> np.ndarray:
        """
        Makes predictions via the hybrid model for the preprocessed data.

        Parameters:
        - data (list[str]): Dataset to be processed

        Returns:
        - data (np.ndarray): Array with predicted labels
        """

        data = self.tokenise(data=data)

        data = self.predict_on_tokens(data=data, return_probabilities=False)

        return data


    def predict_proba(self, data: list[str]) -> np.ndarray:
        """
        Calculates probabilities via the hybrid model for the preprocessed data.

        Parameters:
        - data (list[str]): Dataset to be processed

        Returns:
        - data (np.ndarray): Array with predicted probabilities
        """

        data = self.tokenise(data=data)

        data = self.predict_on_tokens(data=data, return_probabilities=True)

        return data