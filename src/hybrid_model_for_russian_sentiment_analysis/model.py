import warnings

warnings.filterwarnings("ignore")

import os
import sys
import math
from pathlib import Path

import torch
import pandas as pd
from torch import nn
from torch.nn import *
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from hybrid_model_for_russian_sentiment_analysis import logger
from hybrid_model_for_russian_sentiment_analysis.constants import *
from hybrid_model_for_russian_sentiment_analysis.models.head import HeadModel
from hybrid_model_for_russian_sentiment_analysis.models.trio import MHSAParallelConvRecModel
from hybrid_model_for_russian_sentiment_analysis.utils.common import clear_vram, read_yaml, load_pkl
from hybrid_model_for_russian_sentiment_analysis.entity.dataset import CustomDataset, EmbeddingsDataset


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

        tokeniser = AutoTokenizer.from_pretrained(self.config.model_checkpoint,
                                                  **self.config.tokeniser_loader_parameters)

        return tokeniser

    def load_embedding_model(self) -> object:
        """
        Function to load an embedding model

        Returns:
        - model (object): Model's object.
        """

        model = AutoModel.from_pretrained(self.config.model_checkpoint, num_labels=2).to(self.device)
        model.eval()

        return model

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
            torch.load(os.path.join(WEIGHTS_FILE_PATH, f'{self.params[head_model_name].name}_main_model.pt')))
        model._modules['ffnn'].load_state_dict(
            torch.load(os.path.join(WEIGHTS_FILE_PATH, f'{self.params[head_model_name].name}_ffnn.pt')))

        model.eval()
        model = model.to(self.device)

        return model

    def load_second_level_model(self):
        """
        Loads second level model

        Returns:
        - model: Second level model
        """

        model = load_pkl(WEIGHTS_FILE_PATH, f'{self.config.second_level_model}.pkl')
        return model

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

        return output

    def calculate_embeddings(self, data: CustomDataset):
        """
        Calculates embedding using embedding_model

        Parameters:
        - data (CustomDataset): Dataset to be processed

        Returns:
        - output (EmbeddingsDataset): Calculated embbeddings
        """

        # Loading model
        embedding_model = self.load_embedding_model()

        # Creating batch generator and tqdm iterator
        batch_generator = torch.utils.data.DataLoader(dataset=data, batch_size=self.config.batch_size, shuffle=False)
        n_batches = math.ceil(len(data) / batch_generator.batch_size)

        if self.verbose:
            iterator = tqdm(enumerate(batch_generator), desc='batch', leave=True, total=n_batches)
        else:
            iterator = enumerate(batch_generator)

        # Encoding data
        with torch.no_grad():

            output = None

            for it, (batch_ids, batch_masks) in iterator:
                # Moving tensors to specified device
                batch_ids = batch_ids.to(self.device)
                batch_masks = batch_masks.to(self.device)

                # Getting embeddings
                batch_output = embedding_model(input_ids=batch_ids, attention_mask=batch_masks).last_hidden_state.to(
                    'cpu').to(torch.float32)

                # Merging outputs
                output = batch_output if output == None else torch.cat([output, batch_output], axis=0)

        output = EmbeddingsDataset(X=output)

        del embedding_model

        if self.verbose: logger.info('Embeddings have been calculated')

        return output

    def calculate_logits(self,
                         head_model_name: str,
                         head_model: Module,
                         data: EmbeddingsDataset) -> torch.Tensor:
        """
        Function to calculate logits for 0 class using head model on the whole data.

        Parameters:
        - head_model (Module): Head model.
        - batch_size (int): Batch size.
        - data (EmbeddingsDataset): Embeddings.

        Returns:
        - logits (torch.Tensor): Logits for 0 class
        """

        # Creating empty list to store logits
        logits = []

        # Creating batch generator and tqdm iterator
        batch_generator = torch.utils.data.DataLoader(dataset=data, batch_size=self.params[
            head_model_name].training_configs.batch_size, shuffle=False)
        n_batches = math.ceil(len(data) / batch_generator.batch_size)

        if self.verbose:
            iterator = tqdm(enumerate(batch_generator), desc='batch', leave=True, total=n_batches)
        else:
            iterator = enumerate(batch_generator)

        # Calculating logits for each batch of embeddings
        for batch_X in batch_generator:
            # Moving tensors to specified device
            batch_X = batch_X.to(self.device)

            with torch.no_grad():
                # Calculating logits on the batch
                batch_logits = head_model(batch_X)
                batch_logits = batch_logits[:, 0].to('cpu')

            # Appending calculated logits to the list with all logits
            logits.append(batch_logits)

        # Merging list with logits to a single Tensor
        logits = torch.cat(logits, dim=0)

        if self.verbose: logger.info(f'Logits have been calculated for {head_model_name}')

        # Returning logits
        return logits

    def make_first_level_predictions(self, data: EmbeddingsDataset) -> pd.DataFrame:
        """
        Calculates first level predictions (logits) for each head model

        Parameters:
        - data (EmbeddingsDataset): Data with calculated embeddings

        Returns:
        - all_logits (pd.DataFrame): DataFrame with calculated logits
        """

        all_logits = pd.DataFrame()

        if self.verbose:
            iterations = tqdm(self.config.head_models,
                              desc='head_models',
                              leave=True,
                              total=len(self.config.head_models))
            iterations.set_postfix({'HEAD_MODEL': None})
        else:
            iterations = self.config.head_models

        for head_model_name in iterations:

            if self.verbose: iterations.set_postfix({'HEAD_MODEL': head_model_name})

            # Loading head model
            head_model = self.load_head_model(head_model_name=head_model_name)

            # Calculating logits
            logits = self.calculate_logits(head_model_name=head_model_name, head_model=head_model, data=data)
            logits = logits.detach().numpy()
            all_logits[head_model_name] = logits

            # Deleting model
            del head_model

            # Clearing VRAM cache
            clear_vram()

        if self.verbose: logger.info(f'First level predictions have been made')

        return all_logits

    def make_second_level_prediction(self,
                                     data: pd.DataFrame,
                                     return_probabilities: bool = False) -> list[int]:
        """
        Calculates second level prediction using second level model

        Parameters:
        - data (pd.DataFrame): Data with first level predictions (logits)

        Returns:
        - predictions (list): List with predicted labels
        """

        model = self.load_second_level_model()

        if return_probabilities:
            predictions = model.predict_proba(data.values)
        else:
            predictions = model.predict(data.values)

        if self.verbose: logger.info(f'Second level prediction have been made')

        return predictions

    def predict_on_tokens(self,
                          data: CustomDataset,
                          return_probabilities: bool = False) -> list[int]:
        """
        Makes predictions via the hybrid model for the tokenised data.

        Parameters:
        - data (CustomDataset): Dataset to be processed

        Returns:
        - data (list[int]): List with predicted labels
        """

        data = self.calculate_embeddings(data=data)

        data = self.make_first_level_predictions(data=data)

        data = self.make_second_level_prediction(data=data, return_probabilities=return_probabilities)

        return data

    def predict(self, data: list[str]) -> list[int]:
        """
        Makes predictions via the hybrid model for the preprocessed data.

        Parameters:
        - data (list[str]): Dataset to be processed

        Returns:
        - data (list[int]): List with predicted labels
        """

        data = self.tokenise(data=data)

        data = self.predict_on_tokens(data=data)

        return data

    def predict_proba(self, data: list[str]) -> list[float]:
        """
        Calculates probabilities via the hybrid model for the preprocessed data.

        Parameters:
        - data (list[str]): Dataset to be processed

        Returns:
        - data (list[int]): List with predicted probabilities
        """

        data = self.tokenise(data=data)

        data = self.predict_on_tokens(data=data, return_probabilities=True)

        return data