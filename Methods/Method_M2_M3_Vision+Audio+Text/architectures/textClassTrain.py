# -*- coding: utf-8 -*-
"""

@author: Jessica Kick
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import math

from scripts_packages.resultsFileManagement import * #load/save pickle, plot results
from scripts_packages.metricLoss import * #mean average accuracy, average accuracy, mse per trait

from transformers import BertConfig, BertModel, BertTokenizer #huggingface.com BERT

class textFinetuning(torch.nn.Module):
    def __init__(
        self,
        num_classes = 5,
        text_model = BertModel.from_pretrained("bert-base-uncased"), #pretrained BERT base
        input_size = 768,
        dropout_p = 0.2,
        ):
        super(textFinetuning, self).__init__()
        
        self.text_module = text_model
        self.fc = torch.nn.Linear(
            in_features=input_size, 
            out_features=num_classes
        )
        self.dropout = torch.nn.Dropout(dropout_p)
        
    def forward(self, text): 
        text_features = torch.nn.functional.relu(self.text_module(**text).pooler_output)
        
        output_layer = self.dropout(
            text_features
            )

        logits = self.fc(output_layer)
        pred = torch.nn.functional.sigmoid(logits)
        
        return pred