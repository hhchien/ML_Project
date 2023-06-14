import torch
import glob 
import pickle
import numpy as np

    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_matrix, labels):
        # Initialization
        self.input_matrix = input_matrix
        self.labels = labels
        
    def __len__(self):
        # Denotes the total number of samples
        return len(self.labels)
    
    def __getitem__(self, index):
        # Generates one sample of data
        a_input_matrix = self.input_matrix[index]
        label = self.labels[index]
        
        return a_input_matrix, label