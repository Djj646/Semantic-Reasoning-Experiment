import os
from os.path import join, dirname

import torch

print(join(dirname(__file__), '..'))

print("version: "+torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)