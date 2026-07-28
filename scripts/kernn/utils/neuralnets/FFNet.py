# torch imports
import torch
import torch.nn as nn


class FFNet(nn.Module):
  '''
    Simple Feed-Forward Neural Network
  '''
  def __init__(self, n_input, n_hidden, n_out):
    super().__init__()
    self.layers = nn.Sequential(nn.Linear(n_input, n_hidden),
                      nn.Softplus(),
                      nn.Linear(n_hidden, n_hidden),
                      nn.Softplus(),
                      nn.Linear(n_hidden, n_hidden),
                      nn.Softplus(),
                      nn.Linear(n_hidden, n_out))


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)

