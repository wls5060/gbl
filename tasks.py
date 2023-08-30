import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
class BaseTask:
    def __init__(self):
        pass

    def _execute(self):
        return NotImplementedError

    def _evaluate(self):
        return NotImplementedError

    def _train(self):
        return NotImplementedError

class NodePrediction(BaseTask) :
    def __init__(self, data, model, epoch, device, )