import cv2
import sys
import time
import torch
import string
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from loss.transformer import Transformer
import torch.nn.functional as F

# ce_loss = torch.nn.CrossEntropyLoss()
from loss.weight_ce_loss import weight_cross_entropy


def to_gray_tensor(tensor):
    pass


def str_filt(str_, voc_type):
    pass

class GWLoss(nn.Module):
    pass
