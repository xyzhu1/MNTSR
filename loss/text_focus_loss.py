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

class TextFocusLoss(nn.Module):
    def __init__(self, args):
        super(TextFocusLoss, self).__init__()
       
    def forward(self,sr_img, hr_img, label):
        pass


