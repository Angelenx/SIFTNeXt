__version__ = '0.0.1'
__author__ = 'Jinquan Lu'
import torch
import torchvision
import numpy as np
from torch import nn
from . import Modules as SM

class SIFTNeXt(nn.modules):
    def __init__(self):
        super(SIFTNeXt,self).__init__()
        self.sift_layer = SM.SIFT_DOG()





__all__ = [torch,torchvision,nn,SIFTNeXt]