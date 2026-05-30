from datasets import load_dataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import numpy as np
import random
from tqdm import tqdm
import torch.nn as nn
from torchvision.models import resnet18,ResNet18_Weights,efficientnet_b2,EfficientNet_B2_Weights,MobileNet_V3_Large_Weights,mobilenet_v3_large
import torch.nn.functional as F


class FirstModel(nn.Module):
    def __init__(self, num_landmarks=68):
        super(FirstModel, self).__init__()
        modelres = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            modelres.conv1,
            modelres.bn1,
            modelres.relu,
            modelres.maxpool,
            modelres.layer1,
            modelres.layer2,
            modelres.layer3,
            modelres.layer4,
            modelres.avgpool
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_landmarks * 2)
        )

    def forward(self, x):
        x=self.encoder(x)
        x=torch.flatten(x, 1)
        x=self.head(x)
        return x

class SecondModel(nn.Module):
    def __init__(self, num_landmarks=68):
        super(SecondModel, self).__init__()
        modeleff = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            modeleff.features,
            modeleff.avgpool
        )
        self.head = nn.Sequential(
            nn.Linear(1408, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_landmarks * 2)
        )


    def forward(self, x):
        x = self.encoder(x)
        x=torch.flatten(x, 1)
        x=self.head(x)
        return x

class ThirdModel(nn.Module):
    def __init__(self, num_landmarks=68):
        super(ThirdModel, self).__init__()
        modelnet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            modelnet.features,
            modelnet.avgpool
        )
        self.head = nn.Sequential(
            nn.Linear(960, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_landmarks * 2)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
