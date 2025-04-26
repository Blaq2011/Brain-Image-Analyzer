
"""
Created on Tue Feb 13 20:24:30 2024

@author: Evans.siaw
"""

import torch
import numpy as np
import torch.nn as nn
import os
from huggingface_hub import hf_hub_download

# Get the directory of the current Python file
current_directory = os.path.dirname(os.path.abspath(__file__))

# Download model from Hugging Face
model_path = hf_hub_download(
    repo_id="blaq101/Brain-Image-Analyzer",
    filename="model1.pth"
)

def findConv2dOutShape(hin, win, conv, pool=2):
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    hout = np.floor((hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    wout = np.floor((win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    if pool:
        hout /= pool
        wout /= pool
    return int(hout), int(wout)

class BrainScanPlaneDetector(nn.Module):
    def __init__(self, num_classes=3):  # Only train on 3 classes
        super(BrainScanPlaneDetector, self).__init__()
        
        # Convolutional layers (reduced channels)
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, padding=1)     # 3 -> 16
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)    # 16 -> 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)    # 32 -> 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)   # 64 -> 128

        self.maxPool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Assuming input image is 256x256
        # After 4x pooling: 256 -> 128 -> 64 -> 32 -> 16
        self.num_flatten = 128 * 16 * 16

        self.fc1 = nn.Linear(self.num_flatten, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.maxPool(self.relu(self.conv0(x)))
        x = self.maxPool(self.relu(self.conv1(x)))
        x = self.maxPool(self.relu(self.conv2(x)))
        x = self.maxPool(self.relu(self.conv3(x)))
        
        x = x.view(-1, self.num_flatten)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


def load_pretrainedModel():
    test_model = BrainScanPlaneDetector()
    print("✅ Loading model on CPU")
    test_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
    
    return test_model.eval()
