# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:24:30 2024

@author: Evans.siaw
"""



import torch
import numpy as np
import torch.nn as nn


import os

# Get the directory of the current Python file
current_directory = os.path.dirname(os.path.abspath(__file__))

#Get path of pretrained model
pretrainedModel = os.path.join(current_directory, "Plane_detector_model.pth")



def findConv2dOutShape(hin,win,conv,pool=2):
    # get conv arguments
    kernel_size = conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    return int(hout),int(wout)

class BrainScanPlaneDetector(nn.Module):
    
    def __init__(self, num_classes = 2):
        super(BrainScanPlaneDetector, self).__init__()
        
        # Convolutional layers
        self.conv0 = nn.Conv2d(3, 64, kernel_size = 3)
        h,w = findConv2dOutShape(256,256,self.conv0)
        self.conv1 = nn.Conv2d(64, 128, kernel_size = 3)
        h,w = findConv2dOutShape(h,w,self.conv1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size = 3)
        h,w = findConv2dOutShape(h,w,self.conv2)
        self.conv3 = nn.Conv2d(256, 512, kernel_size = 3)
        h,w = findConv2dOutShape(h,w,self.conv3)
            
            
        #Max pooling layers
        self.maxPool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        #Fully connected layers
        self.num_flatten=h*w*512
        self.fc1 = nn.Linear(self.num_flatten,512 )
        self.fc2 = nn.Linear(512, num_classes)
        
        #Activation functions
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,x): 
        # Convolutional layers with ReLU activation and max pooling
        x = self.maxPool(self.relu(self.conv0(x)))
        x = self.maxPool(self.relu(self.conv1(x)))
        x = self.maxPool(self.relu(self.conv2(x)))
        x = self.maxPool(self.relu(self.conv3(x)))
        
        # Flatten the output of the convolutional layers
        x = x.view(-1, self.num_flatten)
        
        # Fully connected layers with dropout
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    
    
    

# def load_pretrainedModel():
#     # Load the pretrained model
#     test_model= BrainScanPlaneDetector() 

#     if torch.cuda.is_available():
#         print("Loading Model on GPU")
#         test_model.load_state_dict(torch.load(pretrainedModel))
#         test_model = test_model.cuda()
#     else:
#         print("Loading Model on CPU")
#         test_model.load_state_dict(torch.load(pretrainedModel, map_location=torch.device("cpu")))

#     return test_model.eval()

import gdown
import os



output_path = "brainNet\Plane_detector_model.pth" 

def download_model():
    file_id = "18fAQ62gvI91JKEmwtIORtNsa5--b0qD6"
    url = f"https://drive.google.com/uc?id={file_id}"
    

    if not os.path.exists(output_path):
        print("🔽 Downloading model...")
        gdown.download(url, output_path, quiet=False)

    if os.path.exists(output_path):
        print("✅ Model downloaded.")
        print("📦 Size:", os.path.getsize(output_path) / (1024 * 1024), "MB")
    else:
        print("❌ Download failed.")

def load_pretrainedModel():

    if not os.path.exists(output_path):
        download_model()

    # Initialize the model
    test_model = BrainScanPlaneDetector()

    if torch.cuda.is_available():
        print("Loading Model on GPU")
        test_model.load_state_dict(torch.load(output_path, weights_only=False))
        test_model = test_model.cuda()
    else:
        print("Loading Model on CPU")
        test_model.load_state_dict(torch.load(output_path, map_location=torch.device("cpu"), weights_only=False))

    return test_model.eval()