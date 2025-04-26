# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:24:30 2024
@author: Evans.siaw
"""

# import os
# import psutil
import torch
from PIL import Image
from torchvision import transforms
from brainNet import load_pretrainedModel

# # --- Utility: Monitor Memory Usage ---
# def get_memory_usage():
#     process = psutil.Process(os.getpid())
#     memory_info = process.memory_info()
#     print(f"Memory Usage: {memory_info.rss / 1024**2:.2f} MB")

# # --- Step 1: Initial Memory ---
# get_memory_usage()

# --- Step 2: Load Model Once Globally ---
test_model = load_pretrainedModel()
device = next(test_model.parameters()).device

# --- Step 3: Check Memory After Loading Model ---
# get_memory_usage()

# --- Step 4: Image Transformations (Only Once) ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Step 5: Prediction Function ---
def predict_plane(img_path, threshold=0.8):
    """Predict brain scan plane from image."""

    # Preprocess the input image
    image = Image.open(img_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    test_model.eval()
    with torch.no_grad():
        outputs = test_model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_confidence = torch.nn.functional.softmax(outputs, dim=1)[0, predicted].item()
    
    # Handle low-confidence predictions
    if class_confidence < threshold:
        return "Unknown Image / Unclear scan", class_confidence

    classes = ["Axial", "Coronal", "Sagittal"]
    return classes[predicted.item()], class_confidence

