# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:24:30 2024

@author: Evans.siaw
"""


from brainNet.brainNet import load_pretrainedModel
import torch
from PIL import Image
import torchvision.transforms as transforms 


import torch.nn.functional as F





def predict_plane(imgpath):
    
    test_model = load_pretrainedModel()

    # Transformation for the input image
    transform = transforms.Compose(
        [
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            # transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
       ]
    )

    # Load and preprocess the image
    image_path = imgpath
    image = Image.open(image_path).convert('RGB')
    
    convert_tensor = transforms.ToTensor()
    image_tensor = convert_tensor(image)
    print(image_tensor.shape)
    
    
    if image_tensor.shape[0] == 1:
        image_tensor = image_tensor.unsqueeze(1).expand(-1, 3, -1, -1) #convert image to have 3 channels
        image_tensor = transform(image_tensor) 
    else:
        image_tensor = transform(image_tensor).unsqueeze(0) 
   
    
    
    

    if torch.cuda.is_available():
        print("Loading image(s) on GPU")
        image_tensor = image_tensor.cuda()
    else:
        print("Loading image(s) on CPU")
        device = torch.device("cpu")
        image_tensor = image_tensor.to(device)


    # Make prediction
    with torch.no_grad():
        outputs = test_model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence_scores = F.softmax(outputs, dim=1)


    confidence_of_predictedClass = ""

    # Assuming binary classification (0 or 1)
    predictedClass = ""
    if predicted.item() == 0:
        confidence_of_predictedClass =  confidence_scores[0, predicted[0]].item()
        print(f"Predicted class: Axial(0) | Confidence: {confidence_of_predictedClass}")
        predictedClass = "Axial"
        
    else:
        confidence_of_predictedClass =  confidence_scores[0, predicted[0]].item()
        print(f"Predicted class: Sagittal(1) | Confidence: {confidence_of_predictedClass}")
        predictedClass = "Sagittal"
       
        
        
    return predictedClass, confidence_of_predictedClass




def main():
    imgPath= input("Enter Image Path: ")
    predict_plane(imgPath)




if __name__ == '__main__':
    main()
