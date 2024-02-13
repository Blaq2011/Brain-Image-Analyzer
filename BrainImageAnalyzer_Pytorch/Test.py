
import torch
from PIL import Image
import torchvision.transforms as transforms 
from brainNet import load_pretrainedModel




def predict_plane(imgpath):
    
    test_model = load_pretrainedModel()

    # Transformation for the input image
    transform = transforms.Compose(
        [
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
       ]
    )

    # Load and preprocess the image
    image_path = imgpath
    image = Image.open(image_path)


    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

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



    # Assuming binary classification (0 or 1)
    predictedClass = ""
    if predicted.item() == 0:
        print("Predicted class: Axial(0)")
        predictedClass = "Axial"
    else:
        print("Predicted class: Sagittal(1)")
        predictedClass = "Sagittal"
    return predictedClass




def get_imagePath():
    imgPath= input("Enter Image Path: ")
    predict_plane(imgPath)




if __name__ == "__init__":
    get_imagePath()


