from io import BytesIO
from PIL import Image
import cv2
import torch
from torchvision.transforms import transforms
from app.model import Convnet
import numpy as np
import timeit as time   
import os
import io

PATH = "./app/checkpoint_2.pth"
device = torch.device('cpu')
maskclasses = ["maskoff", "maskon"]

def read_imagefile(file) -> Image.Image:
  image = Image.open(BytesIO(file))

  return image

def pre_processing(image: Image.Image):
  img_transforms = transforms.Compose([transforms.Resize((160, 160)),transforms.ToTensor()])
  img = img_transforms(Image.fromarray(np.uint8(image)))
  img_tensor = img.unsqueeze(0)
        
  return img_tensor

# def resize_image(image:Image.Image):
#    img = fn.resize(image, size=[500,500])
#    return img

def predict(image_tensor: int, start: float):

  model = Convnet() 
  model.load_state_dict(torch.load(PATH, map_location=device))
  model.eval()

  output = model(image_tensor)
  prob = torch.sigmoid(output)
  pred = int(torch.round(prob))
  end = time.timeit()
  timetaken = end - start

  response = []
  resp = {}
  resp["Response time"] = str(abs(round(timetaken,3))) + " seconds"
  resp["class"] = pred 
  resp["category"] = maskclasses[pred]
  resp["confidence"] = str(round((float(prob)*100),2)) + "%"
  response.append(resp)

  return response

def classifier(image_tensor: int):

  model = Convnet()
  model.load_state_dict(torch.load(PATH, map_location=device))
  model.eval()

  output = model(image_tensor)
  prob = torch.sigmoid(output)
  pred = int(torch.round(prob))

  return pred

def prediction(binary_image):

  #print((binary_image))
  input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
  #print(input_image)
  img = np.uint8(input_image)
  #print(img)

  prepare_image_seq = transforms.Compose([transforms.Resize((160, 160)),transforms.ToTensor()])
  face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

  grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = face_cascade.detectMultiScale(
    grey_image,
    scaleFactor=1.1,
    minNeighbors=11
  )
  response = []

  for x, y, w, h in faces:

    face_to_analyze = img[y-10:y+h+10, x-10:x+w+10]
    face_to_analyze = Image.fromarray(np.uint8(face_to_analyze))
    face_prepared = prepare_image_seq(face_to_analyze)
    
    pred = classifier(face_prepared.unsqueeze(0))

    if pred == 1:
      label = "Mask"
    else:
      label = "No Mask"

    resp = {}
    resp["label"] = label
    resp["x"] = str(x)
    resp["y"] = str(y)
    resp["w"] = str(w)
    resp["h"] = str(h)
    response.append(resp)
  
  return response
