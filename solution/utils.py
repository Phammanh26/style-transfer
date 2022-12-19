from PIL import Image
import numpy as np
from io import BytesIO
import requests
import config
import cv2
import torch
from torchvision import transforms


def load_image(img_path, max_size=400, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image


def process_image(image: bytes, max_size=400, shape=None) -> torch.Tensor:
    '''
    : func: process image from bytes to tensor
    : max_size: max size image
    : shape: shape of image, example width = 400, heigh = 300 -> (400x300)
    : output: tensor with shape (batch, 3, width, heigth)
    '''
    image = Image.open(BytesIO(image)).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image


# helper function for un-normalizing an image 
# and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    """ Display a tensor as an image. """
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    # Returns a cv2 image array from the document vector
    _, image = cv2.imencode(".png", image)
    
    return image


def load_data(content_image_path, style_image_path):
    # load in content and style image
    content = load_image(content_image_path).to(config.device)
    # Resize style to match content, makes code easier
    style = load_image(style_image_path, shape=content.shape[-2:]).to(config.device)