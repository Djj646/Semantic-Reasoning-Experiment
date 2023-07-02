import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.autograd import grad
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def heatmap(img,nav,model,alpha):
    # image process
    img_height = 128
    img_width = 256
    _transforms = [
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
     
    transform = transforms.Compose(_transforms)

    img_ = transform(Image.fromarray(img)).to(device)
    nav_ = transform(Image.fromarray(nav)).to(device)

    inputs = torch.cat([img_,nav_],dim=0)
    inputs = inputs.view(1,6,img_height,img_width)
    inputs.requires_grad_(True) # Need gradient
    result,_ = model(inputs) # tensor(1,1,128,256)

    gradients = grad(result.sum(), inputs, create_graph=True)[0]
    # print(gradients.shape)

    gradients = gradients.view(6,img_height, img_width).detach().cpu().numpy()

    img_gradient = gradients[:3,:,:].sum(axis=0)
    nav_gradient = gradients[4:,:,:].sum(axis=0)
    
    gradients = np.vstack((img_gradient,nav_gradient))
    
    # gradients = np.exp(gradients) / np.sum(np.exp(gradients), axis=0)
    gradients = np.abs(gradients)
    gradients -= np.max(np.min(gradients),0)
    gradients /= np.max(gradients)
    gradients = (gradients*255).astype(np.uint8)

    heat = cv2.applyColorMap(gradients,cv2.COLORMAP_JET)
    # print(img.shape)
    # print(nav.shape)

    
    orign = np.vstack((img,nav))

    heat = alpha*heat + (1-alpha)*orign

    return heat