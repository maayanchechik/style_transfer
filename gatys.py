import utils
import re
import time
import os 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
from torch.utils.data import Dataset, DataLoader
import random
import transformation_net
import math


move_to_cuda = True

project_dir = '/home/mc/StyleTransfer/'
image_dir = project_dir + 'data/Images/'


################################################################################################
"""# images"""
################################################################################################
#load images, ordered as [style_image, content_image]

imgs, imgs_torch = utils.load_images('vangogh_starry_night.jpg', 'Tuebingen_Neckarfront.jpg', image_dir)
style_image, content_image = imgs_torch

# Display images
if False:
    for img in imgs:
        plt.imshow(img)
        plt.show()
################################################################################################
"""# Using a pre-trained convolutional neural network vgg19"""
################################################################################################
vgg = utils.get_vgg_with_avg_pool(False, move_to_cuda)


################################################################################################
"""# style and content targets and variables"""
################################################################################################
ww = 0.2 #according to Gatys' paper
style_layers_weights = {'0': 1e3/64**2, '5': 1e3/128**2, '10': 1e3/256**2, '19': 1e3/512**2, '28': 1e3/512**2}
content_layer = '21'
style_layers = ['0','5','10','19','28']
content_target_dict = utils.get_features(vgg, content_image)
content_target = content_target_dict[content_layer]
style_target_dict = utils.get_features(vgg, style_image)
style_targets = [utils.GramMatrix()(style_target_dict[layer]) for layer in style_layers]

################################################################################################
"""# training"""
################################################################################################

def style_transfer(vgg_model, content_tensor, style_tensor,
                   alpha, beta, n_iterations, learning_rate):
  '''
  The function runs the style transfer algorithm using a pretrained and freezed vgg_model,
  a content image tensor and style image tensor. It weights the content loss with alpha
  and style loss with beta. It runs for n_iterations.
  '''
  # creating a random image and set requires_grad to True
  #target_image = torch.randn_like(content_tensor).requires_grad_(True).to("cuda")
  opt_image = torch.clone(content_image).requires_grad_(True).to("cuda")

  # create optimizer to optimize the target image 
  optimizer = optim.Adam([opt_image], lr=learning_rate)
  ################### Learning the target image #########################
  for i in range(n_iterations):
    optimizer.zero_grad()
    opt_image_features = utils.get_features(vgg_model, opt_image)
    style_input_features = [opt_image_features[layer] for layer in style_layers]
    content_input_features = opt_image_features[content_layer]

    content_loss = utils.calc_content_loss(content_input_features, content_target)
    style_loss = utils.calc_style_loss(style_layers_weights, style_input_features, style_targets)
    total_loss = alpha * content_loss + beta * style_loss

    total_loss.backward()
    optimizer.step()    

    if i % 50 == 0:
      print(f"Iteration {i}, Total Loss: {total_loss.item():.2f}, Content Loss: {content_loss.item():.2f}, Style Loss {style_loss.item():.2f}, Log Style Loss {math.log(style_loss.item()):.2f}")
  ######################
  
  return opt_image


def main(): 
  stylized_image = style_transfer(vgg, content_image, style_image,
                                  alpha=1, beta=1, n_iterations=1000, learning_rate=1)

  out_img = utils.postp(stylized_image.data[0].cpu().squeeze())
  plt.imshow(out_img)
  plt.gcf().set_size_inches(10,10)
  plt.show()


if __name__ == "__main__":
  main()

