
import utils
from gatys import style_transfer
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

move_to_cuda = True

project_dir = '/home/mc/StyleTransfer/'
image_dir = project_dir + 'data/Images/'

def main():
    ################################################################################################
    """# images"""
    ################################################################################################
    #load images, ordered as [style_image, content_image]
    
    imgs, imgs_torch = utils.load_images('modern.jpg', 'pic3.jpg',
                                         image_dir)
    style_image, content_image = imgs_torch

    # Display images
    if False:
        for img in imgs:
            plt.imshow(img)
            plt.show()
    ########################################################################################
    """# Using a pre-trained convolutional neural network vgg19"""
    ########################################################################################
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
    """# A network that learns to style transfer"""
    ################################################################################################
    print("using transformation net")
    from transformation_net import TransformerNet
    
    model_path = "/home/mc/StyleTransfer/model_2906_1725_modern_gatys_transforms_batch_size1_a001_b0.001_lr0.001reg1e-12.model"
    
    style_model = TransformerNet()
    state_dict = torch.load(model_path)
    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    style_model.load_state_dict(state_dict)
    style_model.to("cuda")
            
    start_time = time.time()
    output_of_saved_model = style_model(content_image)
    fast_style_transfer_time = time.time()-start_time
    
    fast_out_img = utils.postp(output_of_saved_model.data[0].cpu().squeeze(),True)
    plt.imshow(fast_out_img)
    plt.gcf().set_size_inches(10,10)
    plt.show()
    
    ###############################################################################################
    #Gatys
    ###############################################################################################
    print("using gatys")
    start_time = time.time()
    stylized_image = style_transfer(vgg, content_image, style_targets, style_layers,
                                    style_layers_weights, content_layer, content_target,
                                    alpha=1, beta=1, n_iterations=1000, learning_rate=1)

    normal_style_transfer_time = time.time()-start_time
    normal_out_img = utils.postp(stylized_image.data[0].cpu().squeeze(),True)
    plt.imshow(normal_out_img)
    plt.gcf().set_size_inches(10,10)
    plt.show()
    
    print("The fast style transfer styles the image within:",fast_style_transfer_time,"seconds")
    print("The normal style transfer styles the image within:",normal_style_transfer_time,"seconds")


if __name__ == "__main__":
    main()
