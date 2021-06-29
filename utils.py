
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
import pickle

def get_vgg_with_avg_pool(prt, move_to_cuda):

    print("Get models.vgg19")
    vgg = models.vgg19(pretrained=True)
    if prt:
        print(vgg)

    # Freeze network paramters. We do not want to train the network paramaters any
    # more, only to train the stylized image.
    for param in vgg.parameters():
        param.requires_grad = False

    # According to leon Gatys' paper avg pooling produces better results than max pooling
    print("Replaxe max with avg pooling")
    for i,layer in enumerate(vgg.features):
        if isinstance(layer, torch.nn.MaxPool2d):
            vgg.features[int(i)] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
    # move the model to the GPU
    if move_to_cuda:
        vgg = vgg.to("cuda").eval()

    if prt:
        print(vgg)

    return vgg


#functions that prepare the images
img_size = 512 
prep = transforms.Compose([transforms.Resize(img_size),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                            std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),])

postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                             transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                      std=[1,1,1]),
                             transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                             ])
postpb = transforms.Compose([transforms.ToPILImage()])

    
def postp(tensor,WITH_CHANGES): # to clip results in the range [0,1]
    if WITH_CHANGES:
        t = postpa(tensor)
    else:
        t = tensor
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img

def load_images(style_image_name, content_image_name,image_dir):
    img_names = [style_image_name, content_image_name]
    imgs = [Image.open(image_dir + img_names[0]),Image.open(image_dir + img_names[1])]
    imgs_torch = [prep(img) for img in imgs]
    if torch.cuda.is_available():
        imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
    else:
        imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
#    style_image, content_image = imgs_torch

    # opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True) #random init
#    opt_img = Variable(content_image.data.clone(), requires_grad=True)
    return imgs, imgs_torch


def get_features(net, image_tensor):
    ''' 
    The function runs a forward pass given an image and extracts the features for 
    several conv layers. It returns a dictionary where the keys are the
    layers name and the values are the features.
    '''
    layers_idx = ['0','5','10','19','21','28']
    features = {}
    cur_output = image_tensor
    for i,layer in enumerate(net.features):
      cur_output = layer(cur_output)
      if str(i) in layers_idx:
        features[str(i)] = cur_output
    ######################
    return features

# gram matrix and style loss
#the GramMatrix class inherites from Module, where it is defined that if an
#instance is called then the function forward is returned.
class GramMatrix(nn.Module):
    def forward(self, input_features):
        b,c,h,w = input_features.size()
        F = input_features.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w) #?
#        print("G.size()",G.size())
        return G

class GramMSELoss(nn.Module):
    # This function receives an input vector which is an activation map, and
    # a target vector which is the target grammatrix. The target is already a 
    # gram matrix for efficiency, since it is a constant so should only be 
    # calculated once.
    def forward(self, input_features, target_gram_matrix):
#       mseloss = nn.MSELoss()
#       gram_matrix_forward = GramMatrix() #GramMatrix() returns the function forward
#       computed_input = gram_matrix_forward(input)
#       out = mseloss(computed_input, target) 
        out = nn.MSELoss()(GramMatrix()(input_features), target_gram_matrix)
        return out


def calc_style_loss(layers_weights, input_features, target_gram_matricies):
  loss = 0
  for i,layer in enumerate(layers_weights):
    loss += layers_weights[layer] * GramMSELoss()(input_features[i], target_gram_matricies[i])
  return loss

def calc_content_loss(input_features, target_features):
  loss = nn.MSELoss()(input_features, target_features)
  return loss

def show_stylized_image(stylized_img,i,dir_name):
    out_img = postp(stylized_img.data[0].cpu().squeeze(),True)
#    plt.imshow(out_img)
#    plt.gcf().set_size_inches(10,10)
#    plt.show()
    path_name = "/home/mc/StyleTransfer/"+dir_name
    if not os.path.exists(path_name):
        os.mkdir(path_name)
    final_path_name = path_name+"/pic"+str(i)+".png"
    out_img.save(final_path_name)
    return final_path_name
    
def save_loss_file(file_name,train_losses):
    loss_file = open(file_name,"wb")
    pickle.dump(train_losses,loss_file)
    loss_file.close()
