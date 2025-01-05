# -*- coding: utf-8 -*-
"""Bagroot Project Style Transfer

# setup
"""
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


project_dir = '/home/mc/StyleTransfer/'
image_dir = project_dir + 'data/Images/'
model_dir = project_dir + 'data/Models/'
mini_dataset = project_dir + 'data/coco_dataset'
whole_dataset = project_dir + 'data/val2017'


seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
move_to_cuda = True
batch_size_ = 1 


################################################################################################
"""#pretraining the net to be the identity network"""
################################################################################################

def pre_train(optimizer, model, nepochs, train_loader,dir_name,saved_test_images_counter,content_image):
    pre_train_losses = []
    for e in range(nepochs):
        model.train()
        for batch_index,(content_batch,y_batch) in enumerate(train_loader):
            batch_len = len(content_batch)
            content_batch = content_batch.to("cuda")
            stn_output = model(content_batch) # the outputs of the whole batch
            loss = nn.MSELoss()(stn_output,content_batch)

            if batch_index%10 == 0:
                print("batch index ",batch_index,"\n\tloss: ", loss)

            if batch_index%150==0:
                test_image = model(content_image)
                final_path_name = utils.show_stylized_image(
                    test_image, saved_test_images_counter, dir_name)
                print("\n\nSave a test output in ", final_path_name, "\n\n")
                saved_test_images_counter += 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pre_train_losses.append(loss.item())
    return pre_train_losses

################################################################################################
"""# training"""
################################################################################################
import math
def train_model(optimizer, model, vgg, nepochs, train_loader, alpha, beta, dir_name, reg,
                saved_test_images_counter, content_image, style_targets, style_layers,
                style_layers_weights, content_layer):
    '''
    Train a pytorch model and evaluate it every epoch.
    Params:
    model - a pytorch model to train
    optimizer - an optimizer 
    nepochs - number of training epochs
    train_loader - dataloader for the trainset
    '''
    train_losses = []
    start_time = time.time()
    overall_start_time = start_time

    for e in range(nepochs):
        #print("Epoch ", e, " out of ", nepochs)
        model.train() # set model in train mode

        for batch_index,(content_batch,y_batch) in enumerate(train_loader):

            batch_len = len(content_batch)
            content_batch = content_batch.to("cuda")
	  
            stn_output = model(content_batch) # the outputs of the whole batch
            
            stn_output_features = utils.get_features(vgg, stn_output)
            stn_output_style_features = [stn_output_features[layer] for layer in style_layers]
            stn_output_content_features = stn_output_features[content_layer]
            
            original_features = utils.get_features(vgg, content_batch)
            original_content_features = original_features[content_layer]
            
            content_loss = utils.calc_content_loss(stn_output_content_features,
                                                   original_content_features)
            style_loss = utils.calc_style_loss(style_layers_weights,
                                               stn_output_style_features, style_targets)

            total_loss = alpha * content_loss + beta * style_loss
            reg_loss = reg * (
                torch.sum(torch.abs(stn_output[:, :, :, :-1] - stn_output[:, :, :, 1:])) + 
                torch.sum(torch.abs(stn_output[:, :, :-1, :] - stn_output[:, :, 1:, :]))
            )
            total_loss += reg_loss

            if batch_index%10 == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print("batch index ",batch_index,"\n\tcontent_loss = ", (content_loss).item(),
	              "\n\tstyle_loss = ", (style_loss).item(),
		      "\n\ttotal_loss = ", total_loss.item(),
                      "\n\telapsed = ", elapsed_time)

            if batch_index%150==0:
                print("\n\nSave a test output\n\n")
                test_image = model(content_image)
                utils.show_stylized_image(test_image,saved_test_images_counter, dir_name)
                saved_test_images_counter += 1
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        train_losses.append(total_loss.item())
    end_time = time.time()-overall_start_time
    print("how much time this training took: ",end_time)
    return train_losses
###############################################################################################




def main():
    ###########################################################################################
    """# images"""
    ###########################################################################################
    #load images, ordered as [style_image, content_image]
    imgs, imgs_torch = utils.load_images('modern.jpg', 'dog.jpg', image_dir)
    style_image, content_image = imgs_torch
    # Display images
    if False:
        for img in imgs:
            plt.imshow(img)
            plt.show()
            
    style_image = style_image.repeat(batch_size_,1,1,1)
    
    #########################################################################################
    """# Using a pre-trained convolutional neural network vgg19"""
    #########################################################################################
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
    from transformation_net import TransformerNet
    stn = TransformerNet() # style transfer network
    #print(stn)
    for param in stn.parameters():
        param.requires_grad = True

    # move the model to the GPU
    if move_to_cuda:
        stn = stn.to("cuda")

    for p in stn.parameters():
        print("stn layer size	 = ", p.size())
        break
    
    
    ###############################################################################################
    """# Load data"""
    ###############################################################################################
    
    dataset = whole_dataset
    image_size = 256
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(dataset, transform)
    # train_loader returns a list of the first batch were the first colomb is the x and the second
    # is the y in the x colomb is a tensor of tensors with the size of the batch,
    # where the inner tensors are the images in the batch.
    train_loader = DataLoader(train_dataset, batch_size=batch_size_)

    saved_test_images_counter = 0
    nepochs = 8
    reg = 0.000000000001
    beta_list = [0.001,0.0001]
    lr_list = [0.001]
    for beta in beta_list:
        for lr in lr_list:
            name = "_2906_2333_modern_gatys_transforms_batch_size1_a001_b"+str(beta)+"_lr"+str(lr)+"reg"+str(reg)
            loss_file_name = "training_losses"+name+".pkl"
            pre_loss_file_name = "pre_training_losses"+name+".pkl"
            dir_name = "training_pics" + name
            optimizer = optim.Adam(stn.parameters(), lr=lr)
            
            pre_train_losses = pre_train(optimizer, stn, 6, train_loader,
                                         dir_name, saved_test_images_counter,content_image)
            
            utils.save_loss_file(pre_loss_file_name,pre_train_losses)
            
            fig_path = "/home/mc/StyleTransfer/"+ dir_name +"/pre_losses"
            plt.plot(pre_train_losses, label = "pre_train_losses")
            plt.xlabel('epoch')
            plt.legend()
            plt.savefig(fig_path)
            
            saved_test_images_counter = 1000
            train_losses = train_model(optimizer, stn, vgg, nepochs, train_loader, 0.01, beta,
                                       dir_name,reg,saved_test_images_counter, content_image,
                                       style_targets, style_layers,
                                       style_layers_weights, content_layer)

            print(train_losses)
            
            utils.save_loss_file(loss_file_name,train_losses)
            
            fig_path = "/home/mc/StyleTransfer/"+ dir_name +"/losses"
            plt.plot(train_losses, label = "train_losses")
            plt.xlabel('epoch')
            plt.legend()
            plt.savefig(fig_path)
            
            test_content_image = stn(content_image)
            utils.show_stylized_image(test_content_image, 2000, dir_name)
            
            ######save model#########################################################################
            model_name = "model" + name +".model"
            stn.eval()
            save_model_path = os.path.join("/home/mc/StyleTransfer", model_name)
            torch.save(stn.state_dict(), save_model_path)
            print("saved model to path")

if __name__ == "__main__":
    main()
