from dataset import get_dataset, get_handler
from model import get_net
from training import Training
from torchvision import transforms

import torch
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    seed = 123
    num_images = 10
    dataset_name = 'CALTECH'
    
    args_pool = {
        'CALTECH':
        {
            'n_epoch': 10,
            'n_classes': 10,
            'fc_only': True,
            'transform':
            {
                'train': transforms.Compose([transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                'test': transforms.Compose([transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
            },
            'loader_tr_args': {'batch_size': 25, 'num_workers': 1},
            'loader_te_args': {'batch_size': 25, 'num_workers': 1},
            'loader_sample_args': {'batch_size': 25, 'num_workers': 1},
            'optimizer_args': {'lr': 0.001}
        }
    }
   
    args = args_pool[dataset_name]

    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # load dataset
    x_train, y_train, x_test, y_test = get_dataset(dataset_name)
    print("x_train: ", len(x_train))
    print("y_train: ", y_train.shape)
    print("x_test: ", len(x_test))
    print("y_test: ", y_test.shape)
    
    net = get_net(dataset_name)
    handler = get_handler(dataset_name)
    
    x_train_replicated = np.concatenate([x_train, x_train, x_train, x_train, x_train, x_train, x_train, x_train, x_train,x_train])
    y_train_replicated = torch.cat([y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train])
    print("x_train_replicated: ", len(x_train_replicated))
    print("y_train_replicated: ", y_train_replicated.shape)
    
    net_replicated = get_net(dataset_name)
    handler_replicated = get_handler(dataset_name)
    rep_training = Training(x_train_replicated, y_train_replicated, x_test, y_test, net_replicated, handler_replicated, args)
    rep_training.train()
    
    rep_training.check_accuracy(x_test, y_test)

    args_pool = {
        'CALTECH':
        {
            'n_epoch': 10*10,
            'n_classes': 10,
            'fc_only': True,
            'transform':
            {
                'train': transforms.Compose([transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                'test': transforms.Compose([transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
            },
            'loader_tr_args': {'batch_size': 25, 'num_workers': 1},
            'loader_te_args': {'batch_size': 25, 'num_workers': 1},
            'loader_sample_args': {'batch_size': 25, 'num_workers': 1},
            'optimizer_args': {'lr': 0.001}
        }
    }
   
    args = args_pool[dataset_name]
    net_100epochs = get_net(dataset_name)
    handler_100epochs = get_handler(dataset_name)
    
    training_100epochs = Training(x_train, y_train, x_test, y_test, net_100epochs, handler_100epochs, args)
    training_100epochs.train()
    
    training_100epochs.check_accuracy(x_test, y_test)
