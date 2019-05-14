from __future__ import print_function
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path
import errno

def get_dataset(name):
    if name == 'MNIST':
        return get_MNIST()
    elif name == 'CIFAR10':
        return get_CIFAR10()
    elif name == 'CALTECH':
        return get_CALTECH(overwrite=False)
    elif name == 'QUICKDRAW':
        return get_QUICKDRAW(0.06)
    

def get_MNIST():
    raw_tr = datasets.MNIST('./MNIST', train=True, download=True)
    raw_te = datasets.MNIST('./MNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te


def get_CIFAR10():
    data_tr = datasets.CIFAR10('./CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10('./CIFAR10', train=False, download=True)
    X_tr = data_tr.train_data
    Y_tr = torch.from_numpy(np.array(data_tr.train_labels))
    X_te = data_te.test_data
    Y_te = torch.from_numpy(np.array(data_te.test_labels))
    return X_tr, Y_tr, X_te, Y_te

def get_QUICKDRAW(percentage):
    dataroot = '/datapool/quick-draw'
    filenames = ['dolphin', 'cat', 'face', 'angel', 'airplane', 'apple', 'broccoli', 'crayon', 'bicycle', 'elephant']
    # filenames = ['dolphin', 'cat', 'face', 'angel', 'airplane']
    # filenames = ['dolphin', 'cat']
    x = []
    for filename in filenames:
        data = np.load(os.path.join(dataroot, filename+'.npy'))
        print('number of samples in {}: {}'.format(filename, data.shape[0]))
        mask = np.random.choice([False, True], len(data), p=[1.0-percentage, percentage])
        print('number of samples in {} after down-sampling: {}'.format(filename, data[mask].shape[0]))
        x.append(data[mask])
    X_tr = []
    X_te = []
    Y_tr = []
    Y_te = []
    for x_data, i in zip(x, range(len(x))):
        # x_data = x_data[:(int)(percentage*len(x_data))]
        n_data = len(x_data)
        n_train_data = (int)(0.8*n_data)
        X_tr.append(x_data[:n_train_data])
        X_te.append(x_data[n_train_data:])
        Y_tr.append([i]*n_train_data)
        Y_te.append([i]*(n_data - n_train_data))
    X_tr = np.concatenate(X_tr, axis=0)
    X_te = np.concatenate(X_te, axis=0)
    Y_tr = np.concatenate(Y_tr, axis=0)
    Y_te = np.concatenate(Y_te, axis=0)
    #print("X_tr.shape: ", X_tr.shape)
    #print("X_te.shape: ", X_te.shape)
    # reshape
    X_tr = X_tr.reshape((X_tr.shape[0], 1, 28, 28))
    X_te = X_te.reshape((X_te.shape[0], 1, 28, 28))
    #print(type(X_tr))
    Y_tr = torch.from_numpy(np.array(Y_tr))
    Y_te = torch.from_numpy(np.array(Y_te))
    return X_tr, Y_tr, X_te, Y_te

class CaltechFolder(Dataset):
    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    '''
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.all_items=find_classes(os.path.join(self.root))
        self.idx_classes=index_classes(self.all_items)

    def __getitem__(self, index):
        filename=self.all_items[index][0]
        img=str.join('/',[self.all_items[index][2],filename])
        target=self.all_items[index][1] #self.idx_classes[self.all_items[index][1]]

        if self.target_transform is not None:
            target = self.target_transform(target)
        return  img, target

    def __len__(self):
        return len(self.all_items)

def find_classes(root_dir):
    retour=[]
    for (root,dirs,files) in os.walk(root_dir):
        for f in sorted(files):
            if (f.endswith("jpg")):
                r=root.split('/')
                lr=len(r)
                # retour.append((f,r[lr-2]+"/"+r[lr-1],root))
                retour.append((f,"/"+r[lr-1],root))                
    print("== Found %d items "%len(retour))
    return retour

def index_classes(items):
    idx={}
    for i in items:
        if (not i[1] in idx):
            idx[i[1]]=len(idx)
    print("== Found %d classes"% len(idx))
    return idx

def create_nparray(dataset, dataroot, processedroot, overwrite=False):
    """
    Constructs a numpy array of image paths and labels for the dataset - train/test 
    dataset - dataset type - train/test/val
    dataroot - data dir path
    processedroot - where to save the nparray
    overwrite - whether to overwrite the existing numpy array, default false
    """
    if not os.path.isfile(os.path.join(processedroot, dataset, 'images.npy')) or overwrite:
        print(str.join('/', [dataroot, dataset]))
        x = CaltechFolder(str.join('/', [dataroot, dataset]))
        images = []
        labels = []
        for (img, label) in x:
            images.append(img)
            labels.append(label)
        images = np.array(images)
        labels = np.array(labels)
        
        if not os.path.exists(os.path.join(processedroot, dataset)):
            os.mkdir(os.path.join(processedroot, dataset))
        if overwrite:
            if os.path.isfile(os.path.join(processedroot, dataset, 'images.npy')):
                os.remove(os.path.join(processedroot, dataset, 'images.npy'))
                os.remove(os.path.join(processedroot, dataset, 'labels.npy'))
            np.save(os.path.join(processedroot, dataset, 'images.npy'), images)
            np.save(os.path.join(processedroot, dataset, 'labels.npy'), labels)
    else:
        images = np.load(os.path.join(processedroot, dataset, 'images.npy'))
        labels = np.load(os.path.join(processedroot, dataset, 'labels.npy'))
    return images, labels

def get_CALTECH(overwrite=False):
    dataroot='/home/nisha/data/256_ObjectCategoriesSubset'
    # dataroot='/datapool/256_ObjectCategoriesSplit'
    # dataroot='/datapool/101_ObjectCategoriesSplit'
    processedroot = str.join('/', [dataroot, 'processed'])
    if not os.path.exists(processedroot):
        os.mkdir(os.path.join(processedroot))
    X_tr, Y_tr = create_nparray('train', dataroot, processedroot, overwrite)
    print(sorted(set(list(Y_tr))))
    print(X_tr[1:5])
    X_te, Y_te = create_nparray('valid', dataroot, processedroot, overwrite)
    print(sorted(set(list(Y_te))))
    print(X_te[1:5])
    # Making sure that the class indices are consistent across train and valid sets
    
    class_indices = []
    for x in enumerate(sorted(set(list(Y_tr)))):
        class_indices.append(x)
    print("class_indices: ", class_indices)
    Y_tr_upd = np.empty(len(Y_tr), dtype=int)
    for i, x in enumerate(list(Y_tr)):
        index = [i for i, y in enumerate(class_indices) if y[1] == x]
        Y_tr_upd[i] = int(index[0])

    Y_te_upd = np.empty(len(Y_te), dtype=int)
    for i, x in enumerate(list(Y_te)):
        index = [i for i, y in enumerate(class_indices) if y[1] == x]
        Y_te_upd[i] = int(index[0])

    #X_tr = torch.from_numpy(X_tr)
    Y_tr = torch.from_numpy(Y_tr_upd)
    #X_te = torch.from_numpy(X_te)
    Y_te = torch.from_numpy(Y_te_upd)
    
    return X_tr, Y_tr, X_te, Y_te

def get_handler(name):
    if name == 'MNIST':
        return MNISTHandler
    elif name == 'CIFAR10':
        return CIFARHandler
    elif name == 'CALTECH':
        return CaltechHandler
    elif name == 'QUICKDRAW':
        return QuickDrawHandler

class MNISTHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
            #print("x.shape: ", x.shape)
        return x, y, index

    def __len__(self):
        return len(self.X)


class CIFARHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
            print("x.shape: ", x.shape)
        return x, y, index

    def __len__(self):
        return len(self.X)

class CaltechHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        # If there are some gray/ single channel images
        x = Image.open(x).convert('RGB')
        if self.transform is not None:
            #x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class QuickDrawHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = x.reshape(-1, x.shape[1])
            x = Image.fromarray(x, mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)
