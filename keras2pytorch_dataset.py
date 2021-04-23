from __future__ import print_function
from PIL import Image
import numpy as np
from misc import AverageMeter
from scripts.eval_accuracy import simple_accuracy
from utils import normalize_minus1_1, denormalize_minus1_1
import torchvision.transforms as transforms
import random

import torch.utils.data as data
import torch
import torchvision.transforms.functional as TF

def softmax(input_tensor):
    act = torch.nn.Softmax(dim=1)
    return act(input_tensor).numpy()

class dataset_pytorch(data.Dataset):
    def __init__(self, train_data, train_labels, test_data, test_labels, train=True,
                 transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.train_data = train_data  # ndarray
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class trainset_pytorch(data.Dataset):
    def __init__(self, train_data, train_labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.train_data = train_data  # ndarray
        self.train_labels = train_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        # img = Image.fromarray(img)  # used if the img is [H, W, C] and the dtype is uint8

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.train_data)

class trainset_pytorch_tiae(data.Dataset):
    def __init__(self, train_data, train_data_transformed, train_labels, data_aug=False, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.train_data = train_data  # ndarray
        self.train_data_transformed = train_data_transformed  # ndarray
        self.train_labels = train_labels
        self.data_aug = data_aug

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, img_transformed, target = self.train_data[index], self.train_data_transformed[index], self.train_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image


        if not self.data_aug and self.transform is not None:
            img = self.transform(img)
            img_transformed = self.transform(img_transformed)
        else:
            # do data augmentation here
            # modified from https://discuss.pytorch.org/t/how-to-apply-same-transform-on-a-pair-of-picture/14914

            img_int = np.uint8(denormalize_minus1_1(img))
            img_transformed_int = np.uint8(denormalize_minus1_1(img_transformed)).squeeze()

            img = Image.fromarray(img_int)  # used if the img is [H, W, C] and the dtype is uint8
            img_transformed = Image.fromarray(img_transformed_int, 'L')  # used if the img is [H, W, C] and the dtype is uint8

            # Random crop
            i, j, h, w = transforms.RandomCrop(size=32, padding=4).get_params(img, output_size=(32, 32))
            img = TF.crop(img, i, j, h, w)
            img_transformed = TF.crop(img_transformed, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                img = TF.hflip(img)
                img_transformed = TF.hflip(img_transformed)

            # to numpy
            img_np = np.array(img, dtype=float)
            img_transformed_np = np.array(img_transformed, dtype=float)

            # Normalize
            img = normalize_minus1_1(img_np)
            img_transformed = normalize_minus1_1(img_transformed_np)

            # to tensor
            img = TF.to_tensor(img).float()
            img_transformed = TF.to_tensor(img_transformed).float()

        if self.target_transform is not None:
            target = self.target_transform(target)

        id = index // 4
        tid = index % 4

        return img, img_transformed, target, id, tid

    def __len__(self):
        return len(self.train_data)

class testset_pytorch(data.Dataset):
    def __init__(self, test_data, transform=None):
        self.transform = transform
        self.test_data = test_data  # ndarray

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.test_data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.test_data)

class dataset_reorganized(data.Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform

        self.data = data  # ndarray

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        imgs = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        # img = Image.fromarray(img)  # used if the img is [H, W, C] and the dtype is uint8

        if self.transform is not None:
            new_imgs = []
            for i in range(imgs.shape[0]):
                img = imgs[i]
                img = self.transform(img)
                new_imgs.append(img.unsqueeze(0))
            new_imgs = torch.cat(new_imgs, dim=0)
        else:
            raise NotImplementedError


        return new_imgs

    def __len__(self):
        return len(self.data)

def train_reorganized(trainloader, model, criterion, optimizer, epochs):
    # train the model
    model.train()
    top1 = AverageMeter()
    losses = AverageMeter()
    for epoch in range(epochs):
        for batch_idx, (inputs) in enumerate(trainloader):
            targets = torch.LongTensor(np.tile(np.arange(inputs.size(1)), inputs.size(0)))
            inputs = inputs.reshape(-1, inputs.size(-3), inputs.size(-2), inputs.size(-1))

            inputs, targets = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(targets.cuda())

            outputs, _ = model(inputs)

            loss = criterion(outputs, targets)

            prec1 = simple_accuracy(outputs.data.cpu(), targets.data.cpu())

            top1.update(prec1, inputs.size(0))
            losses.update(loss.data.cpu(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('Epoch: [{} | {}], batch: {}, loss: {}, Accuracy: {}'.format(epoch + 1, epochs, batch_idx + 1, losses.avg, top1.avg))

def test_reorganized(testloader, model):
    model.eval()
    res = torch.Tensor()
    for batch_idx, (inputs) in enumerate(testloader):
        inputs = inputs.reshape(-1, inputs.size(-3), inputs.size(-2), inputs.size(-1))
        inputs = torch.autograd.Variable(inputs.cuda())
        outputs, _ = model(inputs)
        res = torch.cat((res, outputs.data.cpu()), dim=0)
    return res

def get_scores(outputs, targets):
    scores = []
    for i in range(outputs.shape[0]):
        scores.append(outputs[i, targets[i]])
    return np.array(scores)