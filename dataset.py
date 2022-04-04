
import torch
import struct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset


def image_idx3_ubyte(image_path):
    data = open(image_path,'rb').read()
    fmt_header = '>iiii'
    offset = 0
    magic_num, image_num, row_num, col_num = struct.unpack_from(fmt_header,data,offset)
    # print(magic_num, image_num, row_num, col_num)
    image_size = row_num * col_num
    offset += struct.calcsize(fmt_header)
    # print(offset)
    fmt_image = '>'+str(image_size)+'B'
    # print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((image_num,row_num,col_num))
    fmt_size = struct.calcsize(fmt_image)
    for i in range(image_num):
        pixel = struct.unpack_from(fmt_image,data,offset)
        arr = np.array(pixel)
        arr = arr.reshape((row_num,col_num))
        offset += fmt_size
        images[i] = arr
    return images


def label_idx1_ubyte(label_path):
    
    data = open(label_path,'rb').read()
    fmt_header = '>ii'
    offset = 0
    magic_num, label_num, = struct.unpack_from(fmt_header,data,offset)
    # print(magic_num, label_num)
    offset += struct.calcsize(fmt_header)
    labels = np.empty((label_num),dtype=np.ubyte)
    # print(labels.shape)
    fmt_label = '>B'
    for i in range(label_num):
        label = struct.unpack_from(fmt_label,data,offset)
        offset += struct.calcsize(fmt_label)
        label = label[0]
        labels[i] = label
    return labels



class ImageDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform

        self.images = image_idx3_ubyte(img_dir)
        self.labels = label_idx1_ubyte(annotation_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        return self.images[idx], self.labels[idx]





# if __name__ == '__main__':
#     labels = label_idx1_ubyte(train_label)
#     images = image_idx3_ubyte(trian_image_set)

#     i = 10
#     plt.imshow(images[i],cmap='gray')
#     plt.show()

