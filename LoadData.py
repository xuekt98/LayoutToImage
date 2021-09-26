import pickle
import torch
import torchvision
import numpy as np
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class MyDataset(Dataset):
    def __init__(self, pkl_path, image_dir, image_size=(256, 256)):
        super(MyDataset, self).__init__()       
        self.image_dir = image_dir
        self.image_size = image_size
        transform = [T.Resize(image_size), T.ToTensor()]
        self.transform = T.Compose(transform)

        self.object_to_index = {}
        self.data = []
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)

            #统计所有的物体类别,给每一个类别赋予一个不同的index用来做embedding
            for image_info in self.data:
                image_objects = image_info['objects']
                for single_object in image_objects:
                    if single_object['object_id'] not in self.object_to_index:
                        self.object_to_index[single_object['object_id']] = len(self.object_to_index) + 1
        
        #print(len(self.object_to_index))
        #print(self.object_to_index)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_info = self.data[index]
        img_name = img_info['image_id']
        image = Image.open(os.path.join(self.image_dir, str(img_name)+'.jpg'))
        WW, HH = image.size
        image = self.transform(image)

        #由于原图像大小不相同需要Resize，bounding box原来的信息不能直接用，要改用比例
        objs = img_info['objects']
        objects, boxes = [], []
        for o in objs:
            objects.append(self.object_to_index[o['object_id']])
            x, y, w, h = o['x'], o['y'], o['h'], o['w']
            x0 = x / WW
            y0 = y / HH
            x1 = (w) / WW
            y1 = (h) / HH
            boxes.append(np.array([x0, y0, x1, y1]))
        objects = torch.LongTensor(objects)
        boxes = torch.tensor(boxes)

        #print(objects)
        #print(boxes)
        #由于不同的图像含有的内部类别个数不同，需要进行填充
        return image, objects, boxes
   
