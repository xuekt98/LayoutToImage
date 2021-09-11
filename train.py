import torch
import pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from LoadData import MyDataset
from TransformerEncoder.Embeddings import Embeddings

def get_dataset():
	data = MyDataset(pkl_path='dataset/obj.pkl', image_dir='dataset/images/')
	return data

def train(args=None):
	train_data = get_dataset()
	train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
	image, objects, boxes = iter(train_loader).next()
	
	net = Embeddings(d_model=args['d_model'], num_classes=args['num_classes'])
	
	for epoch in range(args['total_epoch']):
		net.train()

		for index, data in enumerate(train_loader):
			image, objects, boxes = data
			output = net(objects, boxes)
			print(output)
			break

'''
with open('dataset/obj.pkl', 'rb') as fr:
    inf = pickle.load(fr)
    out = open('1.txt', 'a')
    print(inf, file=out)
    for i in range(len(inf)):
        img_info = inf[i]
        img_name = img_info['image_id']
        print(img_name)

inf = []
with open('dataset/obj.pkl', 'rb') as fr:
	inf = pickle.load(fr)

inf = inf[0:64]
with open('dataset/obj.pkl', 'wb') as fr:
	inf = pickle.dump(inf, fr)
'''
args = {'d_model':512,
	   	'batch_size':1,
	   	'num_classes':2500,
	   	'total_epoch':1
	   }
train(args)

'''
t1 = torch.FloatTensor([[1, 2], [5, 6]])
t2 = torch.FloatTensor([[3, 4], [7, 8]])
l = []
l.append(t1)
l.append(t2)
ta = torch.cat(l, dim=0)
print(type(l))
print(ta)
'''