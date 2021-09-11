import torch
import pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from LoadData import MyDataset
from TransformerEncoder.TransformerEncoder import TransformerEncoder

def get_dataset():
	data = MyDataset(pkl_path='dataset/obj.pkl', image_dir='dataset/images/')
	return data

def train(args=None):
	train_data = get_dataset()
	train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
	image, objects, boxes = iter(train_loader).next()
	
	net1 = TransformerEncoder(d_model=args['d_model'],
             				  num_classes=args['num_classes'],
             				  num_trans_layers=args['num_trans_layers'], 
             				  num_heads=args['num_heads'],
             				  d_ffn=args['d_ffn'],
             				  dropout=0.0)
	
	for epoch in range(args['total_epoch']):
		net1.train()

		for index, data in enumerate(train_loader):
			image, objects, boxes = data
			output = net1(objects, boxes)
			#print(output)
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

#测试用参数
args = {'d_model':16,
		'total_epoch':1,
	   	'batch_size':1,
	   	'num_classes':2500,
	   	'num_trans_layers':1,
	   	'num_heads':2,
	   	'd_ffn':16,
	   }
train(args)