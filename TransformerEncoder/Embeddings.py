import torch
import torch.nn as nn

''' 
    !!!bounding box 信息如何做嵌入还需要进一步确定 
    暂时先使用
'''
class Embeddings(nn.Module):
    '''输入label，bounding box，生成嵌入的信息'''
    def __init__(self, d_model, num_label, max_image_size):
        super(Embeddings, self).__init__()
        self.label_embedding = nn.Embedding(num_label, d_model)
        self.d_model = d_model
    
    ''' !!!这个地方还需要重写 '''
    def forward(self, label, bb):
        '''
            Params:
                label: 物体类别
                bb: bounding box信息
        '''
        return self.label_embedding(label)