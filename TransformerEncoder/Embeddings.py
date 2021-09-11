import torch
import torch.nn as nn

''' 
    !!!bounding box 信息如何做嵌入还需要进一步确定 
    暂时先使用
'''
class Embeddings(nn.Module):
    '''输入label，bounding box，生成嵌入的信息'''
    def __init__(self, d_model=512, num_classes=10):
        super(Embeddings, self).__init__()
        self.class_embedding = nn.Embedding(num_classes, d_model)
        self.d_model = d_model
    
    ''' !!!这个地方还需要重写 '''
    def forward(self, classes, bbs):
        '''
            Params:
                classes: 物体类别
                bb: bounding box信息
        '''
        output = self.class_embedding(classes)
        return output