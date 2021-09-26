'''全连接层Fully Connected(FC) Block'''
import torch
import torch.nn as nn

#参数初始化方式 kaiming分布初始化
def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

#参数初始化方式 Xavier分布初始化
def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)

class FCBlock(nn.Module):
    """Fully Connected(FC) Block"""
    def __init__(self, in_features, out_features, hidden_features, num_hidden_layers,
             outermost_linear=False, nonlinearity='relu', weight_init=None):
        """
            Params:
                in_features: FC输入特征维度
                out_features: FC输出特征维度
                hidden_features: FC中间隐藏层特征维度
                num_hidden_layers: FC中间隐藏层个数(不包含第一层和最后一层)
                outermost_linear: FC最后一层是否需要激活函数
                nonlinearity: 激活函数
                weight_init: FC参数初始化方式
        """
        super(FCBlock, self).__init__()
        
        #存储非线性激活函数与参数初始化方式的表
        nls_and_inits = {'relu':(nn.ReLU(inplace=True), init_weights_normal, None)}
        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]
        
        if weight_init is not None:
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init
        
        self.net = [] #MLP中的所有模块
        
        #第一层 从输入维度到隐藏层维度
        self.net.append(nn.Sequential(
            nn.Linear(in_features, hidden_features), nl
        ))
        
        #中间层 
        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))
        
        #最后一层 从隐藏层维度到输出维度
        if outermost_linear:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, out_features), nl
            ))
        
        self.net = nn.Sequential(*self.net)
        
        #初始化模型参数
        if self.weight_init is not None:
            self.net.apply(self.weight_init)
        if first_layer_init is not None:
            self.net[0].apply(first_layer_init)
        
    def forward(self, model_input):
        output = self.net(model_input)
        return output
        