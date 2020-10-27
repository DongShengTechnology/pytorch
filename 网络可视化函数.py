############################# 5.7 ##############################
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

def net_hiddenlayer_view(save_dir,net,input_image_channel):
    assert type(save_dir) is str
    hl_graph=hl.build_graph(net,torch.zeros([1,input_image_channel,224,224]))
    hl_graph.theme=hl.graph.THEMES['blue'].copy()
    hl_graph.save(save_dir,format='png')
def net_pytorchviz_view(save_dir,net,input_image_channel):
    # 使用torchviz可视化
    x = torch.randn(1, input_image_channel, 224, 224).requires_grad_(True)
    y = net(x)
    myconvetvis = make_dot(y, params=dict(list(net.named_parameters()) + [('x',x)]))
    # 将myconvetvis保存为图片
    myconvetvis.format = "png"    #形式转化为png
    myconvetvis.directory = save_dir    #指定文件保存位置
    myconvetvis.view()      #会自动在当前文件夹生成文件