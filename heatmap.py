import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def heatmap_show(net,val_data_loader,class_label,plt_save=False):
    pre_lab=[]
    val_y=[]
    for step,(X,y) in enumerate(val_data_loader):
        X=X.to(device)
        y=y.cpu().numpy().tolist()
        val_y.append(y)
        net.eval()
        output=net(X)
        y_hat=torch.argmax(output,1).cpu().numpy().tolist()
        pre_lab.append(y_hat)

    pre_lab=[i for item in pre_lab for i in item]
    val_y=[i for item in val_y for i in item]

    conf_mat=confusion_matrix(val_y,pre_lab)
    df_cm=pd.DataFrame(conf_mat,index=class_label,columns=class_label)
    heatmap=sns.heatmap(df_cm,annot=True,fmt='d',cmap='YlGnBu')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),rotation=0,ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),rotation=45,ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if plt_save:
        plt.savefig('./heatmap.png')
    plt.show()