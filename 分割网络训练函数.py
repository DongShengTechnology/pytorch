def make_readable(seconds):
    # 把秒转化为分钟和小时
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h,m,s

class Logger(object):
    # 将控制台的输出结果输出至txt文件
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a') # 读写模式设为a

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# 计算混淆矩阵
def _fast_hist(label_true, label_pred, n_class):
    # mask在和label_true相对应的索引的位置上填入true或者false
    # label_true[mask]会把mask中索引为true的元素输出
    mask = (label_true >= 0) & (label_true < n_class)
    # np.bincount()会给出索引对应的元素个数
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    # 一个batch里面可能有多个数据
    # 通过迭代器将一个个数据进行计算
    for lt, lp in zip(label_trues, label_preds):
        # numpy.ndarray.flatten将numpy对象拉成1维
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    pa = np.diag(hist).sum() / hist.sum()
    cpa = np.diag(hist) / hist.sum(axis=0)
    mpa = np.nanmean(cpa)  # nanmean会自动忽略nan的元素求平均
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    miou = np.nanmean(iou)
    return pa, mpa, miou

def train_val_ch7_show(net, train_iter, val_iter,num_classes,criterion, optimizer, device, num_epochs,txt_save=False,
                       plt_show=False,plt_save=False):

    net = net.to(device)
    loss = criterion

    if txt_save:
        sys.stdout = Logger('history.txt', sys.stdout)
        sys.stderr = Logger('history.txt', sys.stderr)  # redirect std err, if necessary

    print("training on ", device)
    print('--'*90)

    # 记录训练集和测试集的损失值和正确率
    # best_model_wts=copy.deeepcopy(net.state_dict())
    # best_acc=0.0

    train_loss_all=[]
    train_PA=[]
    train_MPA=[]
    train_MIoU=[]

    val_loss_all=[]
    val_PA=[]
    val_MPA=[]
    val_MIoU=[]

    for epoch in range(num_epochs):

        train_l_sum, val_l_sum =0.0 ,0.0
        n_,m_,train_batch_count, val_batch_count = 0,0,0,0
        train_pa,train_mpa,train_miou,val_pa,val_mpa,val_miou=0,0,0,0,0,0
        minute,sec,start=0,0,time.time()

        net.train() #训练模式
        for X, y in train_iter:
            X = Variable(X).to(device)
            y = Variable(y).to(device)
            #前向传播
            y_hat = net(X)
            y_hat = F.log_softmax(y_hat,1)
            l = loss(y_hat, y)
            #反向传播
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            #评估指标
            label_pred = y_hat.max(dim=1)[1].data.cpu().numpy()
            label_true = y.data.cpu().numpy()

            for lbt, lbp in zip(label_true, label_pred):
                pa,mpa,miou = label_accuracy_score(lbt, lbp, num_classes)
                train_pa += pa
                train_mpa += mpa
                train_miou += miou

            n_ += y.shape[0]
            train_batch_count += 1

        net.eval() #评估模式
        for X,y in val_iter:
            X =  Variable(X).to(device)
            y =  Variable(y).to(device)

            y_hat = net(X)
            y_hat = F.log_softmax(y_hat,1)
            l = loss(y_hat, y)

            val_l_sum += l.item()

            label_pred = y_hat.max(dim=1)[1].data.cpu().numpy()
            label_true = y.data.cpu().numpy()

            for lbt, lbp in zip(label_true, label_pred):
                pa,mpa,miou = label_accuracy_score(lbt, lbp, num_classes)
                val_pa += pa
                val_mpa += mpa
                val_miou += miou

            m_ += y.shape[0]
            val_batch_count += 1

        _,minute,sec=make_readable(time.time() - start)
        print('Epoch %d:train loss %.3f, val loss %.3f || train PA %.3f, val PA %.3f ||'
              ' train MPA %.3f, val MPA %.3f || train MIoU %.3f, val MIoU %.3f || usage time %02d:%02d (%.2f sec)'
              % (epoch + 1, train_l_sum / train_batch_count, val_l_sum/val_batch_count,train_pa/n_ ,val_pa/m_,
                 train_mpa/n_,val_mpa / m_,train_miou/n_,val_miou /m_,minute,sec,time.time() - start))

        train_loss_all.append(train_l_sum / train_batch_count)
        val_loss_all.append(val_l_sum/ val_batch_count)
        train_PA.append(train_pa/n_)
        val_PA.append(val_pa/m_ )
        train_MPA.append(train_mpa/n_)
        val_MPA.append(val_mpa/m_ )
        train_MIoU.append(train_miou/n_)
        val_MIoU.append(val_miou/m_ )
        # if val_acc_all[-1]>best_acc:
        #     best_acc=val_acc_all[-1]
        #     best_model_wts=copy.deepcopy(net.state_dict())

    print('--'*90)
    # net.load_state_dict(best_model_wts)
    train_process=pd.DataFrame(
        data={'epoch':range(num_epochs),
              'train_loss_all':train_loss_all,
              'val_loss_all':val_loss_all,
              'train_PA':train_PA,
              'val_PA':val_PA,
              'train_MPA':train_MPA,
              'val_MPA':val_MPA,
              'train_MIoU':train_MIoU,
              'val_MIoU':val_MIoU})

    if plt_show:
        plt.figure(figsize=(14,10))
        plt.subplot(2,2,1)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.plot(train_process.epoch,train_process.train_loss_all,color='r',label='Train loss')
        plt.plot(train_process.epoch,train_process.val_loss_all, color='b',label='Val loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.title('Train Loss: %.3f      Val Loss: %.3f'%(train_process.iloc[-1,1],train_process.iloc[-1,2]))
        plt.subplot(2,2,2)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.plot(train_process.epoch,train_process.train_PA,color='r',label='Train PA')
        plt.plot(train_process.epoch,train_process.val_PA,color='b',label='Val PA')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.title('Train PA: %.3f      Val PA: %.3f'%(train_process.iloc[-1,3],train_process.iloc[-1,4]))
        plt.legend()
        plt.subplot(2,2,3)
        plt.subplots_adjust(wspace=0.3,hspace=0.3)
        plt.plot(train_process.epoch,train_process.train_MPA,color='r',label='Train MPA')
        plt.plot(train_process.epoch,train_process.val_MPA,color='b',label='Val MPA')
        plt.xlabel('epoch')
        plt.ylabel('MPA')
        plt.title('Train MPA: %.3f      Val MPA: %.3f'%(train_process.iloc[-1,5],train_process.iloc[-1,6]))
        plt.legend()
        plt.subplot(2,2,4)
        plt.subplots_adjust(wspace=0.3,hspace=0.3)
        plt.plot(train_process.epoch,train_process.train_MIoU,color='r',label='Train MIoU')
        plt.plot(train_process.epoch,train_process.val_MIoU,color='b',label='Val MIoU')
        plt.xlabel('epoch')
        plt.ylabel('MIoU')
        plt.title('Train MIoU: %.3f      Val MIoU: %.3f'%(train_process.iloc[-1,7],train_process.iloc[-1,8]))
        plt.legend()
        if plt_save:
            plt.savefig('./history.png')
        plt.show()