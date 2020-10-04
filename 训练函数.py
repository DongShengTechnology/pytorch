#########################################################
def train_val_ch5_show(net, train_iter, val_iter,criterion, optimizer, device, num_epochs,
                       plt_show=False,plt_save=False,test_iter=None):
    net = net.to(device)
    print("training on ", device)
    print('--'*45)
    loss = criterion
    #记录训练集和测试集的损失值和正确率
    # best_model_wts=copy.deeepcopy(net.state_dict())
    # best_acc=0.0
    train_loss_all=[]
    train_acc_all=[]
    val_loss_all=[]
    val_acc_all=[]
    test_acc_all=[]
    for epoch in range(num_epochs):
        train_l_sum, val_l_sum,train_acc_sum,val_acc_sum =0.0 ,0.0 ,0.0 ,0.0
        n,m, train_batch_count, val_batch_count,start = 0,0,0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            net.train()
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            train_batch_count += 1
        for X,y in val_iter:
            X = X.to(device)
            y = y.to(device)
            net.eval()
            y_hat = net(X)
            l = loss(y_hat, y)
            val_l_sum += l.cpu().item()
            val_acc_sum+=(y_hat.argmax(dim=1)==y).sum().cpu().item()
            m+=y.shape[0]
            val_batch_count += 1
        if test_iter:
            test_acc = evaluate_accuracy(test_iter, net)
            test_acc_all.append(test_acc)
        print('epoch %d:train loss %.3f,val loss %.3f, train acc %.3f, val acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / train_batch_count, val_l_sum/val_batch_count,train_acc_sum / n, val_acc_sum/m, time.time() - start))
        train_loss_all.append(train_l_sum / train_batch_count)
        val_loss_all.append(val_l_sum/ val_batch_count)
        train_acc_all.append(train_acc_sum / n)
        val_acc_all.append(val_acc_sum/ m)
        # if val_acc_all[-1]>best_acc:
        #     best_acc=val_acc_all[-1]
        #     best_model_wts=copy.deepcopy(net.state_dict())
    print('--'*45)
    # net.load_state_dict(best_model_wts)
    train_process=pd.DataFrame(
        data={'epoch':range(num_epochs),
              'train_loss_all':train_loss_all,
              'val_loss_all':val_loss_all,
              'train_acc_all':train_acc_all,
              'val_acc_all':val_acc_all})
    if test_iter:
        train_process['test_acc_all']=test_acc_all
        print('test accuracy:%.4f'%(train_process.iloc[-1,5]))
    if plt_show:
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(train_process.epoch,train_process.train_loss_all,color='r',label='Train loss')
        plt.plot(train_process.epoch,train_process.val_loss_all, color='b',label='Val loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.title('Train Loss: %.3f      Val Loss: %.3f'%(train_process.iloc[-1,1],train_process.iloc[-1,2]))
        plt.subplot(1,2,2)
        plt.plot(train_process.epoch,train_process.train_acc_all,color='r',label='Train acc')
        plt.plot(train_process.epoch,train_process.val_acc_all,color='b',label='Val acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.title('Train acc: %.3f      Val acc: %.3f'%(train_process.iloc[-1,3],train_process.iloc[-1,4]))
        plt
        plt.legend()
        if plt_save:
            plt.savefig('./history.png')
        plt.show()