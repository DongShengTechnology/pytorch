###########################
def load_image_data(data_dir,batch_size=None,shuffle=False,num_workers=None,transform=None):
    print('PATH:',data_dir)
    if batch_size is None:
        batch_size=1
    if num_workers is None:
        num_workers=0
    data=ImageFolder(data_dir,transform=transform)
    data_loader=Data.DataLoader(data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    print('The num of image:',len(data.targets))
    for step,(b_x,b_y) in enumerate(data_loader):
        if step>0:
            break
    print('The shape of image:',b_x.shape)
    print('--'*25)
    return data_loader

#############################

def load_image_data_show(data_dir,batch_size=None,shuffle=False,num_workers=None,transform=None):
    print('PATH:',data_dir)
    if batch_size is None:
        batch_size=1
    if num_workers is None:
        num_workers=0
    data=ImageFolder(data_dir)
    plt.figure(figsize=(16, 16))
    for i in range(16):
        plt.subplot(4, 4, i + 1) 
        plt.imshow(data[i][0])
        plt.axis('off')
    plt.show()
    dataset=ImageFolder(data_dir,transform=transform)
    data_loader=Data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    print('The num of image:',len(dataset.targets))
    for step,(b_x,b_y) in enumerate(data_loader):
        if step>0:
            break
    print('The shape of image:',b_x.shape)
    print('--'*25)
    return data_loader