
from edugrad.tensor import Tensor
import pandas as np
import torch 




#xavier/glorot initialization
def layer_init(m, h):
  ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
  return ret.astype(np.float32)


l1 = Tensor(layer_init(784, 128))
l2 = Tensor(layer_init(128, 10))


#MNIST Dataset
DATASET_PATH = "../datasets"
BATCH_SIZE = 64

train_set = torchvision.datasets.MNIST(DATASET_PATH, train= True, transform= transforms.ToTensor(), download= False)
test_set = torchvision.datasets.MNIST(DATASET_PATH, train= False, transform= transforms.ToTensor(), download= False)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)

#print(len(train_set), len(test_set)) #60K, 10K

dataiter = iter(train_loader)
images, labels = next(dataiter)

#print(images.shape) #torch.Size([64, 1, 28, 28]) -> 64 images in each batch with 28x28
#print(labels.shape) #torch.Size([64]) -> 64 labels 

#check first img
#for i, (img, label) in enumerate(train_loader):
#    print('i = ', i)
#    #print(img)
#    print('img.shape:', img.shape)
#    print('type(img):', type(img))
#    print('label:', label)
#    print('label.shape:', label.shape) 
#    print('type(label):', type(label)) 
#    print('-'*5)
#    if i == 1:
#        break







