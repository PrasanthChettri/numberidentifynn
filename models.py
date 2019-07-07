import torch
from torch import nn

class model1(nn.Module):
    def __init__(self , batch_size = 64): 
        self.batch_size = batch_size
        super().__init__()
        #Convo Layers --> 28 , 28 to
        #C1  28 , 28  to 14  , 14
        self.c1 = nn.Conv2d(1 , 3 , 5 ,stride = 1 , padding = 2) 
        #C1  14 , 14 to 4 , 4 
        self.c2 = nn.Conv2d(3 , 6 , 5 , stride = 1 , padding = 1)
        #Pooling Layers  
        self.maxp = nn.MaxPool2d(2 , 2)
        self.maxp2 = nn.MaxPool2d(3 , 3)
        #Last Layer
        self.lin = nn.Linear(96 , 10)
        #activation func
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.c1(x)
        x = self.maxp(x)
        x = self.relu(x)

        x = self.c2(x)
        x = self.maxp2(x)
        x = self.relu(x)

        x = x.view(self.batch_size , -1)
        x = self.relu(self.lin(x))

        return x
