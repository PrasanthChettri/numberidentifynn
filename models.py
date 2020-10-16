import torch
from torch import nn

class model1(nn.Module):
    def __init__(self , batch_size = 64): 
        super().__init__()
        self.batch_size = batch_size
        #Convo Layers --> 28 , 28 to
        #C1  28 , 28  to 14  , 14
        self.c1 = nn.Conv2d(1 , 3 , 3 ,stride = 1 , padding = 2) 
        #C1  15 , 15 to 4 , 4 
        self.c2 = nn.Conv2d(3 , 6 , 4 , stride = 1 , padding = 1)
        #Pooling Layers  
        self.maxp = nn.MaxPool2d(2 , 2)
        self.maxp2 = nn.MaxPool2d(4 , 4)
        #Last Layer
        self.lin = nn.Linear(54 , 20)
        self.lin2 = nn.Linear(20 , 10)
        #activation func
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.c1(x)
        x = self.maxp(x)
        x = self.tanh(x)

        x = self.c2(x)
        x = self.maxp2(x)
        x = self.tanh(x)

        x = x.view(self.batch_size , -1)
        x = self.lin(x)

        return x

#LINEAR MODEL
class model2(nn.Module):
    def __init__(self, batch_size = 64):
        super().__init__()
        self.batch_size = batch_size
        self.lin1 = nn.Linear(28*28 , 128)
        self.lin2 = nn.Linear(128 , 64) 
        self.lin3 = nn.Linear(64 , 10)
        self.relu = nn.ReLU()
        
    def forward(self,  x):
        #Initialise  forward
        # FORWARD THROUGH DENSE NUERAL NETS FIRST
        # THEN LSMTS 
        x = x.view(self.batch_size , -1)

        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.lin3(x)

        return x
