import torch
from matplotlib import pyplot as plt 
from models  import model1 
import  torchvision
from torchvision import transforms , datasets
from torch import optim , nn 

class train():
    def __init__(self):
        self.model = model1()
        transform = transforms.Compose([
                        transforms.RandomAffine(10), 
                        transforms.ToTensor() , 
                        transforms.Normalize((0.5,) , (0.5,)), 
                        ])
        trainset = datasets.MNIST('MNIST_data/'  , train = True , transform = transform)
        self.dataloader = torch.utils.data.DataLoader(trainset, batch_size = 64,  shuffle = True)
        
    def train_dat(self):    
        len_d = len(self.dataloader)
        opt = optim.SGD(self.model.parameters(), lr = 0.03) 
        criterion = nn.CrossEntropyLoss()
        print ("training.....")
        for epochs in range(1 , 9):
            running_loss = 0 
            print ("epoch no ---" , epochs)
            for feat, lab in self.dataloader :
                opt.zero_grad()
                try :
                    output = self.model.forward(feat)
                except RuntimeError:
                    continue
                loss = criterion(output ,lab)
                loss.backward()
                opt.step()
                running_loss += loss.item()
            print ("LOSS --->" , running_loss/len_d)
        torch.save(self.model.state_dict() , "optim_weights.pth")

        epochs+= 1
        print("LOSS -->" , running_loss/len_d)

if __name__ == '__main__' :
    z = train()
    z.train_dat()
    
        
