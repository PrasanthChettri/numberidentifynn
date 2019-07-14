import torch
import numpy as np
from matplotlib import pyplot as plt 
import  models
import  torchvision
from torchvision import transforms , datasets
from torch import optim , nn 
from torch.utils.data.sampler import SubsetRandomSampler as subsample 
import pickle

class train():
    def __init__(self):
        self.model = models.model1()
        transform = transforms.Compose([
                        transforms.RandomCrop((28 , 28)) ,
                        transforms.ToTensor() , 
                        transforms.Normalize((0.5,) , (0.5,)), 
                        ])

        dataset = datasets.MNIST('MNIST_data/'  , train = True , transform = transform)
        len_d =  len(dataset)
        num_data = list(range(len_d))
        np.random.shuffle(num_data)
        split = int(len_d*0.6)
        split_end = int(len_d*0.8)
        subset_train = subsample(num_data[:split])
        subset_valid = subsample(num_data[split:split_end])
        self.trainloader = torch.utils.data.DataLoader(dataset, batch_size = 64, sampler = subset_train)
        self.validloader = torch.utils.data.DataLoader(dataset , batch_size = 64 , sampler = subset_valid) 
        self.len_v = len(self.validloader)
        self.len_t = len(self.trainloader)
        with open("test_idx", "wb") as f:
            pickle.dump(num_data[split_end:], f)
        
    def train_dat(self):    
        opt = optim.SGD(self.model.parameters(), lr = 0.003) 
        criterion = nn.CrossEntropyLoss()
        print ("training.....")
        old_vloss = float('inf')
        epochs = 0
        while True :
            epochs += 1
            #TRAIN LOOP
            running_loss = 0 
            vloss = 0
            self.model.train()
            print ("epoch no ---" , epochs)
            for feat, lab in self.trainloader:
                opt.zero_grad()
                try :
                    output = self.model.forward(feat)
                except Exception as e:
                    continue
                loss = criterion(output ,lab)
                loss.backward()
                opt.step()
                running_loss += loss.item()
            print ("LOSS --->" , running_loss/self.len_t)
            #VALID LOOP
            self.model.eval()
            with torch.no_grad():
                for feat , lab in self.validloader :
                    try :
                        output = self.model.forward(feat)
                    except Exception as e :
                        continue

                    loss = criterion(output , lab)
                    vloss += loss.item()
            if vloss > old_vloss:
                break 
            print ("VALID_LOSS --->" , vloss/self.len_v)
            old_vloss = vloss
            print("EPOCH NO --->" , epochs)
            print ("===========================================\n\n")
        torch.save(self.model.state_dict() , "optim_weights.pth")
        epochs+= 1

if __name__ == '__main__' :
    z = train()
    z.train_dat()
