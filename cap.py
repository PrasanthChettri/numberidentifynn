import cv2
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
import  models 

#IMPLEMENTING A SHITTY NUMBER SEGMENTER
#TRYING TO MIMIC THE MNIST DATASET

class image_load:
    def __init__(self, img = None):
        if img :
            self.img = img
        else :
            self.img = cv2.imread('canvas.jpg' ,cv2.IMREAD_GRAYSCALE)
        self.x = self.img.shape[-1]//3
        self.y = 24
        self.img = cv2.resize(self.img  ,(self.x , self.y), interpolation = cv2.INTER_AREA)
        print (self.img.shape)

    #SEGMENT THOSE NUMBERS BRO
    def get_segment(self , show = False):
        refrence = self.img[: , 0]
        digit_matrix = []
        temp = [refrence]
        prei = [None]
        for i in self.img.transpose() :
            if (i == refrence).all(): 
                if len(temp) >  1:
                    temp = self.ref(temp)
                    digit_matrix.append(temp)
                temp = []
            else:
                temp.append(i)
        if show :
            for i , j in enumerate(digit_matrix):
                cv2.imshow('number' + str(i) , j)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else :
            return digit_matrix

    #PROCESS THEM DATA BRO
    def ref(self , mat):
        mat = 255 -np.array(mat).transpose()
        add_row = mat.shape[1] 
        add_row = 28 - add_row
#        print (np.zeros((28, 28 - add_row)))
#        exit()
        if add_row > 0 :
            a = int (add_row / 2)
            mat = np.concatenate((mat , np.zeros((self.y, a))) , axis = 1)
            mat = np.concatenate((np.zeros((self.y, add_row- a)) ,mat) , axis = 1)
        else : 
            mat = cv2.resize(mat , (self.y , 28) ,interpolation =  cv2.INTER_AREA)
        b = 28 - self.y
        mat = np.concatenate((mat , np.zeros((b//2 , 28))) , axis = 0)
        mat = np.concatenate((np.zeros((b - b//2  ,  28)) ,mat) , axis = 0)
        return mat

    def show(self):
        cv2.imshow('img' , self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test(model):
    model.eval()
    from torchvision import transforms , datasets
    from torch.utils.data.sampler import SubsetRandomSampler as subsample
    import pickle
    with open("test_idx" , "rb") as f:
        index  = pickle.load(f)
    with torch.no_grad():
        idx = subsample(index) 
        softmax = nn.Softmax(dim = 1)
        transform = transforms.Compose([
                            transforms.RandomCrop((28 , 28)) ,
                            transforms.ToTensor() , 
                            transforms.Normalize((0.5,) , (0.5,)), 
                            ])
        trainset = datasets.MNIST('MNIST_data/'  , train = True , transform = transform)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size = 1 , sampler = idx)
        i = 0 
        for feat , label in dataloader :
            plt.imshow(feat[0][0])
            plt.show()
            if input() != '' :
                exit()
            output = model.forward(feat)
            output = softmax(output)
            k =  output.topk(1)[1].data
            if k == label.data :
                i += 1
        print ((i/len(dataloader))*100 , "%")     
    exit()

if __name__ ==  '__main__':

    z =  image_load()
    model = models.model2(batch_size = 1)
    model.load_state_dict(torch.load("optim_weights2.pth"))
    test(model)
    feat = torch.tensor(z.get_segment()).float()
    softmax = nn.Softmax(dim = 1)
    number = []
    for i in feat :
        output = model.forward(i.view(1 ,1 , *i.shape))
        output = softmax(output)
        number.append(output.topk(1)[1].item())

    print ("NUMBER IS ")
    print (* number , sep = '')
