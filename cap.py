import cv2
import numpy as np
import torch
import torchvision 
from torchvision import transforms , datasets
from torch import nn
from matplotlib import pyplot as plt
from models import model1 as mod

#IMPLEMENTING A SHITTY NUMBER SEGMENTER
class image_load:
    def __init__(self, img = None):
        if img :
            self.img = img
        else :
            self.img = cv2.imread('go.jpg' ,cv2.IMREAD_GRAYSCALE)
        self.img = cv2.resize(self.img , (150 , 28) , interpolation = cv2.INTER_AREA)

#SEGMENT THOSE LETTERS BRO
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

    def ref(self , mat):
        mat = 255 -np.array(mat).transpose()
        add_row = mat.shape[1] 
        add_row = 28 - add_row
#        print (np.zeros((28, 28 - add_row)))
#        exit()
        if add_row > 0 :
            a = int (add_row / 2)
            mat = np.concatenate((mat , np.zeros((28, a))) , axis = 1)
            mat = np.concatenate((np.zeros((28, add_row- a)) ,mat) , axis = 1)
        else : 
            mat = cv2.resize(mat , (28 , 28) ,interpolation =  cv2.INTER_AREA)
        return mat

    def show(self):
        cv2.imshow('img' , self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

            
if __name__ ==  '__main__':
    z =  image_load()
    model = mod(batch_size = 1)
    model.load_state_dict(torch.load("optim_weights.pth"))
    feat = torch.tensor(z.get_segment()).float()
    softmax = nn.Softmax(dim = 1)
    number = []
    for i in feat :
        output = model.forward(i.view(1 ,1 , *i.shape))
        output = softmax(output)
        number.append(output.topk(1)[1].item())
    print (* number , sep = '')
