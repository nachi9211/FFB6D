import pickle

import numpy as np
from matplotlib import pyplot as plt
import cv2

img = '/media/nachiket/Windows/Users/All Users/Documents/Datasets/LineMod/Linemod_preprocessed/renders/ape/13.pkl'

img_clr = '/media/nachiket/Windows/Users/All Users/Documents/Datasets/YCB/YCB_Video_Dataset/data_syn/000000-color.png'
img_lbl = '/media/nachiket/Windows/Users/All Users/Documents/Datasets/YCB/YCB_Video_Dataset/data_syn/000000-label.png'


data_lbl = cv2.imread(img_lbl)






#data_lbl[nonzero] = data_lbl[nonzero]*11

#with open(img) as f:
    #data_lbl = pickle.load(f)

#print('Keys in loaded file: ',data.keys())

#for k in data.keys():
#    plt.imshow(data[k])
#    plt.show()

plt.imshow(data_lbl)
plt.show()


