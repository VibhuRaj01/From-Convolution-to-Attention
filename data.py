import numpy as np
import cv2
import os
import random
from sklearn.model_selection import train_test_split


def data_preprocess(categories, dataset_path):
    img_size=150
    data=[]
    for category in categories:
        folder=os.path.join(dataset_path,category)
        label=categories.index(category)
        for img in os.listdir(folder):
            img_path=os.path.join(folder,img)
            img_arr=cv2.imread(img_path)
            img_arr=cv2.resize(img_arr,(img_size, img_size))
            data.append([img_arr, label])
    random.shuffle(data)
    return data 


def create_input_data(data):
    x=[] 
    y=[]
    for i,j in data:
        x.append(i)
        y.append(j)
    x = np.array(x).astype('float32')
    # x = x/255
    y = np.array(y)
    y = np.asarray(y).astype('float32').reshape((-1,1))
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=2)
    return X_train, X_test, y_train, y_test

#Set the path to dataset directories
categories=['malignant', 'benign']
dataset_path=r'path_to_data_folder'

data = data_preprocess(categories, dataset_path)
print(len(data))
X_train, X_test, y_train, y_test = create_input_data(data)
