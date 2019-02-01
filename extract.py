import os
import re
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import time
from tqdm import tqdm
from keras.layers import Dense, Flatten, Input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input

from keras.preprocessing import image
from keras.models import Model
import pickle


import seaborn as sn

BARCHE = ['Patanella', 'Motobarca','Lanciafino10mBianca','Lanciafino10mMarrone','Mototopo','VaporettoACTV']
#BARCHE = ['Patanella', 'Motobarca']

def extract_features(file_dir):
    
    #base_model = VGG19(weights='imagenet',pooling=max, include_top = False)
    #base_model = InceptionResNetV2(weights='imagenet',input_shape=(224, 224, 3),pooling=max, include_top = False)
    base_model = InceptionV3(weights='imagenet',pooling=max, include_top = False)
    #base_model = ResNet50(weights='imagenet', pooling=max, include_top = False)
    input = Input(shape=(224,224,3),name = 'image_input')
    x = base_model(input)
    model = Model(inputs=input, outputs=x)
    nn_feature_list = []
    classe = []
    list_files = [file_dir+f for f in os.listdir(file_dir)]
    print('Elaborating training files')
    for idx, dirname in enumerate(list_files):
        list_images = [dirname+'/'+f for f in os.listdir(dirname) if re.search('jpg|JPG', f)]
        class_name = list_files[idx]
        class_name = class_name[42:]
        print(class_name)
        for fname in tqdm(list_images):
            img = image.load_img(fname, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            nn_feature = model.predict(img_data)
            nn_feature_np = np.array(nn_feature)
            nn_feature_list.append(nn_feature_np.flatten())
            classe.append(class_name)  
    se = pd.Series(classe)
    nn_feature_list_np = np.array(nn_feature_list)
    np_data = pd.DataFrame(nn_feature_list_np)
    np_data['Class'] = se.values
    return np_data
    

def extract_features_test(file_dir):

    #base_model = VGG19(weights='imagenet',pooling=max, include_top = False)
    #base_model = InceptionResNetV2(weights='imagenet',input_shape=(224, 224, 3),pooling=max, include_top = False)
    base_model = InceptionV3(weights='imagenet',pooling=max, include_top = False)
    #base_model = ResNet50(weights='imagenet', pooling=max, include_top = False)
    input = Input(shape=(224,224,3),name = 'image_input')
    x = base_model(input)
    model = Model(inputs=input, outputs=x)
    nn_feature_list = []
    classe_test= []
    file_dir_test = file_dir+'ground_truth.txt'
    Barche_considerate = BARCHE
    start_time = time.clock()
    print('Elaborating test images')
    with open(file_dir_test, 'r') as f:
        x = f.readlines()
        for line in tqdm(x):
            name_img = line[:21]
            tipo = line[26:]
            tipo = tipo.replace(' ','')
            tipo = tipo.replace(':','')
            tipo = tipo.strip()
            
            if tipo in Barche_considerate:
                fname = file_dir+name_img+'.jpg'
                img = image.load_img(fname, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                nn_feature = model.predict(img_data)
                nn_feature_np = np.array(nn_feature)
                nn_feature_list.append(nn_feature_np.flatten())
                classe_test.append(tipo)  
                

    se = pd.Series(classe_test)
    nn_feature_list_np = np.array(nn_feature_list)
    np_data = pd.DataFrame(nn_feature_list_np)
    np_data['Class'] = se.values
    return np_data    

file_dir_test = 'C:/Users/Clinc/Documents/Università/ML/sc5-2013-Mar-Apr-Test-20130412/'
file_dir = 'C:/Users/Clinc/Documents/Università/ML/im/'

start_time = time.clock()
pd_data = extract_features(file_dir)
pd_data_test = extract_features_test(file_dir_test)
print('Time used: '+str(time.clock()-start_time))

y_train = pd_data['Class']
y_train.to_pickle('y_train_inception.pkl')
X_train = pd_data.drop('Class', 1)
X_train.to_pickle('X_train_inception.pkl')

y_test = pd_data_test['Class']
y_test.to_pickle('y_test_inception.pkl')
X_test = pd_data_test.drop('Class', 1)
X_test.to_pickle('X_test_inception.pkl')








