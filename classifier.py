import os
import re
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans
import time
import sys
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import seaborn as sn
BARCHE = ['Lanciafino10mBianca','Lanciafino10mMarrone', 'Motobarca','Mototopo','Patanella','VaporettoACTV']
#BARCHE = ['Patanella', 'Motobarca']
y_test = pd.read_pickle('y_test_resnet.pkl')
X_test = pd.read_pickle('X_test_resnet.pkl')

X_train = pd.read_pickle('X_train_resnet.pkl')
y_train = pd.read_pickle('y_train_resnet.pkl')
print(X_train.shape)
#sys.exit()
start = time.clock()

svclassifier = SVC(kernel='linear', C=1)  
svclassifier.fit(X_train, y_train) 
print('Fit time: '+str(time.clock() - start))
intermediate = time.clock()

y_pred = svclassifier.predict(X_test) 
print('Predict time: '+ str(time.clock() - intermediate))
con_mat = confusion_matrix(y_test,y_pred)
print(con_mat)

print(classification_report(y_test,y_pred))  
df_cm = pd.DataFrame(con_mat, index = BARCHE, columns =BARCHE)
plt.figure(figsize = (10,7))
sn.set(font_scale=3)
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
plt.ion()
plt.show(block=False)
print(accuracy_score(y_test, y_pred))
input('press <ENTER> to continue')


