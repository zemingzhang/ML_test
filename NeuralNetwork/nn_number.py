
import os, sys
import Tkinter
#pip install sklearn
#pip install numpy
#pip install scipy
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import scale
#pip install scikit-neuralnetwork
from sknn.mlp import Classifier, Layer
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import *
#pip install matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import svm
import numpy as np
from scipy import misc
from sklearn.preprocessing import StandardScaler
#pip install scikit-image
#https://stackoverflow.com/questions/44865576/python-scikit-image-install-failing-using-pip
from skimage import transform,data
def number(number):
#def number():
    digits=load_digits()

    Xdata = digits.data

    Xdata = scale(Xdata)

    X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(Xdata, digits.target, digits.images, test_size=0.25, random_state=42)

    nn = Classifier(layers=[Layer("Sigmoid", units=100),Layer("Softmax")],learning_rate=0.001,n_iter=25)
    #http://blog.sciencenet.cn/blog-669638-1080739.html
    #C:\Python27\Lib\site-packages\lasagne\layers\pool.py
    nn.fit(X_train, y_train)
    print nn.score(X_test, y_test)

    X_test[0]=scale(number)
    predicted = nn.predict(X_test)
    print("predicted is ",predicted[0])

    #images_and_predictions = list(zip(images_test, predicted))

    #plt.figure(figsize=(8, 2))
    #for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    #    plt.subplot(1, 4, index + 1)
    #   plt.axis('off')
    #    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    #    plt.title(u'predicted: ' + str(prediction))

    #plt.show()


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

if len(sys.argv) >= 0:
   print("number check:",sys.argv[1])
   
   lena = mpimg.imread(sys.argv[1])
   #Pre-process the Data
   lena = transform.resize(lena, (8,8))
   lena=rgb2gray(lena)
   number1=np.array(lena)
   number1=number1.flatten()
   for i in range(len(number1)):
       number1[i]=255-number1[i]
   #print(number1)

   number(number1)
   #number()


