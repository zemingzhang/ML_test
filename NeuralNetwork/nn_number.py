
import os, sys
import Tkinter
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import scale
from sknn.mlp import Classifier, Layer
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import svm
import numpy as np
from scipy import misc
from sklearn.preprocessing import StandardScaler
#def number(number):
def number():
    digits=load_digits()


    #number=np.array(number)
    #number=number.flatten()
    #print number
    #print(scale(number).shape)
    #print(digits.data[0].shape)
    #print(digits.images[0].shape)
    #data = scale(digits.data)
    #print data.shape
    Xdata = digits.data
    Xdata -= Xdata.min()
    Xdata /= Xdata.max()

    X_train, X_test, y_train, y_test = train_test_split(Xdata, digits.target)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    labels_train = sc.fit_transform(y_train)
    labels_test = sc.transform(y_test)
    #print "train", X_train.shape
    #print "test:", X_test.shape
    #print(X_test[0])
    #print(images_test[0])
    #plt.imshow(images_test[0])
    #plt.show()

    nn = Classifier(layers=[Layer("Sigmoid", units=100),Layer("Softmax")],learning_rate=0.001,n_iter=25)

    nn.fit(X_train, labels_train)
    print nn.score(X_test, y_test)

    #X_test[0]=scale(number)

    predicted = nn.predict(X_test)
    #predicted = svc_model.predict(scale(number))
    print(predicted[0])

    #images_and_predictions = list(zip(images_test, predicted))

    #plt.figure(figsize=(8, 2))
    #for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    #    plt.subplot(1, 4, index + 1)
    #    plt.axis('off')
    #    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    #    plt.title(u'predicted: ' + str(prediction))

    #plt.show()


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

if len(sys.argv) >= 0:
   #print("number check:",sys.argv[1])
   
   #lena = mpimg.imread(sys.argv[1])
   #Pre-process the Data
   #lena = misc.imresize(lena, (8,8))
   #lena=rgb2gray(lena)
   #number1=np.array(lena)
   #number1=number1.flatten()
   #for i in range(len(number1)):
   #    number1[i]=255-number1[i]
   #print(number1)

   #number(number1)
   number()


