
import os, sys
import Tkinter
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import scale
#from sknn.mlp import Classifier, Layer
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
#from skimage import transfrom
from sklearn import svm
import numpy as np
from scipy import misc
#https://juejin.im/post/591ed703a22b9d00585e0e72
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def number(number):
#def number():
    digits=load_digits()
    #print(digits.data.shape)
    print("data shape is :",digits.data.shape)
    print np.unique(digits.target)
    #print digits.images[0].shape
    #print number.shape
    #plt.imshow(digits.images[0])
    #plt.show()


    #images_and_labels = list(zip(digits.images, digits.target))

    #plt.figure(figsize=(8, 6))
    #for index, (image, label) in enumerate(images_and_labels[:8]):
    #    plt.subplot(2, 4, index + 1)
    #    plt.axis('off')
    #    plt.imshow(image, cmap=plt.cm.gray_r,interpolation='nearest')
    #    plt.title(u'Number:' + str(label))

    #plt.show()



    #pca = PCA(n_components=2)
    #reduced_data_pca = pca.fit_transform(digits.data)
    #print reduced_data_pca.shape




    #colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    #plt.figure(figsize=(8, 6))
    #for i in range(len(colors)):
    #    x = reduced_data_pca[:, 0][digits.target == i]
    #    y = reduced_data_pca[:, 1][digits.target == i]
    #    plt.scatter(x, y, c=colors[i])
    #plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.xlabel(u'first')
    #plt.ylabel(u'second')
    #plt.title(u"PCA")
    #plt.show()
    #number=np.array(number)
    #number=number.flatten()
    #print number
    print(scale(number).shape)
    print(digits.data[0].shape)
    print(digits.images[0].shape)
    data = scale(digits.data)
    print data.shape

    X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size=0.25, random_state=42)

    print "train", X_train.shape
    print "test:", X_test.shape
    #print(X_test[0])
    #print(images_test[0])
    #plt.imshow(images_test[0])
    #plt.show()


    svc_model = svm.SVC(gamma=0.001, C=10, kernel='rbf')
    svc_model.fit(X_train, y_train)
    print svc_model.score(X_test, y_test)
    print(X_test[0])
    X_test[0]=scale(number)
    #X_test[0]=(number)
    print(X_test[0])
    print(images_test[1])
    predicted = svc_model.predict(X_test)
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

if len(sys.argv) >= 1:
   print("number check:",sys.argv[1])
   
   
   #im = Image.open(sys.argv[1])
   #im_array = np.array(im)
   #print(im_array.shape)
   
   
   lena = mpimg.imread(sys.argv[1])
   #Pre-process the Data
   lena = misc.imresize(lena, (8,8))
   lena=rgb2gray(lena)
   print(lena)
   number1=np.array(lena)
   number1=number1.flatten()
   for i in range(len(number1)):
       number1[i]=255-number1[i]
   print(number1)
   plt.imshow(lena)
   #plt.show()
   number(number1)
   #number(lena_new_sz)


