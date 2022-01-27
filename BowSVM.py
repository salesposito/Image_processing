
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import pylab as pl
from sklearn.metrics import confusion_matrix,accuracy_score
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from scipy.cluster.vq import vq
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing


def img_list(path):    
    return [os.path.join(path, f) for f in os.listdir(path)]


def training_data():
    train_data=[]
    
    train_path="catdog"
    class_names=os.listdir(train_path)
    image_paths=[]
    image_classes=[]


    for training_name in class_names:
        dir_=os.path.join(train_path,training_name)
        class_path=img_list(dir_)
        image_paths+=class_path
        

    image_classes_0=[0]*(len(image_paths)//2)
    image_classes_1=[1]*(len(image_paths)//2)
    image_classes=image_classes_0+image_classes_1

    for i in range(len(image_paths)):
        train_data.append((image_paths[i],image_classes[i]))
    random.shuffle(train_data)
    return image_paths, image_classes

image_paths, image_classes =training_data()
print(len(image_paths), len(image_classes))


def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
            x, y = kp.pt
            plt.imshow(cv2.circle(vis, (int(x), int(y)), 2, color))


def detect_features(image_paths):
    des_list=[]
    orb=cv2.ORB_create()
    for image_pat in image_paths:
        im=cv2.imread(image_pat)
        kp=orb.detect(im,None)
        keypoints,descriptor= orb.compute(im, kp)
        des_list.append((image_pat,descriptor))
    descriptors=des_list[0][1]
    for image_path,descriptor in des_list[1:]:
        descriptors=np.vstack((descriptors,descriptor))
    descriptors_float=descriptors.astype(float)
    return des_list, descriptors_float

des_list, descriptors_float=detect_features(image_paths)
print(descriptors_float)
exit(0)
def bag_of_words(des_list, descriptors_float,image_paths):

    k=200
    voc,variance=kmeans(descriptors_float,k,1)
    im_features=np.zeros((len(image_paths),k),"float32")
    for i in range(len(image_paths)):
        words,distance=vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w]+=1
    return im_features

im_features=bag_of_words(des_list, descriptors_float,image_paths)


def train_validation(im_features,image_classes):
    stdslr=StandardScaler().fit(im_features)
    im_features=stdslr.transform(im_features)
    X_train, X_val, y_train, y_val= train_test_split(im_features, image_classes, test_size=0.25)

    return X_train, X_val, y_train, y_val

X_train, X_val, y_train, y_val=train_validation(im_features,image_classes)


def svm_model(X_train,y_train):
    clf=SVC(kernel='linear',probability=True)
    clf.fit(X_train,np.array(y_train))
    return clf

clf=svm_model(X_train,y_train)


def test_set():

    des_list_test=[]
    for image_pat in image_paths_test:
        image=cv2.imread(image_pat)
        kp=orb.detect(image,None)
        keypoints_test,descriptor_test= orb.compute(image, kp)
        des_list_test.append((image_pat,descriptor_test))
    test_features=np.zeros((len(image_paths_test),k),"float32")
    for i in range(len(image_paths_test)):
        words,distance=vq(des_list_test[i][1],voc)
        for w in words:
            test_features[i][w]+=1


def predict_accuracy():
    # test_features=stdslr.transform(test_features)
    y_pred = clf.predict(X_val)
    # calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print('Model accuracy is: ', accuracy)
    return y_pred
y_pred=predict_accuracy()


# # prepare the cross-validation procedure
# cv = KFold(n_splits=3, random_state=1, shuffle=True)
# # evaluate model
# scores = cross_val_score(clf,X_val, y_val, scoring='accuracy', cv=cv)
# # report performance
# print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


def roc_auc(clf, X_val,y_val, y_proba):

    # predict probabilities for X_test using predict_proba
    probabilities = clf.predict_proba(X_val)

    # select the probabilities for label 1.0
    y_proba = probabilities[:, 1]

    # calculate false positive rate and true positive rate at different thresholds
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_proba, pos_label=1)

    # calculate AUC
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    # plot the false positive rate on the x axis and the true positive rate on the y axis
    roc_plot = plt.plot(false_positive_rate,
                        true_positive_rate,
                        label='AUC = {:0.2f}'.format(roc_auc))

    plt.legend(loc=0)
    plt.plot([0,1], [0,1], ls='--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return y_proba

roc_auc(clf, X_val,y_val, y_proba)


def plot_cmatrix(clf, X_val, y_test):
    plot_confusion_matrix(clf, X_val, y_test)  
    plt.show()

clf, X_val, y_test=plot_cmatrix()



