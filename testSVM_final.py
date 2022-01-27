
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
import optuna
import pickle
from tqdm import tqdm
from keras.backend import clear_session



def img_list(path):    
    return [os.path.join(path, f) for f in os.listdir(path)[0:2]]



def get_data(test_set):

    data_paths=[test_set] 
    image_paths = []
    image_classes = []
    for data_path in data_paths:
        categories=os.listdir(data_path)[0:2]
        labels=[i for i in range(len(categories))]
        label_dict=dict(zip(categories,labels)) #empty dictionary
        # print(label_dict)
        # print(categories)
        # print(labels)
        
        for category in categories:
            path = os.path.join(data_path, category)
            for img in tqdm(os.listdir(path)):
     
                image_paths.append(os.path.join(path, img))
                image_classes.append(label_dict[category])
        

    return image_paths, image_classes


def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
            x, y = kp.pt
            plt.imshow(cv2.circle(vis, (int(x), int(y)), 2, color))


def test_set(image_paths_test):

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

def score_SVM_model(filename, X_test, y_test):
# load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, y_test)
    print(result)
    return loaded_model



# def predict_accuracy():
#     # test_features=stdslr.transform(test_features)
#     y_pred = clf.predict(X_val)
#     # calculate accuracy
#     accuracy = accuracy_score(y_val, y_pred)
#     print('Model accuracy is: ', accuracy)
#     return y_pred



# def get_features(image_paths, k=200, iterations=1):
#     #orb features
#     des_list=[]
#     orb=cv2.ORB_create()
#     for image_pat in image_paths:
#         im=cv2.imread(image_pat)
#         kp=orb.detect(im,None)
#         keypoints,descriptor= orb.compute(im, kp)
#         des_list.append((image_pat,descriptor))
#     descriptors=des_list[0][1]
#     for image_path,descriptor in des_list[1:]:
#         descriptors=np.vstack((descriptors,descriptor))
#     descriptors_float=descriptors.astype(float)
#     #bow 
#     # print(descriptors_float)
#     voc,variance=kmeans(descriptors_float,k, iterations) #1 = number of iterations can modify 
#     im_features=np.zeros((len(image_paths),k),"float32")
#     for i in range(len(image_paths)):
#         words,distance=vq(des_list[i][1],voc)
#         for w in words:
#             im_features[i][w]+=1
    
#     return im_features

# def train_validation(im_features, image_classes, test_size=0.25):
#     stdslr=StandardScaler().fit(im_features)
#     im_features=stdslr.transform(im_features)
#     X_train, X_val, y_train, y_val= train_test_split(im_features, image_classes, test_size=test_size)

#     return X_train, X_val, y_train, y_val

# X_train, X_val, y_train, y_val=train_validation(im_features,image_classes)




# def svm_model(X_train,y_train):
#     clf=SVC(kernel='linear',probability=True)
#     clf.fit(X_train,np.array(y_train))
#     return clf

# clf=svm_model(X_train,y_train)






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

# roc_auc(clf, X_val,y_val, y_proba)


def plot_cmatrix(clf, X_val, y_test):
    plot_confusion_matrix(clf, X_val, y_test)  
    plt.show()

# clf, X_val, y_test=plot_cmatrix()


def objective(trial):
    clear_session()
    
       #image paths
    image_paths, image_classes = get_data(two_training_folders=['A', 'B'])
    #extract features
    
    k = trial.suggest_discrete_uniform('k', 500, 4000, 500)
    # print(k)

    iterations = trial.suggest_discrete_uniform('iterations', 5, 20, 5)
    # print(iterations)
    im_features=get_features(image_paths, k=int(k), iterations=int(iterations))
    
    #validation split
    X_train, X_val, y_train, y_val=train_validation(im_features, image_classes, test_size=0.25)


    # kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    kernel = 'rbf'
    regularization = trial.suggest_uniform('svm-regularization', 1, 5)
    degree = trial.suggest_discrete_uniform('degree', 3, 7, 1)
    clf = SVC(kernel=kernel, C=regularization, degree=degree)

    #svm model
    # clf=SVC(kernel='linear', probability=True)
    clf.fit(X_train,np.array(y_train))
    #run predict
    y_pred = clf.predict(X_val)
    # calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print('Model accuracy is: ', accuracy)
    
    
    return accuracy



def main():
    #image paths
    # image_paths, image_classes = get_data(two_training_folders=['A', 'B'])
    # #extract features
    
    # im_features=get_features(image_paths)
    
    # #validation split

    # #svm parameters
    # clf=svm_model(X_train,y_train)
    # y_pred=predict_accuracy()
    # #scoring
    # roc_auc(clf, X_val,y_val, y_proba)
    # clf, X_val, y_test=plot_cmatrix()


    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=None)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

main()
