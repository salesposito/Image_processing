import cv2
import numpy as np
import os
import pylab as pl
from sklearn.metrics import confusion_matrix, accuracy_score #sreeni
import joblib
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn import preprocessing

with open('finalized_svmmodel.pkl', 'rb') as f:
    clf, classes_names, stdSlr, k, voc = pickle.load(f)

def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
            x, y = kp.pt
            plt.imshow(cv2.circle(vis, (int(x), int(y)), 2, color))


def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.show()

data = []

for perturb_type in os.listdir('test'):
    # print(perturb_type)
    perturb_type_path = os.path.join('test', perturb_type)
    for perturb_level in os.listdir(perturb_type_path):
        # print(perturb_level)
        test_path = os.path.join(perturb_type_path, perturb_level)


        # test_path = r"test\gaussian_blurring\0"
        testing_names = os.listdir(test_path)

        image_paths = []
        image_classes = []
        class_id = 0


        for testing_name in testing_names:
            dir = os.path.join(test_path, testing_name)
            class_path = imglist(dir)
            image_paths+=class_path
            image_classes+=[class_id]*len(class_path)
            class_id+=1

        des_list = []

        orb = cv2.ORB_create()

        for image_path in image_paths:
            im2 = cv2.imread(image_path)
            kp = orb.detect(im2, None)
            keypoints, descriptor= orb.compute(im2, kp)
            des_list.append((image_path,descriptor))

        # for image_path in image_paths:
        #     im = cv2.imread(image_path)
        #     kpts, des = brisk.detectAndCompute(im, None)
        #     des_list.append((image_path, des))

        descriptors = des_list[0][1]
        for image_path, descriptor in des_list[1:]: #was des_list[0:]
            try:
                descriptors = np.vstack((descriptors, descriptor))
                descriptors_float = descriptors.astype(float)
            except Exception as e:
                print(e)

        from scipy.cluster.vq import kmeans, vq
        k=200
        voc, variance = kmeans(descriptors_float, k, 1)


        from scipy.cluster.vq import vq    
        test_features = np.zeros((len(image_paths), k), "float32")
        for i in range(len(image_paths)):
            # print(i)
            words, distance = vq(des_list[i][1],voc)
            # print(words)
            for w in words:
                # print(w)
                test_features[i][w] += 1

        scaler = StandardScaler()
        X_test = test_features
        y_test = image_classes

        X_test = scaler.fit_transform(X_test)
        X_test = scaler.transform(X_test)

        min_max_scaler = preprocessing.MinMaxScaler()
        X_test = min_max_scaler.fit_transform(X_test)

        # print(X_test, y_test)

        # score = loaded_model.score(X_test, y_test)

        y_pred = clf.predict(X_test)
        # calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        
        data.append([test_path, accuracy])

        print(data)

with open('testing_results_svm.pkl', 'wb') as f:
    pickle.dump(data, f)
