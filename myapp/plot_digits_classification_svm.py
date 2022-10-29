"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from skimage.transform import resize
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")



###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

# Hyperparameter tuning
def hyperparam_tuning(X_train,y_train,X_test,y_test,gamma_list, c_list):
    hyperparam = []
    lst_accuracy = []
    for g in gamma_list:
        for c in c_list:
            h_params = {
                'gamma': g,
                'C': c
            }
            clf_ = svm.SVC()
            clf_.set_params(**h_params)
            clf_.fit(X_train, y_train)
            hyperparam.append(
                {
                    "hyperparams": {"gamma": g, "C": c},
                    "train_acc": metrics.classification_report(y_train, clf_.predict(X_train), output_dict=True)['accuracy'],
                    "test_acc": metrics.classification_report(y_test, clf_.predict(X_test), output_dict=True)['accuracy']
                }
            )

    df = pd.DataFrame(hyperparam)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    
    # print min, max, mean, median of accuracies
    #cols = ['train_acc','test_acc']
    #df[cols].mean()
    #print(df[['train_acc','test_acc']].describe())

    
    best_hyperparam = df.iloc[df['test_acc'].argmax()]
    return best_hyperparam

def process_svm(split,gamma_list,c_list,resolution):
    digits = datasets.load_digits()
    for r in resolution:        
        print(f"Original Image shape  : {digits.images[0].shape}")
        print(f"Expected resolution :   {resolution[r]} X {resolution[r]}")

        _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
        for ax, image, label in zip(axes, digits.images, digits.target):
            ax.set_axis_off()
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            ax.set_title("Training: %i" % label)


        ###############################################################################
        # Classification
        # --------------
        #
        # To apply a classifier on this data, we need to flatten the images, turning
        # each 2-D array of grayscale values from shape ``(8, 8)`` into shape
        # ``(64,)``. Subsequently, the entire dataset will be of shape
        # ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
        # ``n_features`` is the total number of pixels in each image.
        #
        # We can then split the data into train and test subsets and fit a support
        # vector classifier on the train samples. The fitted classifier can
        # subsequently be used to predict the value of the digit for the samples
        # in the test subset.

        #############################################################################
        # Resize of image
        r_images = []
        for i in range(len(digits.images)):
            r_images.append(resize(digits.images[i], (resolution[r],resolution[r]), anti_aliasing=True))
        digits.images = np.array(r_images)

        # flatten the images
        n_samples = len(digits.images)
        # print Size of original image
        print(f"Image shape resized : {digits.images[0].shape}")
        data = digits.images.reshape((n_samples, -1))
        data = pd.DataFrame(StandardScaler().fit_transform(data))

        # Split data into  train and test
        X_train, X_test, y_train, y_test = train_test_split(
                data, digits.target, test_size=split, shuffle=False)


        print(X_train.shape), print(y_train.shape)
        print(X_test.shape), print(y_test.shape)

        best_hyperparam = hyperparam_tuning(X_train,y_train,X_test,y_test,gamma_list, c_list)

        print("Best hyperparam:\n", best_hyperparam)

        # Create a classifier: a support vector classifier
        clf = svm.SVC()

        hyper_params = best_hyperparam['hyperparams']
        clf.set_params(**hyper_params)

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted = clf.predict(X_test)

        print(
            f"Classification report for classifier {clf}:\n"
            f"{metrics.classification_report(y_test, predicted)}\n"
        )

        ###############################################################################
        # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
        # true digit values and the predicted digit values.

        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
        disp.figure_.suptitle("Confusion Matrix")
        print(f"Confusion matrix:\n{disp.confusion_matrix}")
        plt.show()
        accuracy = metrics.classification_report(y_test, clf.predict(X_test), output_dict=True)['accuracy']
        return predicted,accuracy

def main():
    # Select image resolution ( Options "NO_CHANGE","4*4","16*16","32*32")
    resolution ={"NO_CHANGE":8}      
    
    # Hyperparameter list for iteration
    gamma_list = [0.00005,0.0001,0.0005,0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    c_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    split_list=[0.1,0.15,0.2]
    overall_accuracy = []
    for g in split_list:
        predicted,accuracy = process_svm(g, gamma_list,c_list, resolution)    
        overall_accuracy.append({"Split":split_list.index(g)+1,"SVM_accuracy":accuracy})
    
    df = pd.DataFrame(overall_accuracy)
    print(tabulate(df, headers='keys', tablefmt='psql'))

if __name__ == "__main__":
    main()