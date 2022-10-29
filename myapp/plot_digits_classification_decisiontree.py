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
from sklearn import datasets, svm, metrics,tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from skimage.transform import resize
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score
from collections import Counter


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


def process_decisiontree(split):
    digits = datasets.load_digits()
    



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


    # flatten the images
    n_samples = len(digits.images)
    # print Size of original image
    print(f"Image shape resized : {digits.images[0].shape}")
    data = digits.images.reshape((n_samples, -1))
    data = pd.DataFrame(StandardScaler().fit_transform(data))

        # Split data into  train and test
    X_train, X_test, y_train, y_test = train_test_split(
            data, digits.target, test_size=split, shuffle=True, random_state=1
        )



        # Create a classifier: a support vector classifier
    clf = tree.DecisionTreeClassifier()

        #hyper_params = best_hyperparam['hyperparams']
        #clf.set_params(**hyper_params)

        # Learn the digits on the train subset
    clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)
    accuracy=accuracy_score(y_test, predicted)
    if len(Counter(predicted).keys()) == 10:
            print("Pass")
    print(Counter(predicted).values()) 

    
    
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
    # Split size ( 0.8 to 0.7)
    split_list=[0.1,0.15,0.2]
    overall_accuracy = []
    for g in split_list:
        predicted,accuracy = process_decisiontree(g)
        print(accuracy)
        overall_accuracy.append({"Split":split_list.index(g)+1,"DecisionTree_accuracy":accuracy})
    
    df = pd.DataFrame(overall_accuracy)
    print(tabulate(df, headers='keys', tablefmt='psql'))


if __name__ == "__main__":
    main()