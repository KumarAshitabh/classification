# examples of unit test cases 
import pytest
from myapp.plot_digits_classification_decisiontree import process_decisiontree
from myapp.plot_digits_classification_svm import process_svm
from collections import Counter
import pandas as pd
import numpy as np
from skimage.transform import resize
from tabulate import tabulate
import argparse

def main():
  
    parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--clf_name", help="select model")
    parser.add_argument("--randomseed", help="select random seed")
    args = parser.parse_args()
    config = vars(args)
    print(config)

    split_list=0.3
    randomseed = int(config['randomseed'])
    if config['clf_name']=='SVM':
        # SVM processing
        # Hyperparameter list for iteration
        gamma_list = [0.1]
        c_list = [5]
        # Select image resolution ( Options "NO_CHANGE","4*4","16*16","32*32")
        resolution ={"NO_CHANGE":8} 
        split_list=0.3
        predicted,accuracy,X_train,X_test = process_svm(split_list, gamma_list,c_list, resolution,randomseed) 
        with open('report/svm.txt', 'w+') as f:
            f.write(str(accuracy))

    elif config['clf_name']=='Tree': 
        # Decision Tree processing
        overall_accuracy_decisiontree = []
        predicted,accuracy,X_train,X_test = process_decisiontree(split_list,randomseed)
        with open('report/tree.txt', 'w+') as f:
            f.write(str(accuracy))

    
if __name__ == "__main__":
    main()