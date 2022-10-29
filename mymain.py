# examples of unit test cases 
import pytest
from myapp.plot_digits_classification_decisiontree import process_decisiontree
from myapp.plot_digits_classification_svm import process_svm
from collections import Counter
import pandas as pd
import numpy as np
from skimage.transform import resize
from tabulate import tabulate

def main():
    # Select image resolution ( Options "NO_CHANGE","4*4","16*16","32*32")
    resolution ={"NO_CHANGE":8}      
    
    # SVM processing
    # Hyperparameter list for iteration
    gamma_list = [0.00005,0.0001,0.0005,0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    c_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    split_list=[0.1,0.15,0.2,0.25,0.3]
    overall_accuracy_svm = []
    for g in split_list:
        predicted,accuracy = process_svm(g, gamma_list,c_list, resolution)    
        overall_accuracy_svm.append({"Split":split_list.index(g)+1,"SVM_accuracy":accuracy})
    
    df_svm = pd.DataFrame(overall_accuracy_svm)
    #print(tabulate(df_svm, headers='keys', tablefmt='psql'))
    
    # Decision Tree processing
    overall_accuracy_decisiontree = []
    for g in split_list:
        predicted,accuracy = process_decisiontree(g)
        print(accuracy)
        overall_accuracy_decisiontree.append({"Split":split_list.index(g)+1,"DecisionTree_accuracy":accuracy})
    
    df_decisiontree = pd.DataFrame(overall_accuracy_decisiontree)
    #print(tabulate(df_decisiontree, headers='keys', tablefmt='psql'))
    
    df = pd.merge(df_svm, df_decisiontree,left_on='Split',right_on='Split')
    df.loc[len(df.index)] = ['Mean', df_svm.mean()['SVM_accuracy'], df_decisiontree.mean()['DecisionTree_accuracy']]
    df.loc[len(df.index)] = ['Std', df_svm.std()['SVM_accuracy'], df_decisiontree.std()['DecisionTree_accuracy']]     
    print(tabulate(df, headers='keys', tablefmt='psql'))
    
if __name__ == "__main__":
    main()