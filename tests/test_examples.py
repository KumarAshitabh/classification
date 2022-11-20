# examples of unit test cases 
import pytest
from ..myapp.plot_digits_classification_svm import process_svm
from collections import Counter

    
def test_check_svm_randomseed_sameseed():
    gamma_list = [0.1]
    c_list = [5]
    # Select image resolution ( Options "NO_CHANGE","4*4","16*16","32*32")
    resolution ={"NO_CHANGE":8} 
    split_list=0.3
    randomseed = 42
    predicted,accuracy,X_train1,X_test1 = process_svm(split_list, gamma_list,c_list, resolution,randomseed) 
    randomseed = 42
    predicted,accuracy,X_train2,X_test2 = process_svm(split_list, gamma_list,c_list, resolution,randomseed) 
    #check if Xtest frame is same
    assert X_train1.equals(X_train2)


    
def test_check_svm_randomseed_differentseed():
    gamma_list = [0.1]
    c_list = [5]
    # Select image resolution ( Options "NO_CHANGE","4*4","16*16","32*32")
    resolution ={"NO_CHANGE":8} 
    split_list=0.3
    randomseed = 42
    predicted,accuracy,X_train1,X_test1 = process_svm(split_list, gamma_list,c_list, resolution,randomseed) 
    randomseed = 32
    predicted,accuracy,X_train2,X_test2 = process_svm(split_list, gamma_list,c_list, resolution,randomseed) 
    #check if Xtest frame is same
    assert not X_train1.equals(X_train2)
 
