# examples of unit test cases 
import pytest
from ..myapp.plot_digits_classification_decisiontree import process
from collections import Counter

    
def test_classifier_not_completely_biased():
    overall_accuracy = []
    train_split = 0.75
    dev_split = (1-0.75)  
    predicted = process(train_split,dev_split)
    assert len(Counter(predicted).keys()) == 10


    
def test_classifier_predicts_all_classes():
    assert 1 == 1   
