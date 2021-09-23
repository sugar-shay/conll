# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 08:39:18 2021

@author: Shadow
"""

import numpy as np
import pandas as pd
from datasets import load_dataset

def get_num_slots(labels):
    
    num_slots = []
    for seq_label in labels:
        num_slots.append(len(seq_label))
        
    return num_slots

def get_conll_data(train_size = 14041, val_size = 3250, test_size = 3453):
    
    train_size = 'train[:' + str(train_size) +']'
    val_size = 'validation[:' + str(val_size) +']'
    test_size = 'test[:' + str(test_size) +']'
    
    train_ds, val_ds, test_ds = load_dataset('conll2003', split=[train_size, val_size, test_size])


    train_labels = train_ds['ner_tags']
    val_labels = val_ds['ner_tags']
    test_labels = test_ds['ner_tags']
    
    train_text = train_ds['tokens']
    val_text = val_ds['tokens']
    test_text = test_ds['tokens']
    
    unique_labels = np.unique(train_labels)

    train_slots = get_num_slots(train_labels)
    val_slots = get_num_slots(val_labels)
    test_slots = get_num_slots(test_labels)
    
    train_data = pd.DataFrame(data={'text':train_text,
                                    'labels':train_labels,
                                    'num_slots':train_slots})

    val_data = pd.DataFrame(data={'text':val_text,
                                    'labels':val_labels,
                                    'num_slots':val_slots})
    
    test_data = pd.DataFrame(data={'text':test_text,
                                'labels':test_labels,
                                'num_slots':test_slots})
    
    return train_data, val_data, test_data