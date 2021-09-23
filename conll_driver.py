# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:08:25 2021

@author: Shadow
"""

from tokenize_data import *
from datasets import load_dataset
from pretrain_lit_ner import *
from data_preprocess import *
from sklearn.metrics import classification_report
import pickle 

def main():

    train_data, val_data, test_data, unique_labels = get_conll_data()
    
    print()
    print('# of Training Examples: ', train_data.shape[0])
    print('# of Val Examples: ', val_data.shape[0])
    print('# of Test Examples: ', test_data.shape[0])
    
    encoder_name = 'bert-base-uncased'
    
    tokenizer = NER_tokenizer(max_length=64, tokenizer_name = encoder_name, unique_labels=unique_labels)
    
    train_dataset = tokenizer.tokenize_and_encode_labels(train_data['text'].tolist(), 
                                                         train_data['labels'].tolist(), 
                                                         train_data['num_slots'].tolist())
    
    val_dataset = tokenizer.tokenize_and_encode_labels(val_data['text'].tolist(), 
                                                         val_data['labels'].tolist(), 
                                                         val_data['num_slots'].tolist())
    
    test_dataset = tokenizer.tokenize_and_encode_labels(val_data['text'].tolist(), 
                                                         val_data['labels'].tolist(), 
                                                         val_data['num_slots'].tolist())
    
    model = PRETRAIN_LIT_NER(num_classes = unique_labels.shape[0], 
                     hidden_dropout_prob=.1,
                     attention_probs_dropout_prob=.1,
                     encoder_name = encoder_name,
                     save_fp='bert_conll.pt')
    
    model = train_LitModel(model, train_dataset, val_dataset, max_epochs=10, batch_size=32, patience = 2, num_gpu=1)
    
    complete_save_path = 'results/full_train'
    if not os.path.exists(complete_save_path):
        os.makedirs(complete_save_path)
         
    #saving train stats
    with open(complete_save_path+'/bert_train_stats.pkl', 'wb') as f:
        pickle.dump(model.training_stats, f)
        
    with open(complete_save_path+'/token_inputs.pkl', 'wb') as f:
        pickle.dump(model.token_inputs, f)
    
    #reloading the model for testing
    model = PRETRAIN_LIT_NER(num_classes = unique_labels.shape[0], 
                     hidden_dropout_prob=.1,
                     attention_probs_dropout_prob=.1,
                     encoder_name = encoder_name,
                     save_fp='bert_conll.pt')
    
    model.load_state_dict(torch.load('bert_conll.pt'))
    
    cr = model_testing(model, test_dataset)
    
    cr_df = pd.DataFrame(cr).transpose()
    print(cr_df)
    
    with open(complete_save_path+'/bert_test_stats.pkl', 'wb') as f:
            pickle.dump(cr, f)
    
if __name__ == "__main__":
    
    main()