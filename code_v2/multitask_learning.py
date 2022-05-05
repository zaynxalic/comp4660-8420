from preprocessing_dataset import preprocessing,task_num,kfold_train_test_set,separate_train_test_set_mtl,setting_seed,cal_accuracy
# external library
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import warnings

"""
n_input = 119
shared_layer_first = 175
shared_layer_second = 175
tower_input_size = 50
tower_output_size = 2
learning_rate = 1e-5
epoch = 400
n_output : 2
"""
# the multitasklearning class references from https://github.com/hosseinshn/basic-multi-task-learning/blob/master/mtl-pytorch.ipynb
class mtlnet(nn.Module):
    def __init__(self,input_features,shared_layer_first,shared_layer_second,tower_input_size,tower_output_size):
    # as in the pdf, it shows that the neural network contains two hidden layers.
        super(mtlnet, self).__init__()
        self.sharedlayer = nn.Sequential(
          nn.Linear(input_features, shared_layer_first),
          nn.Sigmoid(),
          nn.Linear(shared_layer_first,shared_layer_second),
          nn.Sigmoid(),
        )

        self.tower = []
        for _ in range(task_num):
            self.tower.append(nn.Sequential(
            nn.Linear(shared_layer_second, tower_input_size),
            nn.Sigmoid(),
            nn.Linear(tower_input_size, tower_output_size)))
      
    def forward(self,x):
        x = self.sharedlayer(x)
        tower_size = int(x.shape[0]/task_num)
        outs = []
        for idx,item in enumerate(self.tower):
            outs.append(item(x)[tower_size*(idx):tower_size*(idx+1)])
        return outs
    
def train_mtl(x_training_set,x_valid_set,y_training_set,y_valid_set):
    # setting the seed #
    setting_seed()
    np.random.seed(0)
    ###################
    input_length,input_features = x_training_set.shape[0],x_training_set.shape[1]
    shared_layer_first = 175
    shared_layer_second = 175
    tower_input_size = 50
    tower_output_size = 2
    learning_rate = 1e-5
    epoch = 400
    cost = []
    epochloss = []
    mtl = mtlnet(input_features,shared_layer_first,shared_layer_second,tower_input_size,tower_output_size)
    optimizer = torch.optim.Adam(mtl.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    ml_train_accuracy, ml_val_accuracy = 0,0
    # start iteration
    for it in range(epoch):
        y_hats = mtl(x_training_set)
        # list of losses
        losses = [criterion(y_hats[i], y_training_set[i].long()) for i in range(task_num)]
        # calculate the training losses for each iteration
        loss =  sum(losses)
        epochloss.append(loss)
        
        if it == epoch -1:
            # print the training loss and its accuracy.
            correct = 0
            max_idxs = []; y_training_sets = []
            for i in range(task_num):
                max_idx = torch.argmax(y_hats[i], dim=1)
                correct += (max_idx == y_training_set[i]).sum().float()  
                max_idxs.append(max_idx.detach().numpy())
                y_training_sets.append(y_training_set[i].detach().numpy())
            max_idxs = torch.tensor(np.concatenate(max_idxs))
            y_training_sets = torch.tensor(np.concatenate(y_training_sets))
            ml_train_accuracy = correct/x_training_set.shape[0]
            train_precision,train_recall,train_f1_score = precision_recall_fscore_support(max_idxs, y_training_sets, average='binary')[:-1]
            train_metrics = np.array([ml_train_accuracy,train_precision,train_recall,train_f1_score])

        # backprop
        for i in range(task_num):
            optimizer.zero_grad()
            losses[i].backward(retain_graph=True)
        # for shared layer
        optimizer.zero_grad()
        loss.backward()
        
        for i in range(task_num):
            torch.optim.Adam(mtl.tower[i].parameters(),lr=learning_rate).step()

        optimizer.step()
        with torch.no_grad(): 
            if it == epoch -1:
                # when no gradient calculates the test hats
                y_valid_hats = mtl(x_valid_set)
                valid_losses = [criterion(y_valid_hats[i], y_valid_set[i].long()) for i in range(task_num)]
                correct = 0
                max_idxs = []; y_valid_sets = []
                for i in range(task_num):
                    max_idx = torch.argmax(y_valid_hats[i], dim=1)
                    correct += (max_idx == y_valid_set[i]).sum().float().item()
                    max_idxs.append(max_idx.detach().numpy())
                    y_valid_sets.append(y_valid_set[i].detach().numpy())
                max_idxs = torch.tensor(np.concatenate(max_idxs))
                y_valid_sets = torch.tensor(np.concatenate(y_valid_sets))
                valid_precision,valid_recall,valid_f1_score = precision_recall_fscore_support(max_idxs, y_valid_sets, average='binary')[:-1]
                ml_val_accuracy = correct/x_valid_set.shape[0]
                valid_metrics = np.array([ml_val_accuracy,valid_precision,valid_recall,valid_f1_score])
    # get the final accuracy
    return train_metrics, valid_metrics

data = preprocessing(True)   
mtl_x_training_set,mtl_x_valid_set,mtl_y_training_set,mtl_y_valid_set = separate_train_test_set_mtl(data,0,is_required_grad=True)
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # setting seed to retain reproducibility
    # It takes 4mins to run this code since it requires 12 fold Cross-validation
    kfold = False  # the initial setting is false
    if kfold:
        k = 12
        X_train, X_valid,y_train,y_valid = kfold_train_test_set(data,True,k)
        train_metrics,valid_metrics = np.zeros(4,), np.zeros(4,)
        for ix in range(k):
            train, valid = train_mtl(X_train[ix],X_valid[ix],y_train[ix],y_valid[ix])
            train_metrics += train
            valid_metrics += valid
        train_metrics = train_metrics/k
        valid_metrics = valid_metrics/k
        print(f"Train -- acc: {train_metrics[0]}, precision: {train_metrics[1]}, recall: {train_metrics[2]}, f1_score: {train_metrics[3]}")
        print(f"Valid -- acc: {valid_metrics[0]}, precision: {valid_metrics[1]}, recall: {valid_metrics[2]}, f1_score: {valid_metrics[3]}")
    _,test_metrics = train_mtl(mtl_x_training_set,mtl_x_valid_set,mtl_y_training_set,mtl_y_valid_set)
    print(f"Test -- acc: {test_metrics[0]}, precision: {test_metrics[1]}, recall: {test_metrics[2]},  f1_score: {test_metrics[3]}")
    
    