from preprocessing_dataset import preprocessing,setting_seed,separate_train_test_set_nn,kfold_train_test_set,cal_accuracy
# external libraray
import torch
import torch.nn as nn
import numpy as np 
import warnings
"""
This python file is about three layer neural network with K-fold CV to test whether the neural network is overfitting or not.
And other file uses this settings too.
"""

"""
n_input : 32
n_hidden : 512
n_output : 2
"""

class LSTMRNN(nn.Module):
    def __init__(self,input_size,hidden_size, output_size,num_layers=1):
        super(LSTMRNN,self).__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers)
        self.linear = nn.Linear(hidden_size,output_size)

    def forward(self,input):
        result, _ = self.lstm(input)
        result = self.linear(result)
        # reduce the dimensionality
        return result.squeeze(0)


def train_lstm(x_training_set,x_test_set,y_training_set,y_test_set):
    """
    input is x_training_set,x_test_set,y_training_set,y_test_set
    output is the train accuracy and test accuracy
    """
    # initialize the lstm deep neural network
    input_neurons = x_training_set.shape[1]
    # setting the seed #
    setting_seed()
    np.random.seed(0)
    ###################
    hidden_neurons = 64
    output_neurons = 2
    learning_rate = 1e-4
    epoch = 350
    lstm = LSTMRNN(input_neurons,hidden_neurons, output_neurons)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    x_training_set = x_training_set.reshape(-1, x_training_set.shape[0], input_neurons)
    x_test_set = x_test_set.reshape(-1, x_test_set.shape[0], input_neurons)
    for it in range(epoch):
        y_training_hat = lstm(x_training_set)
        loss = criterion(y_training_hat, y_training_set.long())
        train_acc,train_precision,train_recall, train_f1_score= cal_accuracy(y_training_hat,y_training_set,True)
        train_metrics = np.array([train_acc, train_precision, train_recall,train_f1_score])
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad(): 
            y_test_hat = lstm(x_test_set)
            test_loss = criterion(y_test_hat, y_test_set.long())
            valid_acc, valid_precision, valid_recall,valid_f1_score  = cal_accuracy(y_test_hat,y_test_set,True)
            valid_metrics = np.array([valid_acc, valid_precision, valid_recall,valid_f1_score])
    return train_metrics, valid_metrics
        
data = preprocessing(False)
lstm_x_training_set,lstm_x_test_set,lstm_y_training_set,lstm_y_test_set = separate_train_test_set_nn(data,0,is_require_grad= True)  
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    kfold = False # the initial setting is false
    k = 12
    if kfold:
        X_train, X_valid,y_train,y_valid = kfold_train_test_set(data,False,k)
        train_metrics,valid_metrics = np.zeros(4,), np.zeros(4,)
        for ix in range(k):
            train, valid = train_lstm(X_train[ix],X_valid[ix],y_train[ix],y_valid[ix])
            train_metrics += train
            valid_metrics += valid
            
        train_metrics = train_metrics/k
        valid_metrics = valid_metrics/k
        print(f"Train -- acc: {train_metrics[0]}, precision: {train_metrics[1]}, recall: {train_metrics[2]}, f1_score: {train_metrics[3]}")
        print(f"Valid -- acc: {valid_metrics[0]}, precision: {valid_metrics[1]}, recall: {valid_metrics[2]}, f1_score: {valid_metrics[3]}")
    _,test_metrics = train_lstm(lstm_x_training_set,lstm_x_test_set,lstm_y_training_set,lstm_y_test_set)
    print(f"Test -- acc: {test_metrics[0]}, precision: {test_metrics[1]}, recall: {test_metrics[2]}, f1_score: {test_metrics[3]}")