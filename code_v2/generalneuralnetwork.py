from preprocessing_dataset import preprocessing,setting_seed,separate_train_test_set_nn,kfold_train_test_set,cal_accuracy
import torch
import torch.nn as nn
import numpy as np 


"""
This python file is about three layer neural network with K-fold CV to test whether the neural network is overfitting or not.
And other file uses this settings too.
"""

"""
n_input : 32
n_hidden : 512
n_output : 2
"""
class ThreeLayerNet(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(ThreeLayerNet, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self,x):
        x = self.hidden(x)
        x = torch.sigmoid(x)
        x = self.out(x)
        return x

def train_gnn(gnn_x_train_set,gnn_x_test_set,gnn_y_training_set,gnn_y_test_set):
    """
    input is gnn_x_train_set,gnn_x_test_set,gnn_y_training_set,gnn_y_test_set
    output is the train accuracy and test accuracy
    """
    # train the two layer neural network
    # setting the seed #
    setting_seed()
    np.random.seed(0)
    ###################
    input_neurons = gnn_x_train_set.shape[1]
    hidden_neurons = 512
    output_neurons = 2
    learning_rate = 0.0001
    epoch = 300
    TLN = ThreeLayerNet(input_neurons,hidden_neurons,output_neurons)
    optimizer = torch.optim.Adam(TLN.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for it in range(epoch):
        y_hat = TLN(gnn_x_train_set)
        loss = criterion(y_hat, gnn_y_training_set.long())
        train_acc,train_precision,train_recall,train_f1_score = cal_accuracy(y_hat,gnn_y_training_set,show_precision_recall=True)
        # calculate the recall_precision_f1_score
        train_metrics = np.array([train_acc, train_precision, train_recall,train_f1_score])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        with torch.no_grad(): 
            y_test_hat = TLN(gnn_x_test_set)
            test_loss = criterion(y_test_hat, gnn_y_test_set.long())
            valid_acc, valid_precision, valid_recall,valid_f1_score = cal_accuracy(y_test_hat,gnn_y_test_set,show_precision_recall=True)
            valid_metrics = np.array([valid_acc, valid_precision, valid_recall,valid_f1_score])
    return train_metrics, valid_metrics

pd_data = preprocessing(True)
gnn_x_train_set,gnn_x_test_set,gnn_y_training_set,gnn_y_test_set = separate_train_test_set_nn(pd_data,0,is_require_grad= True)
if __name__ == "__main__":
    kfold = False # the initial setting is false
    if kfold:
        k = 12
        X_train, X_valid,y_train,y_valid = kfold_train_test_set(pd_data,False,k)
        train_metrics,valid_metrics = np.zeros(4,), np.zeros(4,)
        for ix in range(k):
            train, valid = train_gnn(X_train[ix],X_valid[ix],y_train[ix],y_valid[ix])
            train_metrics += train
            valid_metrics += valid
            
        train_metrics = train_metrics/k
        valid_metrics = valid_metrics/k
        print(f"Train -- acc: {train_metrics[0]}, precision: {train_metrics[1]}, recall: {train_metrics[2]}, f1_score: {train_metrics[3]}")
        print(f"Valid -- acc: {valid_metrics[0]}, precision: {valid_metrics[1]}, recall: {valid_metrics[2]}, f1_score: {valid_metrics[3]}")
    _,test_metrics = train_gnn(gnn_x_train_set,gnn_x_test_set,gnn_y_training_set,gnn_y_test_set)
    print(f"Test -- acc: {test_metrics[0]}, precision: {test_metrics[1]}, recall: {test_metrics[2]}, f1_score: {test_metrics[3]}")
  