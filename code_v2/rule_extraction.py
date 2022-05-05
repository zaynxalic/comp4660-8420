from preprocessing_dataset import task_num,setting_seed
from generalneuralnetwork import ThreeLayerNet,gnn_x_train_set,gnn_x_test_set,gnn_y_training_set,gnn_y_test_set
from multitask_learning import mtlnet,mtl_x_training_set,mtl_x_valid_set,mtl_y_training_set,mtl_y_valid_set
from lstm_dl import LSTMRNN,lstm_x_training_set,lstm_x_test_set,lstm_y_training_set,lstm_y_test_set
from lowess import lowess
from genetic_algorithm import *

# external libraray
import torch
import torch.nn as nn
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
# suppress the warning
import warnings
from scipy.optimize.optimize import OptimizeWarning

"""
In this file, the accuracy of rule extraction of ANN, LSTM and MTL is calculated.
METHODS LIST consists of "ANN", "ANN_GA", "LSTM", "LSTM_GA", "MTL", "MTL_GA", "ANN_DT", "LSTM_DT", "MTL_DT"
"""

METHODS_LIST = ["ANN", "ANN_GA", "LSTM", "LSTM_GA", "MTL", "MTL_GA", "ANN_DT", "LSTM_DT", "MTL_DT"]

"""
softmax the input
"""
def softmax(input):
    soft = nn.Softmax(dim=0)
    return soft(input)

"""
if the output is 0, then it is classified as 0
else if the output is 1, then it is classified as 1.
"""
def clustering_index(x_training_set,y_hat):
    zero_class = []
    one_class = [] 
    for i in range(x_training_set.shape[0]):
        if y_hat[i] == 0:
            zero_class.append(i)
        elif y_hat[i] == 1:
            one_class.append(i)
    return zero_class,one_class


def nnwithgradients(x_training_set,x_test_set,y_training_set,y_test_set):
    """
    x_train_set, y_training_set is in the training set
    x_test_set, y_test_set is in the test set
    in this function, the gradient of dc/dA* is calculated for each group
    return:
        dc0,dc1 the gradient of different classes 
        group_0,group_1 gives the index of different classes
        nattr contains of [x_training_set,x_test_set,y_hat,y_val_hat]
    """
    setting_seed()
    input_neurons = x_training_set.shape[1]
    hidden_neurons = 512
    output_neurons = 2
    learning_rate = 0.0001
    epoch = 425
    TLN = ThreeLayerNet(input_neurons,hidden_neurons,output_neurons)
    optimizer = torch.optim.Adam(TLN.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    nttr = 0
    for it in range(epoch):
        y_hat = TLN(x_training_set)
        loss = criterion(y_hat, y_training_set.long())
        optimizer.zero_grad()
        # retain the graph in order to compute the gradients
        loss.backward(retain_graph=True)
        if it == epoch -1:
            y_val_hat =  TLN(x_test_set)
            dc0s = []
            dc1s = []
            for i in range(len(y_hat)):
                dc0s.append(torch.autograd.grad(softmax(y_hat[i])[0], x_training_set, retain_graph=True)[0][i])
                dc1s.append(torch.autograd.grad(softmax(y_hat[i])[1], x_training_set, retain_graph=True)[0][i])
            # calculate the gradients dC0/dAi and dC1/dAi
            dc0 = torch.stack(dc0s)
            dc1 = torch.stack(dc1s)
            # calculate the prediction of y_hat and y_val_hat by softmax function
            y_hat = torch.argmax(softmax(y_hat),1)
            y_val_hat = torch.argmax(softmax(y_val_hat),1)
            group_0,group_1 = clustering_index(x_training_set,y_hat)
            nnattr = [x_training_set,x_test_set,y_hat,y_val_hat]
        optimizer.step() 
    return dc0,dc1,group_0,group_1,nnattr


"""
x_train_set, y_training_set is in the training set
x_test_set, y_test_set is in the test set
in this function, the gradient of dc/dA* is calculated for each group
return:
    dc0,dc1 the gradient of different classes 
    group_0,group_1 gives the index of different classes
    nattr contains of [x_training_set,x_test_set,y_hat,y_val_hat]
"""
def lstmwithgradients(x_training_set,x_test_set,y_training_set,y_test_set):
    """
    input is x_training_set,x_test_set,y_training_set,y_test_set
    output is the train accuracy and test accuracy
    """
    input_neurons = x_training_set.shape[1]
    # setting the seed #
    setting_seed()
    np.random.seed(0)
    ###################
    hidden_neurons = 64
    output_neurons = 2
    learning_rate = 1e-4
    epoch = 500
    lstm = LSTMRNN(input_neurons,hidden_neurons, output_neurons)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    x_training_set_reshaped = x_training_set.reshape(-1, x_training_set.shape[0], input_neurons)
    x_test_set_reshaped = x_test_set.reshape(-1, x_test_set.shape[0], input_neurons)
    
    for it in range(epoch):
        y_hat = lstm(x_training_set_reshaped)
        loss = criterion(y_hat, y_training_set.long())
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        
        if it == epoch -1:
            y_val_hat =  lstm(x_test_set_reshaped)
            dc0s = []
            dc1s = []
            for i in range(len(y_hat)):
                dc0s.append(torch.autograd.grad(softmax(y_hat[i])[0], x_training_set, retain_graph=True)[0][i])
                dc1s.append(torch.autograd.grad(softmax(y_hat[i])[1], x_training_set, retain_graph=True)[0][i])
            # calculate the gradients dC0/dAi and dC1/dAi
            dc0 = torch.stack(dc0s)
            dc1 = torch.stack(dc1s)
            # calculate the prediction of y_hat and y_val_hat by softmax function
            y_hat = torch.argmax(softmax(y_hat),1)
            y_val_hat = torch.argmax(softmax(y_val_hat),1)
            group_0,group_1 = clustering_index(x_training_set,y_hat)
            nnattr = [x_training_set,x_test_set,y_hat,y_val_hat] 
        optimizer.step()
    return dc0,dc1,group_0,group_1,nnattr

"""
x_train_set, y_training_set is in the training set
x_test_set, y_test_set is in the test set
in this function, the gradient of dc/dA* is calculated for each group
return:
    dc0,dc1 the gradient of different classes 
    group_0,group_1 gives the index of different classes
    nattr contains of [x_training_set,x_test_set,y_hat,y_val_hat]
"""
def mtlwithgradients(x_training_set,x_valid_set,y_training_set,y_valid_set):
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
    epochvalidloss = []
    mtl = mtlnet(input_features,shared_layer_first,shared_layer_second,tower_input_size,tower_output_size)
    optimizer = torch.optim.Adam(mtl.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # start iteration
    for it in range(epoch):
        y_hats = mtl(x_training_set)
        # list of losses
        losses = [criterion(y_hats[i], y_training_set[i].long()) for i in range(task_num)]
        # calculate the training losses for each iteration
        loss =  sum(losses)/task_num
        epochloss.append(loss)
        
        # backprop
        for i in range(task_num):
            optimizer.zero_grad()
            losses[i].backward(retain_graph=True)
        # for shared layer
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        # early stoping, if the validation pattern keeps decrease, then early stops
        if it == epoch-1:
            y_val_hat =  mtl(x_valid_set)
            dc0s = []
            dc1s = []
            y_val_hat = torch.cat(y_val_hat)
            y_hats = torch.cat(y_hats)
            for i in range(len(y_hats)):
                dc0s.append(torch.autograd.grad(softmax(y_hats[i])[0], x_training_set, retain_graph=True)[0][i])
                dc1s.append(torch.autograd.grad(softmax(y_hats[i])[1], x_training_set, retain_graph=True)[0][i])
            # calculate the gradients dC0/dAi and dC1/dAi
            dc0 = torch.stack(dc0s)
            dc1 = torch.stack(dc1s)
            # calculate the prediction of y_hat and y_val_hat by softmax function
            y_hats = torch.argmax(softmax(y_hats),1)
            y_val_hat = torch.argmax(softmax(y_val_hat),1)
            group_0,group_1 = clustering_index(x_training_set,y_hats)
            nnattr = [x_training_set,x_valid_set,y_hats,y_val_hat] 
            break 
        
        for i in range(task_num):
            torch.optim.Adam(mtl.tower[i].parameters(),lr=learning_rate).step()
            
        optimizer.step()
    # get the final accuracy
    return dc0,dc1,group_0,group_1,nnattr

def rate(len1,len2):
    return len1/(len1+len2)


def find_max_local_regression(xs,ys):
    """
    using local regression function to approximate f(xs,ys) function
    """
    yest = lowess(xs, ys, f=0.2, it=3)
    return xs[np.argmax(yest)]
   

def train_explanation(nnattr, g_0, g_1, group_0, group_1):
    """
    nnattr consists of generated neural network: x_train_set, x_test_set, y_hat, y_test_hat
    g_0 consists of gradient with output = 0
    g_1 consists of gradient with output = 1
    group_0 consists of input pattern with output = 0
    group_1 consists of input pattern with output = 1
    """
    ratio = rate(len(group_1),len(group_0))
    threshold = [[] for _ in range(2)]
    groups = [g_0[group_0],g_1[group_1]]
    group_id = [group_0, group_1]
   
    for g_id in range(len(groups)):
        g = groups[g_id]
        for idx,col in enumerate(range(nnattr[0].shape[1])):
            class0_count,class1_count = 0,0
            # calculate the domain of gradient function
            min_row = np.min(nnattr[0][:,idx].detach().numpy())
            max_row = np.max(nnattr[0][:,idx].detach().numpy())
            gr = nnattr[0][group_id[g_id],:]
            boundary = find_max_local_regression(gr[:,idx].detach().numpy(),np.abs(g[:,idx].detach().numpy()))
            for row,x in enumerate(nnattr[0]):
                if x[idx] < boundary:
                    if nnattr[2][row] == 0: class0_count += 1
                    elif nnattr[2][row] == 1: class1_count += 1
            
            prop = len(group_1)/(len(group_0) + len(group_1))
            most_group = 1 if np.random.rand() < prop else 0
            # if class0_count + class1_count == 0 indicating the value is error, 
            # then 
            if class0_count + class1_count == 0: threshold[g_id].append((boundary,most_group))
            # if class1_count occurs more than ratio 
            else: 
                if rate(class1_count,class0_count) > ratio: threshold[g_id].append((boundary, 1))
                else:threshold[g_id].append((boundary, 0))
    return threshold

def eval_explanation(nnattr,g_0, g_1,group_0,group_1,threshold,ind_list,new_input):
    """
    In the evalution stage, give the answer depending on the most occurence number
    """
    label = getlabel(new_input.detach().numpy(),nnattr[0][group_0].detach().numpy(), nnattr[0][group_1].detach().numpy())
    thresh = list(np.array(threshold[label])[ind_list])
    class0_count,class1_count = 0,0  
    for idx in range(new_input.shape[0]):
        if new_input[idx] < thresh[idx][0]:
            if thresh[idx][1] == 0: class0_count += 1
            elif thresh[idx][1] == 1: class1_count += 1
        else:
            if thresh[idx][1] == 1: class0_count += 1
            elif thresh[idx][1] == 0: class1_count += 1
            
    predict = 1 if class1_count > class0_count else 0
    return predict 


def getlabel(new_value, label1, label2):
    """
    new_value is the coming input value, label1, label2 are 
    """
    mean1 = np.mean(label1,axis=0)
    mean2 = np.mean(label2,axis=0)
    label = 1 if np.linalg.norm(mean1 - new_value) > np.linalg.norm(mean2 - new_value) else 0
    return label


def calculate_explanation_ann(nnattr, g_0, g_1, group_0, group_1,ind_list,threshold):
    """
    calculate the accuracy given predict and actual
    """
    row = nnattr[1].shape[0]
    col = ind_list.shape[0]
    count = 0
    for i in range(row):
        predict = eval_explanation(nnattr,g_0, g_1,group_0,group_1,threshold,ind_list,nnattr[1][:,ind_list][i])
        if predict == nnattr[-1][i]:
            count += 1
    return count/row


def Fvalue(pop,g_0,g_1,group_0,group_1,nnattr,inputs,threshold):
    """
    calculate the value for each population
    """
    values = []
    for idx in range(POP_SIZE):
        ind_list = np.where(pop[idx] == 1)[0]
        nnattr[0] = inputs[:,ind_list]
        values.append(calculate_explanation_ann(nnattr, g_0, g_1, group_0, group_1,ind_list,threshold) * 100)
    return values
    
def GA(DNA_SIZE,g_0,g_1,group_0,group_1,nnattr,inputs,threshold):
    pop = np.random.randint(0,2, size=(POP_SIZE, DNA_SIZE)) 
    for _ in range(N_GENERATIONS):
        fitness = Fvalue(pop,g_0,g_1,group_0,group_1,nnattr,inputs,threshold)
        pop = select(pop, fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = crossover(parent, pop_copy,DNA_SIZE)
            child = mutate(child,DNA_SIZE)
            parent[:] = child 
    return  np.max(fitness)/100

def decision_tree(x_training_set,x_valid_set,y_hats,y_valid_hat):
    """
    Calculate the Decision tree classifier 
    the input is trained neural network
    """
    DTC = DecisionTreeClassifier(max_depth=3,random_state = 0)
    x_training_set = x_training_set.detach().numpy()
    y_valid_hat = y_valid_hat.detach().numpy()
    x_valid_set = x_valid_set.detach().numpy()
    DTC = DTC.fit(x_training_set,y_hats)
    testlength = x_valid_set.shape[0]
    # valid the train set
    count = 0
    for idx in range(testlength):
        if DTC.predict([x_valid_set[idx]]) == y_valid_hat[idx]:
            count += 1
    return count/testlength

def main(METHOD):
    if METHOD in METHODS_LIST:
        if "ANN" in METHOD:
            g_0, g_1,group_0,group_1,nnattr = nnwithgradients(gnn_x_train_set,gnn_x_test_set,gnn_y_training_set,gnn_y_test_set)
            threshold = train_explanation(nnattr, g_0, g_1, group_0, group_1)
            x_training_set,x_valid_set,y_hats,y_valid_hat = nnattr[0],nnattr[1],nnattr[2],nnattr[3]
            if "GA" in METHOD:
                DNA_SIZE = gnn_x_train_set.shape[1]
                acc = GA(DNA_SIZE,g_0, g_1,group_0,group_1,nnattr,gnn_x_train_set,threshold)
            elif "DT" in METHOD:
                acc = decision_tree(x_training_set,x_valid_set,y_hats,y_valid_hat)
            else:
                ind_list = np.array([i for i in range(nnattr[0].shape[1])])
                acc = calculate_explanation_ann(nnattr, g_0, g_1, group_0, group_1,ind_list,threshold)  
        elif "LSTM" in METHOD:
            g_0, g_1,group_0,group_1,nnattr = lstmwithgradients(lstm_x_training_set,lstm_x_test_set,lstm_y_training_set,lstm_y_test_set)
            threshold = train_explanation(nnattr, g_0, g_1, group_0, group_1)
            x_training_set,x_valid_set,y_hats,y_valid_hat = nnattr[0],nnattr[1],nnattr[2],nnattr[3]
            if "GA" in METHOD:
                DNA_SIZE = lstm_x_training_set.shape[1]
                acc = GA(DNA_SIZE,g_0, g_1,group_0,group_1,nnattr,lstm_x_training_set,threshold)
            elif "DT" in METHOD:
                acc = decision_tree(x_training_set,x_valid_set,y_hats,y_valid_hat)
            else:
                ind_list = np.array([i for i in range(nnattr[0].shape[1])])
                acc = calculate_explanation_ann(nnattr, g_0, g_1, group_0, group_1,ind_list,threshold)
        elif "MTL" in METHOD:
            g_0, g_1,group_0,group_1,nnattr = mtlwithgradients(mtl_x_training_set,mtl_x_valid_set,mtl_y_training_set,mtl_y_valid_set)
            threshold = train_explanation(nnattr, g_0, g_1, group_0, group_1)
            x_training_set,x_valid_set,y_hats,y_valid_hat = nnattr[0],nnattr[1],nnattr[2],nnattr[3]
            if "GA" in METHOD:
                DNA_SIZE = mtl_x_training_set.shape[1]
                acc = GA(DNA_SIZE,g_0, g_1,group_0,group_1,nnattr,mtl_x_training_set,threshold)
            elif "DT" in METHOD:
                acc = decision_tree(x_training_set,x_valid_set,y_hats,y_valid_hat)
            else:
                ind_list = np.array([i for i in range(nnattr[0].shape[1])])
                acc = calculate_explanation_ann(nnattr, g_0, g_1, group_0, group_1,ind_list,threshold)
        print(f"{METHOD} of explanation accuracy is {acc}")
        
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=OptimizeWarning) 
    main("ANN")
    main("ANN_GA")
    main("ANN_DT")
    main("LSTM")
    main("LSTM_GA")
    main("LSTM_DT")
    main("MTL")
    main("MTL_GA")
    main("MTL_DT")
