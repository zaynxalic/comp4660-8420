# external library
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
import numpy as np 
from sklearn.metrics import precision_recall_fscore_support
from scipy import stats
from scipy.signal import butter,lfilter

"""
Data Preprocessing:
consisting of 
1. seed settings
2. normalizing 
3. change the sequence data
4. separate train and test data set
5. KFOLD
6. calculate the accuracy and other statistical metrics like precision, recall, f1score of train-test set
"""

def k_list(k):
    return [[] for _ in range(k)]

def kfold_train_test_set(data,isflag,k,is_shuffle = False):
    """
    Kfold train_test returns k lists of X_train, X_valid,y_train,y_valid
    """
    # build a 2d of empty list. The index contains number of kfold.
    X_train, X_valid,y_train,y_valid = k_list(k),k_list(k),k_list(k),k_list(k)
    kf = KFold(n_splits=k,shuffle=is_shuffle)
    if isflag:
        tensor_X,X_test,tensor_Y,y_test = separate_train_test_set_mtl(data)
        for i in range(task_num):
            X = tensor_X[len(tensor_X)//task_num*i:len(tensor_X)//task_num* (i+1)]
            Y = tensor_Y[i]
            for idx,(train_idx, test_idx)  in enumerate(kf.split(X)):
                X_train[idx].append(X[train_idx])
                X_valid[idx].append(X[test_idx])
                # i, which is the ith task
                y_train[idx].append((i,Y[train_idx]))
                y_valid[idx].append((i,Y[test_idx]))
        
        # stack the list of item in each kfold
        for idx in range(k):
            X_train[idx],X_valid[idx] = torch.tensor(np.concatenate(X_train[idx])).float(),torch.tensor(np.concatenate(X_valid[idx])).float()
            for i in range(task_num):
                y_train[idx][i] = torch.FloatTensor(y_train[idx][i][1])
                y_valid[idx][i] = torch.FloatTensor(y_valid[idx][i][1])
    else:
        tensor_X,X_test,tensor_Y,y_test = separate_train_test_set_nn(data)
        for idx, (train_idx,test_idx) in enumerate(kf.split(tensor_X)):
            X_train[idx] = tensor_X[train_idx]
            X_valid[idx] = tensor_X[test_idx]
            y_train[idx] = tensor_Y[train_idx]
            y_valid[idx] = tensor_Y[test_idx]
    return  X_train, X_valid,y_train,y_valid


# adding reproducibility
def setting_seed():
    """
    retain the reproducibility
    """
    torch.manual_seed(1)

def normalize(data):
    return (data - np.min(data))/(np.max(data) - np.min(data))

def cal_accuracy(predicted,actual, show_precision_recall=False):
    """
    cal_accuracy return accuracy, and show precision, recall
    """
    if show_precision_recall:
        max_idx = torch.argmax(predicted, dim=1, keepdim=False)
        precision,recall,f1_score = precision_recall_fscore_support(actual.detach().numpy(),max_idx.detach().numpy(), average='binary')[:-1]
        return (max_idx == actual).sum().float().item()/len(actual), precision, recall,f1_score
    else:
        max_idx = torch.argmax(predicted, dim=1, keepdim=False)
        return (max_idx == actual).sum().float().item()/len(actual)

def convertPandaToTensor(df,is_require_grad = False):
    return torch.tensor(df.values.astype(np.float32),requires_grad=is_require_grad)

def separate_train_test_set_nn(data,seed=3,is_require_grad = False):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1].drop(columns=['pid','vid']), data.iloc[:,-1], test_size=0.2, random_state=seed)
    return convertPandaToTensor(X_train,is_require_grad),convertPandaToTensor(X_test,is_require_grad),convertPandaToTensor(y_train,is_require_grad),convertPandaToTensor(y_test,is_require_grad)

def separate_train_test_set_mtl(data,seed=1,is_required_grad=False):
    """
    generate the result of x_train,x_test,y_train,y_test as type of tensor
    """
    x_training_set, x_test_set,y_training_set,y_test_set= [],[],[],[]
    for id in data.iloc[:,0].unique():
        id_participants = data[data.iloc[:,0]==id]
        X_train, X_test, y_train, y_test = train_test_split(id_participants.iloc[:,:-1].drop(columns=['pid','vid']), id_participants.iloc[:,-1], test_size=0.2, random_state=seed)
        x_training_set.append(X_train)
        x_test_set.append(X_test)
        y_training_set.append(convertPandaToTensor(y_train,is_required_grad))
        y_test_set.append(convertPandaToTensor(y_test,is_required_grad))
    return convertPandaToTensor(pd.concat(x_training_set),is_required_grad),convertPandaToTensor(pd.concat(x_test_set),is_required_grad),y_training_set,y_test_set

# calcualte the mean difference of time sequence data
def mean_difference(ele, order):
    ele = np.array(ele)
    length = ele.shape[1]
    diff = ele[:,:length-order] -ele[:,-(length-order):]
    return np.mean(np.abs(diff),axis=1)

def butter_lowpass_filter(data, freq, order=6):
    """
    butter_lowpass_filter with order of 6.
    """
    b,a = butter(order,freq)
    y =  lfilter(b, a, data)
    return y

data = pd.read_csv('subjective_belief_observers_features_labels.csv')
# split the p_id and v_id and perform the multi-task learning
pid_vid = data.iloc[:,0]
pid = pid_vid.str.split('_', expand=True)[0]
vid = pid_vid.str.split('_', expand=True)[1]
data.insert(0,"pid",pid)
data.insert(1,"vid",vid)
data = data.drop(columns='pid_vid')

#normalize the data
data.iloc[:,2:-1] = (data.iloc[:,2:-1]-data.iloc[:,2:-1].min())/(data.iloc[:,2:-1].max()- data.iloc[:2:-1].min())
#caculate the tasknum
task_num = len(data.iloc[:,0].unique())

def preprocessing(flag):
    """
    if requires preprocessing, then do the preprocessing and change the feature dimension to 32
    """
    data_remove_pid = data.iloc[:,2:-1]
    bvp = data_remove_pid.filter(regex='bvp').values.astype(float)
    gsr = data_remove_pid.filter(regex='gsr').values.astype(float)
    temp = data_remove_pid.filter(regex='temp').values.astype(float)
    eye = data_remove_pid.filter(regex='eye').values.astype(float)
    gsr = butter_lowpass_filter(gsr,0.2)
    temp = butter_lowpass_filter(temp,0.3)
    output = data.iloc[:,-1].values
    """
        if flag = True, which needs to perform following step 
        if false, then give the raw data
    """
    if flag:
        # according to the essay 
        # Deceit Detection: Identification of Presenter’s Subjective Doubt Using Affective Observation Neural Network Analysis
        # which separates the dimension into 32 dimentions, each 4 features consisting of minimum, maximum, mean, std, var, rsm and diff1, diff2
        pd_data = {
            "pid": pid.values,
            "vid" : vid.values,
            # recreate bvp_data
            'bvp_min': np.min(bvp,axis=1),
            'bvp_max': np.max(bvp,axis=1),
            'bvp_mean' : np.mean(bvp,axis=1),
            'bvp_std' : np.std(bvp,axis=1),
            'bvp_var' : np.var(bvp,axis=1),
            'bvp_rms' : np.sqrt(np.mean(bvp**2,axis=1)),
            'bvp_diff1' :  mean_difference(bvp,1),
            'bvp_diff2' :  mean_difference(bvp,2),
            
            # recreate gsr_data
            'gsr_min': np.min(gsr,axis=1),
            'gsr_max': np.max(gsr,axis=1),
            'gsr_mean' : np.mean(gsr,axis=1),
            'gsr_std' : np.std(gsr,axis=1),
            'gsr_var' : np.var(gsr,axis=1),
            'gsr_rms' : np.sqrt(np.mean(gsr**2,axis=1)),
            'gsr_diff1' :  mean_difference(gsr,1),
            'gsr_diff2' :  mean_difference(gsr,2),
            
            # recreate temp_data
            'temp_min': np.min(temp,axis=1),
            'temp_max': np.max(temp,axis=1),
            'temp_mean' : np.mean(temp,axis=1),
            'temp_std' : np.std(temp,axis=1),
            'temp_var' : np.var(temp,axis=1),
            'temp_rms' : np.sqrt(np.mean(temp**2,axis=1)),
            'temp_diff1' :  mean_difference(temp,1),
            'temp_diff2' :  mean_difference(temp,2),
            
            # recreate eye_data
            'eye_min': np.min(eye,axis=1),
            'eye_max': np.max(eye,axis=1),
            'eye_mean' : np.mean(eye,axis=1),
            'eye_std' : np.std(eye,axis=1),
            'eye_var' : np.var(eye,axis=1),
            'eye_rms' : np.sqrt(np.mean(eye**2,axis=1)),
            'eye_diff1' :  mean_difference(eye,1),
            'eye_diff2' :  mean_difference(eye,2),
            "presenter_subjective_belief" : output
        }
        pd_data = pd.DataFrame(pd_data)
    else:
        pd_bvp = pd.DataFrame(bvp,columns = data_remove_pid.filter(regex='bvp').columns)
        pd_gsr = pd.DataFrame(gsr,columns = data_remove_pid.filter(regex='gsr').columns)
        pd_temp = pd.DataFrame(temp,columns = data_remove_pid.filter(regex='temp').columns)
        pd_eye = pd.DataFrame(eye,columns = data_remove_pid.filter(regex='eye').columns)
        pd_data = pd.concat([data.iloc[:,:2],pd_bvp,pd_gsr,pd_temp,pd_eye,data.iloc[:,-1]],axis=1,sort=False)
    return pd_data 
