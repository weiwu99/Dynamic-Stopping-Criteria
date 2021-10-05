# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 23:22:55 2021

@author: ww
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.calibration import CalibratedClassifierCV

flash_per_seq = 12
seq_per_trial = 10
train_trial_num = 20

#%% Get the data matrix
def process_data(data):
    
    X = data['X'][0][0]
    y = data['y'][0][0]
    # print(X.shape)

    non_tar_index = np.where(y == 1)
    tar_index = np.where(y == 2)

    bin_avg_2 = []
    bin_avg_1 = []

    stim_non_index = []
    stim_index = []

    for i in non_tar_index[0]:  
        # a non-target hit
        if (y[i-1] != 1):  
            non_tar = X[i:i+205, :]
            avg_non_tar = np.zeros((15,8))

            for j in range(15):
                avg_non_tar[j,:] = non_tar[13*j:13*(j+1),:].mean(0)

            new_non_tar = avg_non_tar.flatten('F')

            bin_avg_1.append(new_non_tar)
            stim_non_index.append(i)

    for i in tar_index[0]:   
        # a target hit
        if y[i-1] != 2:    
            tar = X[i:i+205, :]
            avg_tar = np.zeros((15,8))
            
            for j in range(15):
                avg_tar[j,:] = tar[13*j:13*(j+1),:].mean(0)
            
            new_tar = avg_tar.flatten('F')

            bin_avg_2.append(new_tar)
            stim_index.append(i)


    bin_avg_1 = np.array(bin_avg_1)
    bin_avg_2 = np.array(bin_avg_2)
    # print(f'The shape of the bin average for non-target is{bin_avg_1.shape}')
    # print(f'The shape of the bin average for target is{bin_avg_2.shape}') 

    processed_X = np.concatenate((bin_avg_1, bin_avg_2), axis=0)
    processed_y = np.concatenate((stim_non_index, stim_index), axis=0)
    processed_data = np.c_[processed_X, processed_y] 

    ### Sort data
    data_sorted = processed_data[np.argsort(processed_data[:, -1])]

    X_sorted = data_sorted[:,0:120]
    # print(data_sorted[:,-1])

    y_sorted = y[np.int_(data_sorted[:,-1])].flatten()
    y_sorted = y_sorted - 1
    data_sorted = np.c_[X_sorted, y_sorted] 
    return data_sorted, X_sorted, y_sorted

def find_char_index(coord_1, coord_2):
    if (coord_1 < coord_2): # if column first, row next, normal case
        real_c = coord_1 - 1
        real_r = coord_2 - 6 - 1
    elif (coord_1 > coord_2): # if row first, column next
        real_c = coord_2 - 1
        real_r = coord_1 - 6 - 1
    
    return np.int64(real_r), np.int64(real_c)

def process_stim(y_stim):

    flash_period = np.where(y_stim != 0)[0] #find stim intervals
    flash_index = []

    for i in flash_period:
        if (y_stim[i-1] == 0):
            flash_index.append(i)

    flash_index = np.array(flash_index) #time stamps when flashes
    flashes = y_stim[flash_index].flatten() #actual y_stim value for each flash

    flash_mat = np.reshape(flashes, (350,12)) #reshape flashes into 350 sequences
    return flash_mat

def char_list(new_flashes, y_data):  # Identify which character is being flashed in the sequence
    
    selected_r = np.zeros(y_data.shape[0])
    selected_c = np.zeros(y_data.shape[0])

    # new_flashes = process_stim(y_stim)
    masked_flash = np.multiply(new_flashes, y_data) #Find the 2 target flashes in each sequence
    
    hit_coord = masked_flash[masked_flash!=0] #Remove zeros from the og matrix
    hit_coord = np.reshape(hit_coord, (y_data.shape[0],2))
    
    # Find the intended char from the original board with the given coordinate pair
    vec_char = np.vectorize(find_char_index)
    selected_r, selected_c = vec_char(hit_coord[:,0], hit_coord[:,1])
    selected_char = create_board()[selected_r, selected_c]
    return selected_char, selected_r, selected_c

def update_char_proba(char_proba, y_score_trial, flash_trial, seq_per_trial, flash_per_seq):
    for seq_index in range(seq_per_trial): # 10 sequences in a trial ## static stop criterion
        y_score_seq = y_score_trial[seq_index]
        flash_seq = flash_trial[seq_index]
        
        loc_char_proba = np.ones((6,6))
        
        # Proba score for each char = P_row * P_col
        for index in range(flash_per_seq): # 12 flashes
            if ((flash_seq[index] > 6) and (flash_seq[index] <= 12)):
                row_idx = flash_seq[index] - 1 - 6
                    
                loc_char_proba[row_idx, :] *= y_score_seq[index]
                    
            elif ((flash_seq[index] <= 6) and (flash_seq[index] >= 0)):
                col_idx = flash_seq[index] - 1
                
                loc_char_proba[:,col_idx] *= y_score_seq[index]
                
        char_proba += loc_char_proba
        
    ### Calculate mean probability for each character to select the char w/ largest proba for this trial
    char_proba = char_proba/seq_per_trial

def plotROC(new_y_train_total, y_score_train, new_y_test, y_score_total, clf_type):
    plt.figure(1, figsize=(8,8))
    plt.clf()
    
    p_FA_train, p_D_train, _  = roc_curve(new_y_train_total, y_score_train)
    auc_train = roc_auc_score(new_y_train_total, y_score_train)
    p_FA_test, p_D_test, _  = roc_curve(new_y_test, y_score_total)
    auc_test = roc_auc_score(new_y_test, y_score_total)
    
    plt.plot(p_FA_train, p_D_train, label = f'ROC for the training  set with AUC = {auc_train}')
    plt.plot(p_FA_test, p_D_test, label = f'ROC for the testing set with AUC = {auc_test}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Chance')
    plt.xlabel('Probability of False Alarm\n($P_{FA}$)', fontsize=12)
    plt.ylabel('Probability of Detection\n($P_D$)', fontsize=12)
    plt.title(f'ROC curves for A08 with {clf_type}')
    plt.grid()
    plt.legend() 

def apply_classifier(clf_type, cc):

    if (clf_type == 'LinearSVC'):
        svm = LinearSVC(C = cc, class_weight = 'balanced', dual = False)  
        clf = CalibratedClassifierCV(svm) 
    elif (clf_type == 'LogReg'):
        clf = LogisticRegression(C = cc, class_weight = 'balanced', dual = False, max_iter = 500)
    elif (clf_type == 'RBF'):
        clf = SVC(C = cc, probability=True, class_weight = 'balanced')
    elif (clf_type == 'LDA'):
        clf = LinearDiscriminantAnalysis()
    # elif (clf_type == 'SWLDA'):
    #     clf = Clfr(penter=0.1,premove=0.15,max_iter=60)
    
    return clf

def create_board():
    board = np.array([
        ['A', 'B', 'C', 'D', 'E', 'F'],
        ['G', 'H', 'I', 'J', 'K', 'L'],
        ['M', 'N', 'O', 'P', 'Q', 'R'],
        ['S', 'T', 'U', 'V', 'W', 'X'],
        ['Y', 'Z', '1', '2', '3', '4'],
        ['5', '6', '7', '8', '9', '_']
        ])

    return board

def constant_for_split(shape, flash_per_seq, seq_per_trial, train_trial_num):
    seq_num = np.int64(shape/flash_per_seq)
    trial_num = np.int64(seq_num/seq_per_trial)
    test_trial_num = trial_num - train_trial_num

    return seq_num, trial_num, test_trial_num


# def test_correct_arr(test_trial_num, y_score_mat_total, flash_test, char_label):
#     char_correct_test = []
#     for i in range(test_trial_num):
        
#         y_score_trial = y_score_mat_total[i]
#         flash_test_trial = flash_test[i]
        
#         mean_char_proba = np.zeros((6,6))
        
#         update_char_proba(mean_char_proba, y_score_trial, flash_test_trial, seq_per_trial, flash_per_seq)
        
#         char_hat = create_board()[np.where(mean_char_proba == np.max(mean_char_proba))]
        
#         if (char_hat == char_label[i+train_trial_num]): 
#             char_correct_test.append(char_hat) #Identify how many chars get correct

#     char_correct_test = np.array(char_correct_test).flatten()

#     return char_correct_test