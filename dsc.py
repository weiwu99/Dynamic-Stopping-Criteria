import numpy as np
import scipy.io as sio

from timeit import default_timer as timer
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing

from tqdm import tqdm

import sys

import util

def dynamic_stopping_criterion(dataset, clf_type = 'LDA', C_best = 1, train_trial_num = 20, threshold = 0.9):

    board = util.create_board()
        
    data = dataset['data']
    y_stim = data['y_stim'][0][0]

    ### Reorganize the matrix
    _, X_sorted, y_sorted = util.process_data(data)
    flash_mat = util.process_stim(y_stim) #which row/column being flashed

    ### Some constants for data splitting
    flash_per_seq = 12
    seq_per_trial = 10
    seq_num, trial_num, test_trial_num = util.constant_for_split(X_sorted.shape[0], flash_per_seq, seq_per_trial, train_trial_num)

    new_X = np.reshape(X_sorted, (seq_num,flash_per_seq,120))
    new_y = np.reshape(y_sorted, (seq_num,flash_per_seq))

    char_true, _, _ = util.char_list(flash_mat, new_y) # the character flashed for each sequence
    char_true = np.reshape(char_true, (trial_num,seq_per_trial))
    
    char_label = char_true[:,0] # Correct character for each trial
    og_label = np.reshape(char_label, (7,5)) # just for reference, not used in our code
    
    # For 35 trials, each trial has 10 sequences, each sequence has 12 flashes
    indices = np.arange(seq_num)
    test_index = indices[train_trial_num*10:]
    total_train_index = indices[:train_trial_num*10]
    
    # print(f'The best reguarlization parameter C is {C_best}')
    
    ### Testing data
    X_train_total = new_X[total_train_index]
    y_train_total = new_y[total_train_index]
    X_test = new_X[test_index]
    y_test = new_y[test_index]
    
    flash_test = flash_mat[test_index]
    
    # reshape data for clf
    new_X_train_total = np.reshape(X_train_total, (X_train_total.shape[0]*X_train_total.shape[1],120))
    new_y_train_total = np.reshape(y_train_total, (y_train_total.shape[0]*y_train_total.shape[1]))
    new_X_test = np.reshape(X_test, (X_test.shape[0]*X_test.shape[1],120))
    new_y_test = np.reshape(y_test, (y_test.shape[0]*y_test.shape[1]))

    ### https://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/
    
    ### Offline Analysis
    clf = util.apply_classifier(clf_type, cc = C_best)
    clf.fit(new_X_train_total, new_y_train_total)

    y_stats_offline = clf.decision_function(new_X_train_total)
    y_stats_H0 = y_stats_offline[new_y_train_total == 0]
    y_stats_H1 = y_stats_offline[new_y_train_total == 1]

    scaler = preprocessing.StandardScaler().fit(y_stats_offline[:, None])
    kde_0_scale, kde_1_scale = get_likelihood_pdf(y_stats_H0, y_stats_H1, scaler)

    ### Online analysis
    X_test_trial_total = np.reshape(X_test, (X_test.shape[0]//seq_per_trial, seq_per_trial, flash_per_seq,120)) # 4D matrix

    flash_test_mat = np.reshape(flash_test, (np.int64(flash_test.shape[0]/seq_per_trial),seq_per_trial, flash_per_seq))

    prior = 1/36
    char_correct_test = []
    seq_needed = []
    start = timer()

    for i in range(test_trial_num): ### for each trial 15
        
        flash_test_trial = flash_test_mat[i]
        X_test_trial = X_test_trial_total[i]
        
        char_proba_mat = np.zeros((6,6)) + prior # store posterior proba
        # char_proba_mat_H0 = np.zeros((6,6)) + p_H0
        
        seq_index = 0
        char_hat = ""

        ### dynamic stopping criteria no larger than 10 seqs
        while (seq_index < seq_per_trial): 

            X_test_seq = X_test_trial[seq_index]
            flash_seq = flash_test_trial[seq_index]
            
            # Bayesian update
            for index in range(flash_per_seq): # 12 flashes per seq
                y_stats_online = clf.decision_function(X_test_seq[index].reshape(1, -1))
                flash_index = flash_seq[index]
                
                # scaled decision statistics
                y_stats_online_scaled = scaler.transform(y_stats_online[:, None])
                
                # likelihood function
                p_lambda_H0 = np.exp(kde_0_scale.score_samples(y_stats_online_scaled.reshape(1, -1)))
                p_lambda_H1 = np.exp(kde_1_scale.score_samples(y_stats_online_scaled.reshape(1, -1)))

                # update w. Bayes Rule:
                if ((flash_index > 6) and (flash_index <= 12)):
                    row_idx = flash_index - 1 - 6
                    
                    # masking
                    mask = np.ones(char_proba_mat.shape[0], bool)
                    mask[row_idx] = False

                    # update p_lambda_H1 for flashed
                    char_proba_mat[row_idx, :] = char_proba_mat[row_idx, :] * p_lambda_H1/(np.sum(char_proba_mat[row_idx, :] * p_lambda_H1) + np.sum(char_proba_mat[mask, :] * p_lambda_H0))

                    # update p_lambda_H0 for unflashed
                    char_proba_mat[mask, :] = char_proba_mat[mask, :] * p_lambda_H0/(np.sum(char_proba_mat[row_idx, :] * p_lambda_H1) + np.sum(char_proba_mat[mask, :] * p_lambda_H0))
                    
                elif ((flash_index <= 6) and (flash_index >= 0)):
                    col_idx = flash_index - 1
                                
                    # masking
                    mask = np.ones(char_proba_mat.shape[0], bool)
                    mask[col_idx] = False
            
                    # update p_lambda_H1 for flashed
                    char_proba_mat[:, col_idx] = char_proba_mat[:, col_idx] * p_lambda_H1/(np.sum(char_proba_mat[:, col_idx] * p_lambda_H1) + np.sum(char_proba_mat[:, mask] * p_lambda_H0))

                    # update p_lambda_H0 for unflashed                
                    char_proba_mat[:, mask] = char_proba_mat[:, mask] * p_lambda_H0/(np.sum(char_proba_mat[:, col_idx] * p_lambda_H1) + np.sum(char_proba_mat[:, mask] * p_lambda_H0))
    
            if (np.max(char_proba_mat) >= threshold):
                char_hat = board[np.where(char_proba_mat == np.max(char_proba_mat))]
                print(f'{seq_index} out of 10 sequences are used')
                seq_needed.append(seq_index)
                break 
                
            if (seq_index >= seq_per_trial - 1): # no val passes the threshold, choose maximum
                char_hat = board[np.where(char_proba_mat == np.max(char_proba_mat))]
                print(f'{seq_index} out of 10 sequences are used')
                seq_needed.append(seq_index)

            seq_index += 1
            
        # print(char_hat)
        if (char_hat == char_label[i+train_trial_num]): 
            char_correct_test.append(char_hat) #Identify how many chars get correct
            
    char_correct_test = np.array(char_correct_test).flatten()     

    train_accuracy = clf.score(new_X_train_total, new_y_train_total)        
    test_accuracy = clf.score(new_X_test, new_y_test)

    print_results(test_trial_num, char_correct_test, start, train_accuracy, test_accuracy)
    print(f'Mean number of sequences required: {np.mean(np.array(seq_needed))}')

    return clf
def static_stopping_criterion(dataset, clf_type = 'LDA', C_best = 1, train_trial_num = 20):   

    board = util.create_board()
        
    data = dataset['data']
    y_stim = data['y_stim'][0][0]

    ### Reorganize the matrix
    _, X_sorted, y_sorted = util.process_data(data)
    flash_mat = util.process_stim(y_stim) #which row/column being flashed

    ### Some constants for data splitting
    flash_per_seq = 12
    seq_per_trial = 10
    seq_num, trial_num, test_trial_num = util.constant_for_split(X_sorted.shape[0], flash_per_seq, seq_per_trial, train_trial_num)

    new_X = np.reshape(X_sorted, (seq_num,flash_per_seq,120))
    new_y = np.reshape(y_sorted, (seq_num,flash_per_seq))

    char_true, _, _ = util.char_list(flash_mat, new_y) # the character flashed for each sequence
    char_true = np.reshape(char_true, (trial_num,seq_per_trial))
    
    char_label = char_true[:,0] # Correct character for each trial
    og_label = np.reshape(char_label, (7,5)) # just for reference, not used in our code
    
    # For 35 trials, each trial has 10 sequences, each sequence has 12 flashes
    indices = np.arange(seq_num)
    test_index = indices[train_trial_num*10:]
    total_train_index = indices[:train_trial_num*10]
    
    C_best = 0.01 #dummy for lda
    # print(f'The best reguarlization parameter C is {C_best}')
    
    ### Testing data
    X_train_total = new_X[total_train_index]
    y_train_total = new_y[total_train_index]
    X_test = new_X[test_index]
    y_test = new_y[test_index]
    
    flash_test = flash_mat[test_index]
    
    # reshape data for clf
    new_X_train_total = np.reshape(X_train_total, (X_train_total.shape[0]*X_train_total.shape[1],120))
    new_y_train_total = np.reshape(y_train_total, (y_train_total.shape[0]*y_train_total.shape[1]))
    new_X_test = np.reshape(X_test, (X_test.shape[0]*X_test.shape[1],120))
    new_y_test = np.reshape(y_test, (y_test.shape[0]*y_test.shape[1]))

    ### https://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/
    
    ### Offline Analysis
    clf = util.apply_classifier(clf_type, cc = C_best)

    clf.fit(new_X_train_total, new_y_train_total)
    # y_score_offline = clf.predict_proba(new_X_train_total)
    y_stats_offline = clf.decision_function(new_X_train_total)
    y_stats_H0 = y_stats_offline[new_y_train_total == 0]
    y_stats_H1 = y_stats_offline[new_y_train_total == 1]

    scaler = preprocessing.StandardScaler().fit(y_stats_offline[:, None])
    kde_0_scale, kde_1_scale = get_likelihood_pdf(y_stats_H0, y_stats_H1, scaler)


    ### Online analysis
    X_test_trial_total = np.reshape(X_test, (X_test.shape[0]//seq_per_trial, seq_per_trial, flash_per_seq,120))
    y_stats_offline = clf.decision_function(new_X_train_total)

    flash_test_mat = np.reshape(flash_test, (np.int64(flash_test.shape[0]/seq_per_trial),seq_per_trial, flash_per_seq))

    prior = 1/36

    char_correct_test = []

    start = timer()

    for i in range(test_trial_num): ### for each trial 15
        
        flash_test_trial = flash_test_mat[i]
        X_test_trial = X_test_trial_total[i]
        
        char_proba_mat = np.zeros((6,6)) + prior # store posterior proba
        # char_proba_mat_H0 = np.zeros((6,6)) + p_H0
        
        seq_index = 0
        char_hat = ""

        ### static stopping criteria 10
        for seq_index in range(seq_per_trial):

            X_test_seq = X_test_trial[seq_index]
            flash_seq = flash_test_trial[seq_index]
            
            # Bayesian update
            for index in range(flash_per_seq): # 12 flashes per seq
                y_stats_online = clf.decision_function(X_test_seq[index].reshape(1, -1))
                flash_index = flash_seq[index]
                
                # scaled decision statistics
                y_stats_online_scaled = scaler.transform(y_stats_online[:, None])
                
                # likelihood function
                p_lambda_H0 = np.exp(kde_0_scale.score_samples(y_stats_online_scaled.reshape(1, -1)))
                p_lambda_H1 = np.exp(kde_1_scale.score_samples(y_stats_online_scaled.reshape(1, -1)))

                # update w. Bayes Rule:
                if ((flash_seq[index] > 6) and (flash_seq[index] <= 12)):
                    row_idx = flash_seq[index] - 1 - 6
                    
                    # masking
                    mask = np.ones(char_proba_mat.shape[0], bool)
                    mask[row_idx] = False

                    # update p_lambda_H1 for flashed
                    char_proba_mat[row_idx, :] = char_proba_mat[row_idx, :] * p_lambda_H1/(np.sum(char_proba_mat[row_idx, :] * p_lambda_H1) + np.sum(char_proba_mat[mask, :] * p_lambda_H0))

                    # update p_lambda_H0 for unflashed
                    char_proba_mat[mask, :] = char_proba_mat[mask, :] * p_lambda_H0/(np.sum(char_proba_mat[row_idx, :] * p_lambda_H1) + np.sum(char_proba_mat[mask, :] * p_lambda_H0))
                    
                elif ((flash_seq[index] <= 6) and (flash_seq[index] >= 0)):
                    col_idx = flash_seq[index] - 1
                                
                    # masking
                    mask = np.ones(char_proba_mat.shape[0], bool)
                    mask[col_idx] = False
            
                    # update p_lambda_H1 for flashed
                    char_proba_mat[:, col_idx] = char_proba_mat[:, col_idx] * p_lambda_H1/(np.sum(char_proba_mat[:, col_idx] * p_lambda_H1) + np.sum(char_proba_mat[:, mask] * p_lambda_H0))

                    # update p_lambda_H0 for unflashed                
                    char_proba_mat[:, mask] = char_proba_mat[:, mask] * p_lambda_H0/(np.sum(char_proba_mat[:, col_idx] * p_lambda_H1) + np.sum(char_proba_mat[:, mask] * p_lambda_H0))
        
            char_hat = board[np.where(char_proba_mat == np.max(char_proba_mat))]
    
        if (char_hat == char_label[i+train_trial_num]): 
            char_correct_test.append(char_hat) #Identify how many chars get correct
            
    char_correct_test = np.array(char_correct_test).flatten()     

    train_accuracy = clf.score(new_X_train_total, new_y_train_total)        
    test_accuracy = clf.score(new_X_test, new_y_test)

    print_results(test_trial_num, char_correct_test, start, train_accuracy, test_accuracy)

    return clf

def print_results(test_trial_num, char_correct_test, start, train_accuracy, test_accuracy):
    print(f'The accuracy of test character selection: {len(char_correct_test)/test_trial_num}') # n/15      
    print(f'The total training accuracy of the classifier is: {train_accuracy}')
    print(f'The testing accuracy of the classifier is: {test_accuracy}')
    print(f'Runtime is: {timer() - start}')

def get_likelihood_pdf(y_stats_H0, y_stats_H1, scaler):
    y_stats_H0_scale = scaler.transform(y_stats_H0[:, None])
    y_stats_H1_scale = scaler.transform(y_stats_H1[:, None])

    ### likelihood pdf kde
    kde_0_scale = KernelDensity(kernel='gaussian').fit(y_stats_H0_scale)
    kde_1_scale = KernelDensity(kernel='gaussian').fit(y_stats_H1_scale)
    return kde_0_scale,kde_1_scale

if __name__ == '__main__':    
    #%% Load data
    A01 = sio.loadmat("C:\\Users\\y\\OneDrive\\Documents\\College\\Junior\\BCIs\\Projects\\1\\A01.mat")
    A02 = sio.loadmat("C:\\Users\\y\\OneDrive\\Documents\\College\\Junior\\BCIs\\Projects\\1\\A02.mat")
    A03 = sio.loadmat("C:\\Users\\y\\OneDrive\\Documents\\College\\Junior\\BCIs\\Projects\\1\\A03.mat")
    A04 = sio.loadmat("C:\\Users\\y\\OneDrive\\Documents\\College\\Junior\\BCIs\\Projects\\1\\A04.mat")
    A05 = sio.loadmat("C:\\Users\\y\\OneDrive\\Documents\\College\\Junior\\BCIs\\Projects\\1\\A05.mat")
    A06 = sio.loadmat("C:\\Users\\y\\OneDrive\\Documents\\College\\Junior\\BCIs\\Projects\\1\\A06.mat")
    A07 = sio.loadmat("C:\\Users\\y\\OneDrive\\Documents\\College\\Junior\\BCIs\\Projects\\1\\A07.mat")
    A08 = sio.loadmat("C:\\Users\\y\\OneDrive\\Documents\\College\\Junior\\BCIs\\Projects\\1\\A08.mat")

    # C_reg = np.logspace(-6, 0, num=7)

    clf = dynamic_stopping_criterion(A08, threshold = 0.9)
    # clf.score(A01)
    print("################################# DONE ###################################")
    # static_stopping_criterion(A06)

    # ssc.single_stop_criterion(A08)
    # dsc.dsc_CV(C_reg = C_reg, clf_type = 'LDA', dataset = A04, train_trial_num = 20)

