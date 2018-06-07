#!/bin/env python3


#SBATCH -n 2


#SBATCH --gres=gpu:2


#SBATCH --mem=64G


#SBATCH -p long


#SBATCH -C K80


#SBATCH -o results/test_image-based_3secs_64bin.out


#SBATCH -t 168:00:00


#### NOTE ####
# In Version 2, we grid search the hyper-parameter for our CNN.

### THE RATIO IS 1.0 for training !!!!

from os import walk
import numpy as np
import pandas as pd
import copy
import keras
# import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from collections import defaultdict # High performance container datatypes compared to dict, list, set & tuple
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from random import choice
import csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

csv_output = open("results/image_3secs_64bin.csv",'wt')
csv_output_binCase = open("results/image_3secs_64bin_binCase.csv",'wt')
csv_output_tacc = open("results/image_3secs_64bin_bestTrainAcc.csv", 'wt')


############# Data loading and processing pipeline ###########
# A couple of classes let us deal with parameters, and preprocessing of our downloaded Physionet data
# we do some preprocessing on ECG signal, incuding butterworth bandpass filter, moving average filter,
# and select the usable training data (previously identified usable)

# For Param class, there are several things need to be specified:
# 1. window size
# 2. overlap percentage for window to create one data instance (decides number of unaltered class data instance)
# 3. start and end time of train and test dataset
# 4. the number of the altered data (decides number of all other classes data instances)

class Param(object):
    
    def __init__(self):        

        self.subject_list = ['f2o01', 'f2o03', 'f2o04', 'f2o06', 'f2o07', 'f2y01',
                        'f2y02', 'f2y03', 'f2y04', 'f2y05', 'f2y06', 'f2y07',
                        'mgh001', 'mgh010', 'mgh016', 'mgh019', 'mgh029', 'mgh033', 'mgh035', 'mgh036', 'mgh051', 
                        'mgh052', 'mgh069', 'mgh079', 'mgh087', 'mgh088', 'mgh098', 'mgh102', 'mgh105', 'mgh129', 
                        'mgh143', 'mgh191', 'mgh195']
    
        self.subject_G_list = ['f2o01', 'f2o03', 'f2o04', 
                               'f2y01','f2y02', 'f2y03', 'f2y04', 'f2y05',
                               'mgh001', 'mgh010', 'mgh016', 'mgh019', 'mgh029', 'mgh033', 
                               'mgh052', 'mgh069', 'mgh079', 'mgh087', 'mgh088', 'mgh098',
                               'mgh143', 'mgh191', 'mgh195'
                              ]
        self.subject_notG_list = ['f2o06', 'f2o07',
                                  'f2y06', 'f2y07',
                                  'mgh035','mgh036', 'mgh051',
                                  'mgh102', 'mgh105', 'mgh129'
                                 ]
        
        self.sampling_rate = 360
        self.window_size = 3
        self.overlap = 0.9 # overlap percentage
        self.segment_len = self.window_size*self.sampling_rate
        self.lowcut_freq = 1
        self.highcut_freq = 50
        self.num_signals = 2
        self.num_classes = 5 # change this model in CNN model manually
        self.bin = 50

        self.data_start = 5*60*self.sampling_rate - 1 
        self.data_end = 51*60*self.sampling_rate - 1
        
        self.hist_data_start =  5*60*self.sampling_rate - 1
        self.hist_data_end = 15*60*self.sampling_rate - 1
        
        self.train_data_start = 15*60*self.sampling_rate - 1
        self.train_data_end = 35*60*self.sampling_rate - 1
        
        self.test_data_start_G = 35*60*self.sampling_rate - 1
        self.test_data_end_G = 51*60*self.sampling_rate - 1
        
        self.test_data_start_notG = 15*60*self.sampling_rate - 1
        self.test_data_end_notG = 51*60*self.sampling_rate - 1
        
        self.async_duration_diff = [2,3,5,10,20,30,60,90,120,150,180,210,240,270,300]


        # number of the Unaltered segments in train and test dataset for target subject
        win_stride = int(self.segment_len*(1-self.overlap)) # calculate steps to move the window
        num_unalt_seg_train = ((self.train_data_end - self.train_data_start) - self.segment_len)//win_stride + 1
        num_unalt_seg_test_G = ((self.test_data_end_G - self.test_data_start_G) - self.segment_len)//win_stride + 1
        num_unalt_seg_test_notG = 0
        
        
        # Set the number of the Altered data in train and test dataset for target subject
        # max number of segments of train and test data of target subject
        max_alt_seg_train = (self.train_data_end - self.train_data_start)/self.segment_len
        self.max_alt_seg_test_G = int((self.test_data_end_G - self.test_data_start_G)/self.segment_len)
        max_alt_seg_test_notG = (self.test_data_end_notG - self.test_data_start_notG)/self.segment_len
        
        
        self.seg_altered_per_subject_train = int(1 * max_alt_seg_train) # set this ratio
        self.seg_altered_per_subject_test_G = int(1 * self.max_alt_seg_test_G) # set this ratio       
        self.seg_altered_per_subject_test_notG = int(1 * max_alt_seg_test_notG) # set this ratio 
        
   
        print("For each subject, Train Dataset ===>>>\n")
        print("\tTotal # of data instance of Unaltered Class: " + str(num_unalt_seg_train))     
        print("\tTotal # of data instance of three Altered Classes: " + 
              str(self.seg_altered_per_subject_train * (len(self.subject_G_list) - 1)*4))
        print("\tTotal # of data instance of train dataset: " + 
              str(num_unalt_seg_train + self.seg_altered_per_subject_train * (len(self.subject_G_list) - 1)*4) + "\n")
        
        print("For each subject, Group G's Test Dataset ===>>>")
        print("\tTotal # of data instance of Unaltered Class: " + str(num_unalt_seg_test_G))
        print("\tTotal # of data instance of three Altered Classes: " + 
              str(self.seg_altered_per_subject_test_G * (len(self.subject_G_list) - 1)*4))
        print("\tTotal # of data instance of test dataset: " + 
              str(num_unalt_seg_test_G + self.seg_altered_per_subject_test_G * (len(self.subject_G_list) - 1)*4) + "\n")
        
        print("For each subject, Group NOT G's Test Dataset ===>>>")
        print("\tTotal # of data instance of Unaltered Class: " + str(num_unalt_seg_test_notG))
        print("\tTotal # of data instance of three Altered Classes: " + 
              str(self.seg_altered_per_subject_test_notG * len(self.subject_notG_list)*4))
        print("\tTotal # of data instance of test dataset: " + 
              str(num_unalt_seg_test_notG + self.seg_altered_per_subject_test_notG * len(self.subject_notG_list)*4) + "\n")
        
        
class PhyDataset(object):
    
    def __init__(self,data_path,Param):
        self.data_path = data_path
        self.p = Param
        
    def butter_bandpass_filter(self, data):
        nyq = 0.5 * self.p.sampling_rate
        low = self.p.lowcut_freq / nyq
        high = self.p.highcut_freq / nyq
        b, a = butter(3, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y
   
    def load_data(self,subject_list):
        data = defaultdict(dict)
        for subject in subject_list:
            data[subject] = pd.read_table(self.data_path + subject + ".txt", skiprows=[0,1],delim_whitespace=True,
                                         names = ["ElapsedTime", "RESP", "ECG", "ABP"])
            data[subject].drop(['RESP'], axis = 1, inplace = True) # Drop Respiration signal
        return data
    
    def preprocess_data(self, data):
        for subject in list(data.keys()):            
            # Filter the ECG of the subject data
            data[subject]['ECG'] = self.butter_bandpass_filter(data[subject]['ECG'])
            data[subject]['ECG'] = data[subject]['ECG'].rolling(window = 5).mean()
            data[subject] = data[subject].iloc[self.p.data_start:self.p.data_end].reset_index(drop=True)
        return data
    
    # Generate raw train data 
    def gen_raw_data(self,df_rawdata,start,end):
        
        rawdata = defaultdict(dict)
        # find the index of train/test data regarding to the total usable data
        temp_start = int(start - self.p.data_start)
        temp_end = int(end - self.p.data_start)
        
        for subject in list(df_rawdata.keys()):
            rawdata[subject] = df_rawdata[subject].iloc[temp_start:temp_end].reset_index(drop=True)

        return rawdata



# Functions for generating 5 different classes of training data (ONLY FOR ONE SUBJECT)
# Class_0 means that both ECG and ABP are Not Modified
# Class_1 means that only ECG is Modified 
# Class_2 means that only ABP is Modified
# Class_3 means that both ECG and ABP are Modified
# Class_4 means that it's historical data manipulation

class generate_data_instance(object):
    
    def __init__(self,Param):
        self.p = Param
        self.data = np.array([])
        self.labels = np.array([])
        
#      # make data snippets based on the window size
#     def make_segment(self, data, idx_start, idx_end, signal_idx):
#         temp_data = np.empty(shape=(1, int(self.p.segment_len),1))
#         temp_data[0,:,0] = np.array(data[idx_start: idx_end])
        
#         return temp_data
    
    def append_data_labels(self, data, labels):
        if self.data.size:
            self.data = np.append(self.data, data, axis=0)
            self.labels = np.append(self.labels, labels, axis=0)
        else:
            self.data = data
            self.labels = labels
    
    def norm(self, data):
        seg_max = np.max(data)
        seg_min = np.min(data)
        
        if(seg_max - seg_min) == 0:
            norm_data = np.zeros(len(data))
        else:
            norm_data = (data - seg_min)/(seg_max - seg_min)
        
        return norm_data
           
    
    # Both ECG and ABP are not Modified
    # Greate class 0 data instances using overlap window
    def make_class_0_data(self, subject, rawdata):        
        temp_x = []
        image = []
        
        stride = int(self.p.segment_len*(1-self.p.overlap)) # calculate steps to move the window
        total_num_seg = (rawdata[subject].shape[0] - self.p.segment_len)//stride + 1
                
        # For-loop to make segment of each signal
        for start in range(0,int(total_num_seg)):
        
            # Get Unaltered ECG from target subject
            temp_sig_1 = np.array(rawdata[subject].iloc[start*stride:start*stride+self.p.segment_len,1])
                        
            temp_sig_1 = self.norm(temp_sig_1)

            # Get Unaltered ABP from target subject
            temp_sig_2 = np.array(rawdata[subject].iloc[start*stride:start*stride+self.p.segment_len,2])
            
            # remove MGH ABP and Fantasia ECG for which has noises
            if (subject[0] == "m" and np.std(temp_sig_2) >= 1 and np.max(temp_sig_2)>= 10) or (
                subject[0] == "f" and np.max(temp_sig_2) >= 0):
                
                temp_sig_2 = self.norm(temp_sig_2)
                # ignore (not include) the straight line signal (i.e., seg_min = seg_max)
                if np.any(temp_sig_1) and np.any(temp_sig_2):
                    portrait = self.ts_to_portrait(temp_sig_1,temp_sig_2,self.p.bin)
                    image.append(self.portrait_to_image(portrait))
        
        temp_x = np.asarray(image) #data
        temp_y = np.zeros(np.shape(temp_x)[0]) #labels
        self.append_data_labels(temp_x,temp_y)       
     
    # Only ECG is Modified
    # Greate class 1 data instances
    def make_class_1_data(self, subject, rawdata,seg_altered_per_subject,option):
        temp_x = []
        image = []
        
        if option == 'G':
            other_subject_list = self.p.subject_G_list.copy() #find out the other subjects, for python 3
            # other_subject_list = copy.copy(self.p.subject_list) #find out the other subjects, for python 2
            other_subject_list.remove(subject)
        if option == 'notG':
            other_subject_list = self.p.subject_notG_list.copy()
            
        for other_subject in other_subject_list:            
            for start in range(0,seg_altered_per_subject):
                
                # The start and end point of segment
                # Use mod operation as moving window is longer than target subject's data
                other_tmp_start = start*self.p.segment_len
                other_tmp_end = (start+1)*self.p.segment_len
                target_tmp_start = (start*self.p.segment_len) % (len(rawdata[subject])+1)
                target_tmp_end = ((start+1)*self.p.segment_len) % (len(rawdata[subject])+1)
                
                if target_tmp_end - target_tmp_start == 1080:
                    # Get Altered ECG from other subject
                    temp_sig_1 = np.array(rawdata[other_subject].iloc[other_tmp_start:other_tmp_end,1])
                    temp_sig_1 = self.norm(temp_sig_1)

                    # Get Unaltered ABP from target subject
                    temp_sig_2 = np.array(rawdata[subject].iloc[target_tmp_start:target_tmp_end,2])
                    # remove MGH ABP and Fantasia ECG for which has noises
                    if (subject[0] == "m" and np.std(temp_sig_2) >= 1 and np.max(temp_sig_2)>= 10) or (
                        subject[0] == "f" and np.max(temp_sig_2) >= 0):
                        temp_sig_2 = self.norm(temp_sig_2)

                        # ignore (not include) the straight line signal (i.e., seg_min = seg_max)
                        if np.any(temp_sig_1) and np.any(temp_sig_2):
                            portrait = self.ts_to_portrait(temp_sig_1,temp_sig_2,self.p.bin)
                            image.append(self.portrait_to_image(portrait))
        
        temp_x = np.asarray(image) #data
        temp_y = np.ones(np.shape(temp_x)[0]) #labels
        self.append_data_labels(temp_x,temp_y)
    
    # Only ABP is Modified
    # Greate class 2 data instances
    def make_class_2_data(self, subject, rawdata,seg_altered_per_subject,option):
        temp_x = []
        image = []
        
        if option == 'G':
            other_subject_list = self.p.subject_G_list.copy() #find out the other subjects, for python 3
            # other_subject_list = copy.copy(self.p.subject_list) #find out the other subjects, for python 2
            other_subject_list.remove(subject)
        if option == 'notG':
            other_subject_list = self.p.subject_notG_list.copy()
        
        for other_subject in other_subject_list:            
            for start in range(0,seg_altered_per_subject):
                
                # The start and end point of segment
                other_tmp_start = start*self.p.segment_len
                other_tmp_end = (start+1)*self.p.segment_len
                target_tmp_start = (start*self.p.segment_len) % (len(rawdata[subject])+1)
                target_tmp_end = ((start+1)*self.p.segment_len) % (len(rawdata[subject])+1)
                
                if target_tmp_end - target_tmp_start == 1080:
                    # Get Unaltered ECG from target subject
                    temp_sig_1 = np.array(rawdata[subject].iloc[target_tmp_start:target_tmp_end,1])
                    temp_sig_1 = self.norm(temp_sig_1)

                    # Get Altered ABP from other subject
                    temp_sig_2 = np.array(rawdata[other_subject].iloc[other_tmp_start:other_tmp_end,2])
                    # remove MGH ABP and Fantasia ECG for which has noises
                    if (subject[0] == "m" and np.std(temp_sig_2) >= 1 and np.max(temp_sig_2)>= 10) or (
                        subject[0] == "f" and np.max(temp_sig_2) >= 0):
                        
                        temp_sig_2 = self.norm(temp_sig_2)

                        # ignore (not include) the straight line signal (i.e., seg_min = seg_max)
                        if np.any(temp_sig_1) and np.any(temp_sig_2):
                            portrait = self.ts_to_portrait(temp_sig_1,temp_sig_2,self.p.bin)
                            image.append(self.portrait_to_image(portrait))
        
        temp_x = np.asarray(image) #data
        temp_y = np.full(np.shape(temp_x)[0],2) #labels
        self.append_data_labels(temp_x,temp_y)
    
    # Both ECG and ABP are Modified
    # Greate class 3 data instances
    def make_class_3_data(self, subject, rawdata,seg_altered_per_subject,option):
        temp_x = []
        image = []
        
        if option == 'G':
            other_subject_list = self.p.subject_G_list.copy() #find out the other subjects, for python 3
            # other_subject_list = copy.copy(self.p.subject_list) #find out the other subjects, for python 2
            other_subject_list.remove(subject)
        if option == 'notG':
            other_subject_list = self.p.subject_notG_list.copy()
        
        # Altered ECG and ABP are both from the same (other) subject
        for other_subject in other_subject_list:            
            for start in range(0,seg_altered_per_subject):
                
                tmp_start = start*self.p.segment_len
                tmp_end = (start+1)*self.p.segment_len
                
                # Get Altered ECG from other subject
                temp_sig_1 = np.array(rawdata[other_subject].iloc[tmp_start:tmp_end,1])
                temp_sig_1 = self.norm(temp_sig_1)

                # Get Altered ABP from other subject
                temp_sig_2 = np.array(rawdata[other_subject].iloc[tmp_start:tmp_end ,2])
                
                # remove MGH ABP and Fantasia ECG for which has noises
                if (subject[0] == "m" and np.std(temp_sig_2) >= 1 and np.max(temp_sig_2)>= 10) or (
                    subject[0] == "f" and np.max(temp_sig_2) >= 0):
                    
                    temp_sig_2 = self.norm(temp_sig_2)
                    # ignore (not include) the straight line signal (i.e., seg_min = seg_max)
                    if np.any(temp_sig_1) and np.any(temp_sig_2):
                        portrait = self.ts_to_portrait(temp_sig_1,temp_sig_2,self.p.bin)
                        image.append(self.portrait_to_image(portrait))
       
        # Altered ECG and ABP are from the different (other) subject
        for other_subject in other_subject_list:
            # create a other_other_subject_list excluding the other_subject
            # random pick other subject's abp or ecg signals
            other_other_subject_list = other_subject_list.copy()
            other_other_subject_list.remove(other_subject)
            
            for start in range(0,seg_altered_per_subject):
                if (start % 2) == 0:
                    ecg_subject = other_subject
                    abp_subject = choice(other_other_subject_list)
                else:
                    ecg_subject = choice(other_other_subject_list)
                    abp_subject = other_subject
                
                tmp_start = start*self.p.segment_len
                tmp_end = (start+1)*self.p.segment_len
                
                # Get Unaltered ECG from other subject
                temp_sig_1 = np.array(rawdata[ecg_subject].iloc[tmp_start:tmp_end,1])
                temp_sig_1 = self.norm(temp_sig_1)
            
                # Get Altered ABP from other subject
                temp_sig_2 = np.array(rawdata[abp_subject].iloc[tmp_start:tmp_end ,2])
                
                # remove MGH ABP and Fantasia ECG for which has noises
                if (subject[0] == "m" and np.std(temp_sig_2) >= 1 and np.max(temp_sig_2)>= 10) or (
                    subject[0] == "f" and np.max(temp_sig_2) >= 0):
                    
                    temp_sig_2 = self.norm(temp_sig_2)
                    # ignore (not include) the straight line signal (i.e., seg_min = seg_max)
                    if np.any(temp_sig_1) and np.any(temp_sig_2):
                        portrait = self.ts_to_portrait(temp_sig_1,temp_sig_2,self.p.bin)
                        image.append(self.portrait_to_image(portrait))
        
        temp_x = np.asarray(image) #data
        temp_y = np.full(np.shape(temp_x)[0],3) #labels
        self.append_data_labels(temp_x,temp_y)
    
    
    def make_hist_class_data_ECG(self, subject,rawdata_test,rawdata_hist):
        temp_x = []
        image = []
        
        ## Only replace current ECG with historical ECG
        ## hist data is shorter than test data
        for start in range(0,int(len(rawdata_test[subject])/self.p.segment_len)):
            target_tmp_start = start*self.p.segment_len
            target_tmp_end = (start+1)*self.p.segment_len
            hist_tmp_start = (start*self.p.segment_len) % (len(rawdata_hist[subject])+1)  
            hist_tmp_end = ((start+1)*self.p.segment_len) % (len(rawdata_hist[subject])+1)
            
            if hist_tmp_end - hist_tmp_start == 1080:
                # Get Altered ECG from historical data
                temp_sig_1 = np.array(rawdata_hist[subject].iloc[hist_tmp_start:hist_tmp_end,1])
                temp_sig_1 = self.norm(temp_sig_1)
    
                # Get Unaltered ABP from target subject
                temp_sig_2 = np.array(rawdata_test[subject].iloc[target_tmp_start:target_tmp_end,2])
                # remove MGH ABP and Fantasia ECG for which has noises
                if (subject[0] == "m" and np.std(temp_sig_2) >= 1 and np.max(temp_sig_2)>= 10) or (
                    subject[0] == "f" and np.max(temp_sig_2) >= 0):
                    temp_sig_2 = self.norm(temp_sig_2)

                    # ignore (not include) the straight line signal (i.e., seg_min = seg_max)
                    if np.any(temp_sig_1) and np.any(temp_sig_2):
                        portrait = self.ts_to_portrait(temp_sig_1,temp_sig_2,self.p.bin)
                        image.append(self.portrait_to_image(portrait))
        
        temp_x = np.asarray(image) #data
        temp_y = np.full(np.shape(temp_x)[0],4) #labels
        self.append_data_labels(temp_x,temp_y)
    
    def make_hist_class_data_ABP(self, subject,rawdata_test,rawdata_hist):
        temp_x = []
        image = []
        
        for start in range(0,int(len(rawdata_test[subject])/self.p.segment_len)):

            ## Only replace current ABP with historical ABP
            ## hist data is shorter than test data
            target_tmp_start = start*self.p.segment_len
            target_tmp_end = (start+1)*self.p.segment_len
            hist_tmp_start = (start*self.p.segment_len) % (len(rawdata_hist[subject])+1)
            hist_tmp_end = ((start+1)*self.p.segment_len) % (len(rawdata_hist[subject])+1)

            if hist_tmp_end - hist_tmp_start == 1080:
                # Get Unaltered ECG from target subject
                temp_sig_1 = np.array(rawdata_test[subject].iloc[target_tmp_start:target_tmp_end,1])
                temp_sig_1 = self.norm(temp_sig_1)

                # Get Altered historical ABP from other subject
                temp_sig_2 = np.array(rawdata_hist[subject].iloc[hist_tmp_start:hist_tmp_end,2])
                # remove MGH ABP and Fantasia ECG for which has noises
                if (subject[0] == "m" and np.std(temp_sig_2) >= 1 and np.max(temp_sig_2)>= 10) or (
                    subject[0] == "f" and np.max(temp_sig_2) >= 0):

                    temp_sig_2 = self.norm(temp_sig_2)

                    # ignore (not include) the straight line signal (i.e., seg_min = seg_max)
                    if np.any(temp_sig_1) and np.any(temp_sig_2):
                        portrait = self.ts_to_portrait(temp_sig_1,temp_sig_2,self.p.bin)
                        image.append(self.portrait_to_image(portrait))
        
        temp_x = np.asarray(image) #data
        temp_y = np.full(np.shape(temp_x)[0],4) #labels
        self.append_data_labels(temp_x,temp_y)
        
    # def make_hist_class_data_sync(self, subject, rawdata_hist):
    #     temp_x = []

    #     sig_idx_1 = [] #ECG
    #     sig_idx_2 = [] #ABP
        
        
    #     ## Only replace current ECG with historical ECG
    #     ## hist data is shorter than test data  
    #     for start in range(0,int((self.p.hist_data_end - self.p.hist_data_start)/self.p.segment_len - 1)): # since we replaced with both hist data
                        
    #         hist_tmp_start = (start*self.p.segment_len) 
    #         hist_tmp_end = ((start+1)*self.p.segment_len)

    #         # Get Altered ECG from historical data
    #         temp_sig_1 = np.array(rawdata_hist[subject].iloc[hist_tmp_start:hist_tmp_end,1])
    #         temp_sig_1 = self.norm(temp_sig_1)

    #         # Get Altered ABP from historical data
    #         temp_sig_2 = np.array(rawdata_hist[subject].iloc[hist_tmp_start:hist_tmp_end,2])
    #         # remove MGH ABP and Fantasia ECG for which has noises
    #         if (subject[0] == "m" and np.std(temp_sig_2) >= 1 and np.max(temp_sig_2)>= 10) or (
    #             subject[0] == "f" and np.max(temp_sig_2) >= 0):
    #             temp_sig_2 = self.norm(temp_sig_2)

    #             # ignore (not include) the straight line signal (i.e., seg_min = seg_max)
    #             if np.any(temp_sig_1) and np.any(temp_sig_2):
    #                 sig_idx_1.append(temp_sig_1) 
    #                 sig_idx_2.append(temp_sig_2)

    #     sig_idx_1 = np.asarray(sig_idx_1)
    #     sig_idx_2 = np.asarray(sig_idx_2)

    #     temp_x = np.asarray([sig_idx_1,sig_idx_2]) #data
    #     temp_x = np.swapaxes(temp_x,0,1)
    #     temp_x = np.swapaxes(temp_x,1,2)
    #     temp_y = np.full(int(temp_x.shape[0]),4) #labels
    #     self.append_data_labels(temp_x,temp_y)
        
    def make_hist_class_data_notSync(self, subject, rawdata_hist):
        temp_x = []
        image = [] 
        
        tmp_seg_num = int((self.p.hist_data_end - self.p.hist_data_start)/self.p.segment_len - 1)
        ## Only replace current ECG with historical ECG
        ## hist data is shorter than test data
        for start in range(0,tmp_seg_num): # since we replaced with both hist data

            # time for one historical signal
            hist_tmp_start = (start*self.p.segment_len) 
            hist_tmp_end = ((start+1)*self.p.segment_len)
            # time for another async historical signal
            asy_sig_start = choice([j for j in range(0,tmp_seg_num) if j not in [start]]) * self.p.segment_len
            asy_sig_end = asy_sig_start + self.p.segment_len

            if (start % 2) == 0:
                sig_1_start = hist_tmp_start
                sig_1_end = hist_tmp_end
                sig_2_start = asy_sig_start
                sig_2_end = asy_sig_end
            else:
                sig_1_start = asy_sig_start
                sig_1_end = asy_sig_end
                sig_2_start = hist_tmp_start
                sig_2_end = hist_tmp_end

            # Get Altered ECG from historical data
            temp_sig_1 = np.array(rawdata_hist[subject].iloc[sig_1_start:sig_1_end,1])
            temp_sig_1 = self.norm(temp_sig_1)

            # Get Unaltered ABP from target subject
            temp_sig_2 = np.array(rawdata_hist[subject].iloc[sig_2_start:sig_2_end,2])
            # remove MGH ABP and Fantasia ECG for which has noises
            if (subject[0] == "m" and np.std(temp_sig_2) >= 1 and np.max(temp_sig_2)>= 10) or (
                subject[0] == "f" and np.max(temp_sig_2) >= 0):
                temp_sig_2 = self.norm(temp_sig_2)

                # ignore (not include) the straight line signal (i.e., seg_min = seg_max)
                if np.any(temp_sig_1) and np.any(temp_sig_2):
                    portrait = self.ts_to_portrait(temp_sig_1,temp_sig_2,self.p.bin)
                    image.append(self.portrait_to_image(portrait))
        
        temp_x = np.asarray(image) #data
        temp_y = np.full(np.shape(temp_x)[0],4) #labels
        self.append_data_labels(temp_x,temp_y)
    
    def make_hist_class_data_notSync_difsecs(self, subject, rawdata_hist,duration_diff):
        temp_x = []
        image = [] 
        
        ## Only replace current ECG with historical ECG
        ## hist data is shorter than test data
        tmp_seg_num = int((self.p.hist_data_end - self.p.hist_data_start)/self.p.segment_len - 1)

        for sig_first_flag in range(1,3):
            for start in range(0,tmp_seg_num): # since len(rawdata_hist) <= len(rawdata_test)

                # time for one historical signal
                hist_tmp_start = (start*self.p.segment_len) 
                hist_tmp_end = ((start+1)*self.p.segment_len)
                # time for another async historical signal
                asy_sig_start = hist_tmp_start + duration_diff*self.p.sampling_rate
                asy_sig_end = asy_sig_start + self.p.segment_len

                if asy_sig_start <= (len(rawdata_hist[subject]) - 1) and asy_sig_end <= (len(rawdata_hist[subject]) -1):      
                    if sig_first_flag == 1:  #means ECG is older than ABP
                        sig_1_start = hist_tmp_start
                        sig_1_end = hist_tmp_end
                        sig_2_start = asy_sig_start
                        sig_2_end = asy_sig_end
                    else: #means ABP is older than ECG 
                        sig_1_start = asy_sig_start
                        sig_1_end = asy_sig_end
                        sig_2_start = hist_tmp_start
                        sig_2_end = hist_tmp_end

                    # Get Altered ECG from historical data
                    temp_sig_1 = np.array(rawdata_hist[subject].iloc[sig_1_start:sig_1_end,1])
                    temp_sig_1 = self.norm(temp_sig_1)

                    # Get Unaltered ABP from target subject
                    temp_sig_2 = np.array(rawdata_hist[subject].iloc[sig_2_start:sig_2_end,2])
                    # remove MGH ABP and Fantasia ECG for which has noises
                    if (subject[0] == "m" and np.std(temp_sig_2) >= 1 and np.max(temp_sig_2)>= 10) or (
                        subject[0] == "f" and np.max(temp_sig_2) >= 0):
                        temp_sig_2 = self.norm(temp_sig_2)

                        # ignore (not include) the straight line signal (i.e., seg_min = seg_max)
                        if np.any(temp_sig_1) and np.any(temp_sig_2):
                            portrait = self.ts_to_portrait(temp_sig_1,temp_sig_2,self.p.bin)
                            image.append(self.portrait_to_image(portrait))
        
        temp_x = np.asarray(image) #data
        temp_y = np.full(np.shape(temp_x)[0],4) #labels
        self.append_data_labels(temp_x,temp_y)

    def timeseries_to_image(self,data):
        imaged_data = data.copy().reshape(data.shape[0],data.shape[1],1,data.shape[2])
        return imaged_data

    def ts_to_portrait(self,norm_sig_1,norm_sig_2,bin_num):
        channel_num = 1
        portrait = np.zeros((bin_num, bin_num,channel_num), dtype=int)
        for i in range(0,len(norm_sig_1)):
            col = int(norm_sig_2[i]/(1/bin_num))
            row = bin_num - int(norm_sig_1[i]/(1/bin_num)) #flip the row
            if row == bin_num:
                row = row - 1
            if col == bin_num:
                col = col - 1
            portrait[row,col,0] = portrait[row,col,0] + 1
        
        return(portrait)
    
    def portrait_to_image(self,portrait):
        image = np.copy(portrait)
        nonzeroind = np.nonzero(image)
        for idx in range(0,np.shape(nonzeroind)[1]):
            image[nonzeroind[0][idx],nonzeroind[1][idx],nonzeroind[2][idx]] = 1
            
        return(image)
    


class my_model(object):
    
    # kernel_constraint: Constraining the weight matrix directly is another kind of regularization. 
    #                    If you use a simple L2 regularization term you penalize high weights with your 
    #                    loss function. With this constraint, you regularize directly.
    
    def __init__(self,Param):
        self.p = Param

        self.param_grid = {'batch_size': [50],
                           'epochs': [20],
                           # 'optimizer':['adam','rmsprop','sgd','adamax','nadam'],
                           'maxnorm_val':[5], 
                           'num_neuron_fc':[200],
                           'act_func1':['relu'],
                           'act_func2':['softmax'],
                           'filt_num_lvl_1':[20],
                           'filt_num_lvl_2':[15],
                           'filt_shape_lvl_1':[(5,5)], # best
                           'filt_shape_lvl_2':[(4,4)], # best
                           'pool_size_lvl_1':[(2,2),(3,3)],
                           'pool_size_lvl_2':[(2,2),(3,3)],
                           'dropOut_rate_lvl_1':[0.2],
                           'dropOut_rate_lvl_2':[0.2],
                           'loss':['categorical_crossentropy']
                          }
    
    def create_ConvolutionalNN(self, epochs, maxnorm_val, num_neuron_fc, act_func1, act_func2, filt_num_lvl_1, filt_num_lvl_2, 
                               filt_shape_lvl_1, filt_shape_lvl_2, pool_size_lvl_1, pool_size_lvl_2,
                               dropOut_rate_lvl_1, dropOut_rate_lvl_2,loss):
        
        model = Sequential()
        model.add(Conv2D(filt_num_lvl_1, filt_shape_lvl_1, input_shape=(self.p.bin, self.p.bin,1),
                         strides=(1, 1),data_format="channels_last", padding='valid', 
                         activation='relu', kernel_constraint=maxnorm(maxnorm_val)))
        model.add(Dropout(dropOut_rate_lvl_1))
        model.add(MaxPooling2D(pool_size = pool_size_lvl_1))
        model.add(Conv2D(filt_num_lvl_2, filt_shape_lvl_2, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size = pool_size_lvl_2))
        model.add(Flatten())
        model.add(Dense(units = num_neuron_fc, activation = act_func1))
        model.add(Dropout(dropOut_rate_lvl_2))
        model.add(Dense(5, activation = act_func2))      
        lrate = 0.01
        decay = lrate/epochs
        sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) #overcome the imbalanced dataset problem
        model.compile(loss = loss, optimizer=sgd, metrics=['accuracy'])
#         model.compile(loss = loss, optimizer=optimizer, metrics=['accuracy'])
        return model




########################################## MAIN CODE ##########################################

parameters = Param()
data_path = "Fantasia_MGH_MIT_360Hz/"

ds = PhyDataset(data_path,parameters)
Group_G_data = ds.load_data(parameters.subject_G_list)
Group_G_data = ds.preprocess_data(Group_G_data)
Group_notG_data = ds.load_data(parameters.subject_notG_list)
Group_notG_data = ds.preprocess_data(Group_notG_data)

## get the raw data (dataframe) for train and test 
rawdata_train = ds.gen_raw_data(Group_G_data,parameters.train_data_start,parameters.train_data_end)
rawdata_test_historical = ds.gen_raw_data(Group_G_data,parameters.hist_data_start,parameters.hist_data_end)
rawdata_test_G = ds.gen_raw_data(Group_G_data,parameters.test_data_start_G,parameters.test_data_end_G)
rawdata_test_notG = ds.gen_raw_data(Group_notG_data,parameters.test_data_start_notG,parameters.test_data_end_notG)
rawdata_test = dict(rawdata_test_G, **rawdata_test_notG)

rawdata_train_1sthalf = ds.gen_raw_data(Group_G_data,parameters.train_data_start,
                                        parameters.train_data_start + int((parameters.train_data_end - parameters.train_data_start)/2))
rawdata_train_2ndhalf = ds.gen_raw_data(Group_G_data, parameters.train_data_start + int((parameters.train_data_end - parameters.train_data_start)/2),
                                        parameters.train_data_end)


#### Use grid search to find the best parameters for each subject's model
#### Then use the best parameters to fit the model and then save the model
#### Last use the trained model to do the prediction
csv_header = ["No Attack", "ECG_altered_G", "ABP_altered_G", "Both_altered_G",
            "ECG_altered_NotG", "ABP_altered_NotG", "Both_altered_NotG", 
            "Hist_test_allCases", "ECG_altered_Hist", "ABP_altered_Hist", "Both_alterd_Hist_notSync"]

if len(parameters.async_duration_diff) != 0:
    for duration_diff in parameters.async_duration_diff:
        tmp_diff = "Both_altered_Hist_" + str(duration_diff) + "_secs"
        csv_header.append(tmp_diff)

writer = csv.writer(csv_output)
writer.writerow(csv_header)

writer_bin = csv.writer(csv_output_binCase)
writer_bin.writerow(csv_header)

writer_tacc = csv.writer(csv_output_tacc)

train_acc = []


for subject in parameters.subject_G_list:

    # Data creation for subject
    # Create train data for each Group G's subject-specific model
    x_train = []
    y_train = []
    y_train_bin = []

    gen_train = generate_data_instance(parameters)
    gen_train.make_class_0_data(subject,rawdata_train)
    gen_train.make_class_1_data(subject,rawdata_train,parameters.seg_altered_per_subject_train,'G')
    gen_train.make_class_2_data(subject,rawdata_train,parameters.seg_altered_per_subject_train,'G')
    gen_train.make_class_3_data(subject,rawdata_train,parameters.seg_altered_per_subject_train,'G')
    gen_train.make_hist_class_data_ECG(subject, rawdata_train_2ndhalf, rawdata_train_1sthalf)
    gen_train.make_hist_class_data_ABP(subject, rawdata_train_2ndhalf, rawdata_train_1sthalf)
    gen_train.make_hist_class_data_notSync(subject, rawdata_train)

    x_train = gen_train.data
    y_train_bin = keras.utils.to_categorical(gen_train.labels) # one hot encode outputs
    y_train = gen_train.labels

    # Create Group G's test data for each subject-specific model
    x_test_G = []
    y_test_G = []
    y_test_G_bin = []

    gen_test_G = generate_data_instance(parameters)
    gen_test_G.make_class_0_data(subject,rawdata_test)
    gen_test_G.make_class_1_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G')
    gen_test_G.make_class_2_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G')
    gen_test_G.make_class_3_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G')
    gen_test_G.make_hist_class_data_ECG(subject, rawdata_test_G, rawdata_test_historical) #ignore this, for non-empty in confusion matrix

    x_test_G  = gen_test_G.data
    y_test_G_bin = keras.utils.to_categorical(gen_test_G.labels) # one hot encode outputs
    y_test_G = gen_test_G.labels


    # Create Group Not G's test data for each subject-specific model
    x_test_notG = []
    y_test_notG = []
    y_test_notG_bin = []

    gen_test_notG = generate_data_instance(parameters)
    gen_test_notG.make_class_0_data(subject,rawdata_test)
    gen_test_notG.make_class_1_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_notG,'notG')
    gen_test_notG.make_class_2_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_notG,'notG')
    gen_test_notG.make_class_3_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_notG,'notG')
    gen_test_notG.make_hist_class_data_ECG(subject, rawdata_test_G, rawdata_test_historical) #ignore this, for non-empty in confusion matrix

    x_test_notG = gen_test_notG.data
    y_test_notG_bin = keras.utils.to_categorical(gen_test_notG.labels) # one hot encode outputs
    y_test_notG = gen_test_notG.labels

    # Create Group G Historical Not_Sync test data (Combined Results)
    x_test_hist = []
    y_test_hist = []
    y_test_hist_bin = []

    gen_test_hist = generate_data_instance(parameters)
    gen_test_hist.make_class_0_data(subject,rawdata_test_G) #this is only for not-empty
    gen_test_hist.make_class_1_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G') #this is only for not-empty
    gen_test_hist.make_class_2_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G') #this is only for not-empty
    gen_test_hist.make_class_3_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G') #this is only for not-empty
    gen_test_hist.make_hist_class_data_ECG(subject, rawdata_test_G, rawdata_test_historical) 
    gen_test_hist.make_hist_class_data_ABP(subject, rawdata_test_G, rawdata_test_historical)
    gen_test_hist.make_hist_class_data_notSync(subject, rawdata_test_historical) 

    x_test_hist = gen_test_hist.data
    y_test_hist_bin = keras.utils.to_categorical(gen_test_hist.labels) # one hot encode outputs
    y_test_hist = gen_test_hist.labels



    print("---------------------------------------------------------------------------------------")
    print(subject)
    print("\n")
    
    print("Evaluation for " + subject + " --- BEGIN!")

    m = my_model(parameters)
    model_built  = KerasClassifier(build_fn = m.create_ConvolutionalNN, 
                                  batch_size = 50,
                                  epochs = 40,
                                  maxnorm_val = 5,
                                  num_neuron_fc = 200,
                                  act_func1 = 'relu', 
                                  act_func2 = 'softmax', 
                                  filt_num_lvl_1 = 20,
                                  filt_num_lvl_2 = 15,
                                  filt_shape_lvl_1 = (5,5), 
                                  filt_shape_lvl_2 = (4,4), 
                                  pool_size_lvl_1 = (3,3),
                                  pool_size_lvl_2 = (3,3),
                                  dropOut_rate_lvl_1 = 0.2, 
                                  dropOut_rate_lvl_2 = 0.2,
                                  loss = 'categorical_crossentropy',  
                                  verbose = False
                                 )

    # kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    # tmp_score = cross_val_score(model_built, x_train, y_train_bin, cv=kfold)
    # train_acc.append(tmp_score.mean()) # 10foldcross score

    model_built.fit(x_train, y_train_bin)
    tmp_loss, tmp_acc = model_built.model.evaluate(x_train, y_train_bin, batch_size = 50,verbose = False)
    train_acc.append(tmp_acc) # evaluation on the training dataset accuracy

    print(model_built.model.summary())


    ##### Group G #####
    pred_class_G = model_built.predict(x_test_G)
    print("Group G test")
    conf = confusion_matrix(y_test_G, pred_class_G)
    print("The confusion matrix for Test Case Group G is showing below:")
    print(conf)
    print("both unmodified prediction accuracy rate:" + str((conf[0][0]/sum(conf[0]))))
    print("modified ECG prediction accuracy rate:" + str((conf[1][1]/sum(conf[1]))))
    print("modified ABP prediction accuracy rate:" + str((conf[2][2]/sum(conf[2]))))
    print("both modified prediction accuracy rate:" + str((conf[3][3]/sum(conf[3]))))
    print("\n")
    row_result = []
    row_result.append(conf[0][0]/sum(conf[0])) 
    row_result.append(conf[1][1]/sum(conf[1])) 
    row_result.append(conf[2][2]/sum(conf[2]))
    row_result.append(conf[3][3]/sum(conf[3]))

    row_result_bin = []
    row_result_bin.append(conf[0][0]/sum(conf[0])) 
    row_result_bin.append(1 - conf[1][0]/sum(conf[1])) 
    row_result_bin.append(1 - conf[2][0]/sum(conf[2])) 
    row_result_bin.append(1 - conf[3][0]/sum(conf[3])) 

    ##### Group Not G #####
    pred_class_notG = model_built.predict(x_test_notG)
    print('Group Not G Test')
    conf = confusion_matrix(y_test_notG, pred_class_notG)
    print("The confusion matrix for Test Case Group NOT G is showing below (class 0 result is meaningless):")
    print(conf)
    print("modified ECG prediction accuracy rate:" + str((conf[1][1]/sum(conf[1]))))
    print("modified ABP prediction accuracy rate:" + str((conf[2][2]/sum(conf[2]))))
    print("both modified prediction accuracy rate:" + str((conf[3][3]/sum(conf[3]))))
    print("\n")
    row_result.append(conf[1][1]/sum(conf[1])) 
    row_result.append(conf[2][2]/sum(conf[2]))
    row_result.append(conf[3][3]/sum(conf[3]))

    row_result_bin.append(1 - conf[1][0]/sum(conf[1])) 
    row_result_bin.append(1 - conf[2][0]/sum(conf[2])) 
    row_result_bin.append(1 - conf[3][0]/sum(conf[3])) 

    ##### Historical Test (combined) #####
    pred_class_hist = model_built.predict(x_test_hist)
    print('Historical not_Sync test (combined)')
    conf = confusion_matrix(y_test_hist, pred_class_hist)
    print("The confusion matrix is showing below:")
    print(conf)
    print("prediction accuracy rate for historical data test is: " + str((conf[4][4]/sum(conf[4]))))
    print("\n")
    row_result.append(conf[4][4]/sum(conf[4])) 
    row_result_bin.append(1 - conf[4][0]/sum(conf[4])) 

    ##### Historical Test (ECG replaced) #####
    x_test_hist_2 = []
    y_test_hist_2 = []
    y_test_hist_bin_2  = []

    gen_test_hist_2 = generate_data_instance(parameters)
    gen_test_hist_2.make_class_0_data(subject,rawdata_test_G) #this is only for not-empty
    gen_test_hist_2.make_class_1_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G') #this is only for not-empty
    gen_test_hist_2.make_class_2_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G') #this is only for not-empty
    gen_test_hist_2.make_class_3_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G') #this is only for not-empty
    gen_test_hist_2.make_hist_class_data_ECG(subject, rawdata_test_G, rawdata_test_historical) 
    # gen_test_hist_2.make_hist_class_data_ABP(subject, rawdata_test_G, rawdata_test_historical)
    # gen_test_hist_2.make_hist_class_data_notSync(subject, rawdata_test_historical) 

    x_test_hist_2 = gen_test_hist_2.data
    y_test_hist_bin_2 = keras.utils.to_categorical(gen_test_hist_2.labels) # one hot encode outputs
    y_test_hist_2 = gen_test_hist_2.labels

    pred_class_hist_2 = model_built.predict(x_test_hist_2)
    print('Historical not_Sync test (ECG Replaced)')
    conf = confusion_matrix(y_test_hist_2, pred_class_hist_2)
    print("The confusion matrix is showing below:")
    print(conf)
    print("prediction accuracy rate for hist ECG replacement :" + str((conf[4][4]/sum(conf[4]))))
    print("\n")
    row_result.append(conf[4][4]/sum(conf[4]))
    row_result_bin.append(1 - conf[4][0]/sum(conf[4]))

    ##### Historical Test (ABP replaced) #####
    x_test_hist_2 = []
    y_test_hist_2 = []
    y_test_hist_bin_2  = []

    gen_test_hist_2 = generate_data_instance(parameters)
    gen_test_hist_2.make_class_0_data(subject,rawdata_test_G) #this is only for not-empty
    gen_test_hist_2.make_class_1_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G') #this is only for not-empty
    gen_test_hist_2.make_class_2_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G') #this is only for not-empty
    gen_test_hist_2.make_class_3_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G') #this is only for not-empty
    gen_test_hist_2.make_hist_class_data_ABP(subject, rawdata_test_G, rawdata_test_historical)
    # gen_test_hist_2.make_hist_class_data_notSync(subject, rawdata_test_historical) 

    x_test_hist_2 = gen_test_hist_2.data
    y_test_hist_bin_2 = keras.utils.to_categorical(gen_test_hist_2.labels) # one hot encode outputs
    y_test_hist_2 = gen_test_hist_2.labels

    pred_class_hist_2 = model_built.predict(x_test_hist_2)
    print('Historical not_Sync test (ABP Replaced)')
    conf = confusion_matrix(y_test_hist_2, pred_class_hist_2)
    print("The confusion matrix is showing below:")
    print(conf)
    print("prediction accuracy rate for hist ABP replacement :" + str((conf[4][4]/sum(conf[4]))))
    print("\n")
    row_result.append(conf[4][4]/sum(conf[4]))
    row_result_bin.append(1 - conf[4][0]/sum(conf[4]))

    ##### Historical Test (Both replaced) #####
    x_test_hist_2 = []
    y_test_hist_2 = []
    y_test_hist_bin_2  = []

    gen_test_hist_2 = generate_data_instance(parameters)
    gen_test_hist_2.make_class_0_data(subject,rawdata_test_G) #this is only for not-empty
    gen_test_hist_2.make_class_1_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G') #this is only for not-empty
    gen_test_hist_2.make_class_2_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G') #this is only for not-empty
    gen_test_hist_2.make_class_3_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G') #this is only for not-empty
    gen_test_hist_2.make_hist_class_data_notSync(subject, rawdata_test_historical) 

    x_test_hist_2 = gen_test_hist_2.data
    y_test_hist_bin_2 = keras.utils.to_categorical(gen_test_hist_2.labels) # one hot encode outputs
    y_test_hist_2 = gen_test_hist_2.labels

    pred_class_hist_2 = model_built.predict(x_test_hist_2)
    print('Historical not_Sync test (Both Replaced)')
    conf = confusion_matrix(y_test_hist_2, pred_class_hist_2)
    print("The confusion matrix is showing below:")
    print(conf)
    print("prediction accuracy rate for both signals replacement :" + str((conf[4][4]/sum(conf[4]))))
    print("\n")
    row_result.append(conf[4][4]/sum(conf[4]))
    row_result_bin.append(1 - conf[4][0]/sum(conf[4]))

    ### Pred_class_hist_3 is a series of test 
    for duration_diff in parameters.async_duration_diff:

        # Create Group G historical test data III (not_sync hist data duration of different start picked)
        x_test_hist_3 = []
        y_test_hist_3 = []
        y_test_hist_bin_3 = []

        gen_test_hist_3 = generate_data_instance(parameters)
        gen_test_hist_3.make_class_0_data(subject,rawdata_test_G) #this is only for not-empty
        gen_test_hist_3.make_class_1_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G') #this is only for not-empty
        gen_test_hist_3.make_class_2_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G') #this is only for not-empty
        gen_test_hist_3.make_class_3_data(subject,rawdata_test,parameters.seg_altered_per_subject_test_G,'G') #this is only for not-empty
        gen_test_hist_3.make_hist_class_data_notSync_difsecs(subject, rawdata_test_historical, duration_diff) # for non-sync historical data
                
        x_test_hist_3 = gen_test_hist_3.data
        y_test_hist_bin_3 = keras.utils.to_categorical(gen_test_hist_3.labels) # one hot encode outputs
        y_test_hist_3 = gen_test_hist_3.labels

        ### Prediction part
        pred_class_hist_3 = model_built.predict(x_test_hist_3)
        print('Historical not_Sync test --- ' + str(duration_diff) + ' secs')
        conf = confusion_matrix(y_test_hist_3, pred_class_hist_3)
        print("The confusion matrix is showing below:")
        print(conf)
        print("prediction accuracy rate for both modified :" + str(duration_diff) + "_secs_case is: " + str((conf[4][4]/sum(conf[4]))))
        print("\n")
        row_result.append(conf[4][4]/sum(conf[4]))
        row_result_bin.append(1 - conf[4][0]/sum(conf[4]))

    if K.backend() == 'tensorflow':
        K.clear_session()

    writer.writerow(row_result)
    writer_bin.writerow(row_result_bin)
    print("Evaluation for subject " + subject + " --- END!")

writer_tacc.writerow(train_acc)
csv_output_tacc.close()
csv_output.close()
csv_output_binCase.close()
