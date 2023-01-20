import os
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from libs.itk_stuff import read_nifti
from libs.paths_dirs_stuff import path_contents_pattern
from libs.quantify_metrics import evaluate_results_volumetric


def mean_subject_dice(vertebra_dice):
    dice_val = []
    for keys, vals in vertebra_dice.items():
        dice_val.append(vals)
        
    mean_dice = sum(dice_val)/len(dice_val)
    return mean_dice
        
def del_values(uniqe_labels, value):
    idx = np.where(uniqe_labels==value)
    idx = np.array(idx)
    
    if idx.shape[-1]!=0:
        idx_ = idx[0][0]
        out_labels = np.delete(uniqe_labels, idx_)
    else:
        out_labels = uniqe_labels
    return out_labels
    
    
    
    
label_dir = '/mnt/work/projects/2_Vertebra/PredictedMasks/3_Model3_Spine1K/2_Test_VerSe/labels'
pred_dir = '/mnt/work/projects/2_Vertebra/PredictedMasks/3_Model3_Spine1K/2_Test_VerSe/predictions'


label_files = path_contents_pattern(label_dir, '.nii.gz')
pred_files = path_contents_pattern(pred_dir, '.nii.gz')


mismatched_files = []
subject_metrics = {}
label_frequency = []
subjectwise_mean_dice = {}
if len(label_files) == len(label_files):
    for ix in tqdm(range(len(label_files))):

        label_subject = label_files[ix]
        pred_subject = pred_files[ix]
        
        if label_subject == pred_subject:
            
            label_abs_dir = os.path.join(label_dir,label_subject) 
            pred_abs_dir = os.path.join(pred_dir,pred_subject) 
            
            label_array, _, label_size, label_spacing, _, _ = read_nifti(label_abs_dir)
            pred_array, _, pred_size, pred_spacing, _, _ = read_nifti(pred_abs_dir)
            
            
            uniqe_labels = np.unique(label_array)
            my_tmp_label_uni = deepcopy(uniqe_labels)
            my_tmp_label_uni = my_tmp_label_uni.astype('int32')
            label_frequency.append(list(my_tmp_label_uni))
            
            uniqe_labels_ = del_values(uniqe_labels, 25) # remove inconsistency of VerSE
            uniqe_labels_ = del_values(uniqe_labels_, 28)
            vertebra_dice = {}
            vertebra_precision = {}
            vertebra_recall = {}
            temp_dict = {}
            for label_val in uniqe_labels_:
                if label_val == 0:
                    pass
                else:
                    
                    temp_label = deepcopy(label_array)
                    temp_pred = deepcopy(pred_array)
                    
                    temp_label[temp_label!=label_val] = 0
                    temp_pred[temp_pred!=label_val] = 0
                    
                    score_dice, score_recall, score_spc, score_precision, score_fpr = evaluate_results_volumetric(temp_pred, temp_label, 0.5)
                    label_val_int32 = label_val.astype('int32')
                    vertebra_dice[label_val_int32] = score_dice
                    vertebra_recall[label_val_int32] = score_recall
                    vertebra_precision[label_val_int32] = score_precision
            temp_dict['Dice'] = vertebra_dice
            temp_dict['Precision'] = vertebra_precision
            temp_dict['Recall'] = vertebra_recall
            subject_metrics[label_subject] = temp_dict
            subject_avg_dice = mean_subject_dice(vertebra_dice)
            subjectwise_mean_dice[label_subject] = subject_avg_dice
            print('\n')
            print('Mean dice of case {} is {}'.format(ix, subject_avg_dice))
            print('\n')
            
        else:
            mismatched_files.append(label_subject)

        

params = {}  
params['predict_dir'] = pred_dir
params['grand_truth_dir'] = label_dir
params['subjectwise_mean_dice'] = subjectwise_mean_dice
params['label_frequency'] = label_frequency
params['subject_metrics'] = subject_metrics
params['mismatched_files'] = mismatched_files

pkl_file_name = 'TrainOnSpine1K_PredOnVerSe_Summary_Results.pickle'
with open(pkl_file_name, 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)


