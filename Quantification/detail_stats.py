import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from libs.detail_quant import get_subjectwise_metric, get_separate_metrics
from libs.detail_quant import plot_metrics, remove_poorts, get_detail_stats

    
    
main_path = '/mnt/work/projects/2_Vertebra/PredictedMasks/3_Model3_Spine1K'
file_name1 = 'TrainOnSpine1K_PredOnVerSe_Summary_Results.pickle'
file_name2 = 'TrainOnSpine1K_PredOnVerSe_Summary_Results.pickle'
file_name3 = 'TrainOnSpine1K_PredOnVerSe_Summary_Results.pickle'


file_path1 = os.path.join(main_path,file_name1)
file_path2 = os.path.join(main_path,file_name2)
file_path3 = os.path.join(main_path,file_name3)

md_file1 = pd.read_pickle(file_path1)
md_file2 = pd.read_pickle(file_path2)
md_file3 = pd.read_pickle(file_path3)


sbj_metrics1 = md_file1['subject_metrics']
sbj_metrics2 = md_file2['subject_metrics']
sbj_metrics3 = md_file3['subject_metrics']


all_dice1, all_precision1, all_recall1 = get_separate_metrics(sbj_metrics1)
all_dice2, all_precision2, all_recall2 = get_separate_metrics(sbj_metrics2)
all_dice3, all_precision3, all_recall3 = get_separate_metrics(sbj_metrics3)

sbj_dice1 = get_subjectwise_metric(all_dice1)
sbj_dice2 = get_subjectwise_metric(all_dice2)
sbj_dice3 = get_subjectwise_metric(all_dice3)

sbj_precision1 = get_subjectwise_metric(all_precision1)
sbj_precision2 = get_subjectwise_metric(all_precision2)
sbj_precision3 = get_subjectwise_metric(all_precision3)

sbj_recall1 = get_subjectwise_metric(all_recall1)
sbj_recall2 = get_subjectwise_metric(all_recall2)
sbj_recall3 = get_subjectwise_metric(all_recall3)


label_num = list(sbj_dice2.keys())

dice_metric_all = list(sbj_dice2.values())
dice_metric_NoPoor =  remove_poorts(dice_metric_all, .7)

precision_metric_all = list(sbj_precision2.values())
precision_metric_NoPoor =  remove_poorts(precision_metric_all, .7)

recall_metric_all = list(sbj_recall2.values())
recall_metric_NoPoor =  remove_poorts(recall_metric_all, .7)


plot_metrics(dice_metric_NoPoor, 'Model3-train(933)-PredOnVerSe', 'Vertebra', 'Dice Score')


save_path_fig = '/home/mehdi/Desktop/VerTeb_Summary'
fig_name = 'Model3_PredOnVerSe_Detailed_Dice_NoPoor.png'
save_dir = os.path.join(save_path_fig, fig_name)
plt.savefig(save_dir)


detail_metrics_dice = get_detail_stats(dice_metric_NoPoor)
detail_metrics_precision = get_detail_stats(precision_metric_NoPoor)
detail_metrics_recall = get_detail_stats(recall_metric_NoPoor)

summary_dict = {}
summary_dict['Dice_details'] = detail_metrics_dice
summary_dict['Precision_details'] = detail_metrics_precision
summary_dict['Recall_details'] = detail_metrics_recall

pkl_file_name = './logs/Model3_PredOnVerSe_Details_Metrics_NoPoor.pickle'
with open(pkl_file_name, 'wb') as handle:
    pickle.dump(summary_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
