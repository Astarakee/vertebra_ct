import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_poor_result(mean_dict, thr):
    new_dict = {}
    for key, val in mean_dict.items():
        if val<thr:
            new_dict[key] = val
            
    return new_dict
        

main_path = '/mnt/work/projects/2_Vertebra/PredictedMasks/3_Model3_Spine1K'
file_name1 = 'TrainOnSpine1K_PredOnTotalSeg_Summary_Results.pickle'
file_name2 = 'TrainOnSpine1K_PredOnVerSe_Summary_Results.pickle'
file_name3 = 'TrainOnSpine1K_PredOnVerSe_Summary_Results.pickle'


file_path1 = os.path.join(main_path,file_name1)
file_path2 = os.path.join(main_path,file_name2)
file_path3 = os.path.join(main_path,file_name3)

md_file1 = pd.read_pickle(file_path1)
md_file2 = pd.read_pickle(file_path2)
md_file3 = pd.read_pickle(file_path3)


mean1 = md_file1['subjectwise_mean_dice']
mean2 = md_file2['subjectwise_mean_dice']
mean3 = md_file3['subjectwise_mean_dice']


thr_error = 0.7
poor1 = get_poor_result(mean1,thr_error)
poor2 = get_poor_result(mean2,thr_error)
poor3 = get_poor_result(mean3,thr_error)



mean1_val = np.array(list(mean1.values()))
mean2_val = np.array(list(mean2.values()))
mean3_val = np.array(list(mean3.values()))

mean1_val = [x for x in mean1_val if x>0.7]
mean2_val = [x for x in mean2_val if x>0.7]
mean3_val = [x for x in mean3_val if x>0.7]

data_mean_val = [mean1_val, mean2_val, mean3_val]


fig = plt.figure(figsize =(15, 10)) 
ticks = ['Data1_validation(1200)', 'Data2_all(313)', 'Data3_all(313)']
#ticks = ['Data1_validation(300)', 'Data2_all(933)', 'Data3_all(313)']
green_diamond = dict(markerfacecolor='k', marker='D')
x_position = np.array(range(len(data_mean_val)))*.5-1
bpl = plt.boxplot(data_mean_val,
            notch=True, 
            vert=True,
            patch_artist=True,
            showfliers=green_diamond,
            positions=x_position,widths=0.15)

color = ['peru', 'wheat', 'skyblue']
for patch, color in zip(bpl['boxes'],color):
    patch.set_facecolor(color)
plt.grid()
plt.legend(prop={'weight':'bold'}, fontsize='large')
plt.xticks(x_position, ticks, fontsize=10)
plt.xticks(rotation=15)
plt.title('Model3 stats. - Train on Data2(933)', fontsize=12, fontweight='bold')
plt.xlabel('Datasets')
plt.ylabel('Dice score')



save_path_fig = '/home/mehdi/Desktop/VerTeb_Summary'
fig_name = 'Model3_Mean_Dice_NoPoor.png'
save_dir = os.path.join(save_path_fig, fig_name)
plt.savefig(save_dir)


def get_stats(my_list):
    metric_array = np.array(my_list)
    mean_val = np.mean(metric_array)
    std_val = np.std(metric_array)
    med_val = np.median(metric_array)
    return mean_val, std_val, med_val


mean_val1, std_val1, med_val1 = get_stats(mean1_val)
mean_val2, std_val2, med_val2 = get_stats(mean2_val)
mean_val3, std_val3, med_val3 = get_stats(mean3_val)

stats_dict = {}
data1_stats = {}
data2_stats = {}
data3_stats = {}
data1_stats['mean_subjecwise'] = mean_val1
data1_stats['std_subjectwise'] = std_val1
data1_stats['median_subjectwise'] = med_val1
data2_stats['mean_subjecwise'] = mean_val2
data2_stats['std_subjectwise'] = std_val2
data2_stats['median_subjectwise'] = med_val2
data3_stats['mean_subjecwise'] = mean_val3
data3_stats['std_subjectwise'] = std_val3
data3_stats['median_subjectwise'] = med_val3

stats_dict['Data1_validation(200)'] = data1_stats
stats_dict['Data2_all(313)'] = data2_stats
stats_dict['Data3_all(313)'] = data3_stats

pkl_file_name = './logs/Model3_Mean_dice_NoPoor_stats.pickle'
with open(pkl_file_name, 'wb') as handle:
    pickle.dump(stats_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)















