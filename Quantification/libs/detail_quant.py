import numpy as np
import matplotlib.pyplot as plt


def get_subjectwise_metric(all_metric):
    label_1 = []
    label_2 = []
    label_3 = []
    label_4 = []
    label_5 = []
    label_6 = []
    label_7 = []
    label_8 = []
    label_9 = []
    label_10 = []
    label_11 = []
    label_12 = []
    label_13 = []
    label_14 = []
    label_15 = []
    label_16 = []
    label_17 = []
    label_18 = []
    label_19 = []
    label_20 = []
    label_21 = []
    label_22 = []
    label_23 = []
    label_24 = []
    for subject_vals in all_metric:
        key = subject_vals.keys()
        val = list(subject_vals.values())
        for key_label in enumerate(key):

            if key_label[1]==1:
                label_1.append(val[key_label[0]])
            elif key_label[1]==2:
                label_2.append(val[key_label[0]])
            elif key_label[1]==3:
                label_3.append(val[key_label[0]])
            elif key_label[1]==4:
                label_4.append(val[key_label[0]])
            elif key_label[1]==5:
                label_5.append(val[key_label[0]])
            elif key_label[1]==6:
                label_6.append(val[key_label[0]])
            elif key_label[1]==7:
                label_7.append(val[key_label[0]])
            elif key_label[1]==8:
                label_8.append(val[key_label[0]])
            elif key_label[1]==9:
                label_9.append(val[key_label[0]])
            elif key_label[1]==10:
                label_10.append(val[key_label[0]])
            elif key_label[1]==11:
                label_11.append(val[key_label[0]])
            elif key_label[1]==12:
                label_12.append(val[key_label[0]])
            elif key_label[1]==13:
                label_13.append(val[key_label[0]])
            elif key_label[1]==14:
                label_14.append(val[key_label[0]])
            elif key_label[1]==15:
                label_15.append(val[key_label[0]])
            elif key_label[1]==16:
                label_16.append(val[key_label[0]])
            elif key_label[1]==17:
                label_17.append(val[key_label[0]])
            elif key_label[1]==18:
                label_18.append(val[key_label[0]])
            elif key_label[1]==19:
                label_19.append(val[key_label[0]])
            elif key_label[1]==20:
                label_20.append(val[key_label[0]])
            elif key_label[1]==21:
                label_21.append(val[key_label[0]])
            elif key_label[1]==22:
                label_22.append(val[key_label[0]])
            elif key_label[1]==23:
                label_23.append(val[key_label[0]])
            elif key_label[1]==24:
                label_24.append(val[key_label[0]])               
            else:
                print('Errors! some label values are strange!')
    
    outdict = {}
    outdict['1'] = label_1
    outdict['2'] = label_2
    outdict['3'] = label_3
    outdict['4'] = label_4
    outdict['5'] = label_5
    outdict['6'] = label_6
    outdict['7'] = label_7
    outdict['8'] = label_8
    outdict['9'] = label_9
    outdict['10'] = label_10
    outdict['11'] = label_11
    outdict['12'] = label_12
    outdict['13'] = label_13
    outdict['14'] = label_14
    outdict['15'] = label_15
    outdict['16'] = label_16
    outdict['17'] = label_17
    outdict['18'] = label_18
    outdict['19'] = label_19
    outdict['20'] = label_20
    outdict['21'] = label_21
    outdict['22'] = label_22
    outdict['23'] = label_23
    outdict['24'] = label_24
    
    return outdict


def get_separate_metrics(my_dict):
    all_dice = []
    all_precision = []
    all_recall = []
    all_sbj_metric = list(my_dict.values())
    for item in all_sbj_metric:
        all_dice.append(item['Dice'])
        all_precision.append(item['Precision'])
        all_recall.append(item['Recall'])
        
    return  all_dice, all_precision, all_recall
        
        
  
def plot_metrics(label_val, TITLE, XLABEL, YLABEL):
    fig = plt.figure(figsize =(25, 12)) 
    ticks = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
                        'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',
                        'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']
    
    green_diamond = dict(markerfacecolor='k', marker='D')
    x_position = np.array(range(len(label_val)))*.5-1
    bpl = plt.boxplot(label_val,
                notch=True, 
                vert=True,
                patch_artist=True,
                showfliers=green_diamond,
                positions=x_position,widths=0.15)
    
    color = 'blue'
    for patch in bpl['boxes']:
        patch.set_facecolor(color)
    plt.grid()
    plt.legend(prop={'weight':'bold'}, fontsize='large')
    plt.xticks(x_position, ticks, fontsize=10)
    plt.xticks(rotation=15)
    plt.title(TITLE, fontsize=12, fontweight='bold')
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    
    return None
  
def remove_poorts(label_val, thr):
    label_NoPoor = []
    for x in label_val:
        temp = []
        for y in x:
            if y>=thr:
                temp.append(y)
        label_NoPoor.append(temp)
        
    return label_NoPoor   


def get_detail_stats(label_val):
    vertebra_metric = {}
    for xx in range(len(label_val)):
        
        verteb_values = np.array(label_val[xx])
        verteb_values = verteb_values[~np.isnan(verteb_values)] # remove the NaNs
        mean_val = np.mean(verteb_values)
        std_val = np.std(verteb_values)
        med_val = np.median(verteb_values)
        temp_dict = {}
        temp_dict['metric_mean'] = mean_val
        temp_dict['metric_std'] = std_val
        temp_dict['metric_median'] = med_val
        vertebra_metric[xx+1] = temp_dict
        
    return vertebra_metric