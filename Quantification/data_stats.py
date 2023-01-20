import os
import pickle
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


def read_pickle(pkl_file_name):
    with open(pkl_file_name, 'rb') as handle:
        pickle_dict = pickle.load(handle)
        
    return pickle_dict
    
    
# loading pickle summary file
file_path = '/mnt/work/projects/2_Vertebra/PredictedMasks/2_Model2_MixDataSets'
pickle_name = 'Train_VAL_CombinedTotaSegSpine1K_Summary_Results.pickle'
pkl_abs_path = os.path.join(file_path, pickle_name)

pickle_dict = read_pickle(pkl_abs_path)

# Taking the label frequency item
label_frq = pickle_dict['label_frequency']
label_frq_list = [item for sublist in label_frq for item in sublist]
label_frq_np = np.asarray(label_frq_list)

# label distribution
unique_label, counts_label = np.unique(label_frq_np, return_counts=True)


# creating templates for full vertebra 
tempelate_label = np.zeros((25), dtype = 'int32')
template_count =  np.zeros((25),dtype = 'int32')

label_list = ['Background', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
                    'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',
                    'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5'] 
# adapting the template with the data distribution
for ix, ix_val in enumerate(unique_label):
    tempelate_label[ix_val] = ix_val
    template_count[ix_val] = counts_label[ix]
        

# ploting stuffs
colors = cm.seismic(template_count / float(max(template_count)))
fig, ax = plt.subplots()
plot = ax.scatter(template_count, template_count,
                  c=template_count, cmap='seismic')
fig.set_size_inches(25, 12)
plt.cla()
ax.bar(range(len(tempelate_label)), template_count, color=colors)
for i, v in enumerate(template_count):
    ax.text(i, v, str(v), horizontalalignment='center',
            size = 14, color='black')
  
ax.set_xticks(list(range(25)))
ax.set_xticklabels(label_list, rotation=-45, fontsize = 12)
ax.set_xlabel('Vertebra Label', fontsize = 16)
ax.set_ylabel('Frequency', fontsize = 16)
ax.set_title('Vertebra Label Histogram', fontsize = 16)

save_path_fig = '/home/mehdi/Desktop/VerTeb_Summary'
fig_name = 'VerSe_label_distribution.png'
save_dir = os.path.join(save_path_fig, fig_name)
#plt.savefig(save_dir)