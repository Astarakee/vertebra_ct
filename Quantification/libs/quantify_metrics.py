import numpy as np
from math import isnan
from scipy.ndimage import morphology
import copy


def dice(P, G):
    psum = np.sum(P.flatten())
    gsum = np.sum(G.flatten())
    pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
    score = (2 * pgsum) / (psum + gsum)
    return score


def confusion_matrix(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    fn = np.sum(np.multiply(np.invert(P.flatten()), G.flatten()))
    tn = np.sum(np.multiply(np.invert(P.flatten()), np.invert(G.flatten())))
    return tp, fp, tn, fn


def tpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fn = np.sum(np.multiply(np.invert(P.flatten()), G.flatten()))
    return tp / (tp + fn)


def fpr(P, G):
    tn = np.sum(np.multiply(np.invert(P.flatten()), np.invert(G.flatten())))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    return fp / (fp + tn)


def precision(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    return tp / (tp + fp)


def recall(P, G):
    return tpr(P, G)


def specificty(P, G):
    return 1-fpr(P, G)


def surfd(input1, input2, sampling=1, connectivity=1):
    '''
    https://mlnotebook.github.io/post/surface-distance-function/
    Surface Distance Function
    '''
    
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))
    

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    eroded1 = morphology.binary_erosion(input_1, conn)
    eroded2 = morphology.binary_erosion(input_2, conn)
    
    S = (input1.astype(np.float32)-eroded1.astype(np.float32)).astype(np.bool)
    Sprime = (input2.astype(np.float32)-eroded2.astype(np.float32)).astype(np.bool)

    
    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)
    
    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])
    temp = copy.deepcopy(sds)
    percent95 = np.percentile(temp, 95)
    temp[temp>percent95] = 0
       
    hausdorff_95 = temp.max()
    mean_sd = sds.mean()
    return hausdorff_95, mean_sd

#msd = surface_distance.mean() #Mean Surface Distance (MSD)
#rms = np.sqrt((surface_distance**2).mean()) #Residual Mean Square Distance (RMS) 
#hd  = surface_distance.max() #Hausdorff Distance (HD)

    
    
def evaluate_results_volumetric(pred, truth, thr):

    pred[pred>thr] = 1
    pred[pred<=thr] = 0
    truth[truth>thr] = 1
    truth[truth<=thr] = 0
    
    pred = pred.astype('bool')
    truth = truth.astype('bool')
    
    score_dice = dice(pred, truth)
    score_precision = precision(pred, truth)
    score_recall = recall(pred, truth)
    score_fpr = fpr(pred, truth)
    score_spc = specificty(pred, truth)
    
    return score_dice, score_recall, score_spc, score_precision, score_fpr 
    
def calculate_distance_volumetric(model_predict, seg_mask, thr=0.5, resolution=[1,1,1]):
    
       
    pred_mask = model_predict
    pred_mask[pred_mask<thr]=0
    pred_mask[pred_mask>=thr]=1
    
    label_mask = seg_mask
    
    mask_sum_intensity = label_mask.sum()
    if mask_sum_intensity == 0:
        hausdorff_distance = 0
        surf_distance = 0
    else:
        hausdorff, mean_sd = surfd(pred_mask, label_mask, sampling= resolution, connectivity=1)
        hausdorff_distance = hausdorff
        surf_distance = mean_sd
        
    return hausdorff_distance, surf_distance       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    