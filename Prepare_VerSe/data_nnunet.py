from paths_dirs_stuff import path_contents, path_contents_pattern, natural_sort_key
import os
from copy import deepcopy
from itk_stuff import read_nifti, reorient_itk
from writer import write_nifti_from_val
from tqdm import tqdm

img_path1 = '/mnt/work/data/VerSe/Orig/01_training' 
img_path2 = '/mnt/work/data/VerSe/Orig/02_validation' 
img_path3 = '/mnt/work/data/VerSe/Orig/03_test' 

write_path_img = '/mnt/work/data/VerSe/Orig/all_img'
write_path_mask = '/mnt/work/data/VerSe/Orig/all_label'



def get_nifti_files(main_path):
    all_nifti = []
    for root, dirs, files in os.walk(main_path):
        for xx in files:
            if '.nii.gz' in xx:
                abs_path = os.path.join(root, xx)
                all_nifti.append(abs_path)
    
    return all_nifti


all_nifti1 = get_nifti_files(img_path1)
all_nifti2 = get_nifti_files(img_path2)
all_nifti3 = get_nifti_files(img_path3)

all_nifti_paths = all_nifti1+all_nifti2+all_nifti3
all_nifti_img = deepcopy(all_nifti_paths)
all_nifti_mask = [x for x in all_nifti_paths if '_seg' in x]

for xx in all_nifti_paths:
    for yy in all_nifti_mask:
        if xx == yy:
            all_nifti_img.remove(xx)
            
sanity_check = deepcopy(all_nifti_img)
for xx in all_nifti_img:
    xx1 = xx.split('/')[-1].split('.nii')[0]
    for yy in all_nifti_mask:
        yy1 = yy.split('/')[-1].split('_seg')[0]
        if xx1 == yy1:
            sanity_check.remove(xx)
            
            
for ix in sanity_check:
    print(ix)
    all_nifti_img.remove(ix)

         
all_nifti_img.sort(key=natural_sort_key)
all_nifti_mask.sort(key=natural_sort_key)

n_subject_img = len(all_nifti_img)
n_subject_mask = len(all_nifti_mask)


mismatch_file_size = []
if n_subject_img == n_subject_mask:
    for idx in tqdm(range(n_subject_img)):
        
        subject_abs_img = all_nifti_img[idx]
        subject_abs_mask = all_nifti_mask[idx]
        
        img_file_full_name = subject_abs_img.split('/')[-1].split('.nii.gz')[0]
        mask_file_fulle_name = subject_abs_mask.split('/')[-1].split('_seg')[0]
        
        if img_file_full_name == mask_file_fulle_name:
            
            img_array, img_itk, img_size, img_spacing, img_origin, img_direction = read_nifti(subject_abs_img)
            mask_array, mask_itk, mask_size, mask_spacing, mask_origin, mask_direction = read_nifti(subject_abs_mask)
            
            reorient_array, reoriented, reoriented_spacing, reoriented_origin, reoriented_direction = reorient_itk(img_itk)
            reorient_mask, reoriented_m, reoriented_spacing_m, reoriented_origin_m, reoriented_direction_m = reorient_itk(mask_itk)
            
            reorient_mask = reorient_mask.astype('uint8')
            reorient_array = reorient_array.astype('float32')
            
            nn_unet_filesname = img_file_full_name+'_0000.nii.gz'
            nn_unet_maskname = img_file_full_name+'.nii.gz'
            abs_name_volume = os.path.join(write_path_img, nn_unet_filesname)
            abs_name_label = os.path.join(write_path_mask, nn_unet_maskname)
            write_nifti_from_val(reorient_array, reoriented_origin_m, reoriented_spacing_m, reoriented_direction_m, abs_name_volume)
            write_nifti_from_val(reorient_mask, reoriented_origin_m, reoriented_spacing_m, reoriented_direction_m, abs_name_label)
            
            if img_size == mask_size:
                pass
            else:
                mismatch_file_size.append(subject_abs_img)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        
        
        