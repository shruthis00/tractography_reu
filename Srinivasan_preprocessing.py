# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 01:00:40 2021

@author: shrut

Handles median otsu mask and resampling. Essentially pre-processing before
Freesurfer input
"""
from os.path import join
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu
import matplotlib.pyplot as plt
from dipy.core.histeq import histeq
from dipy.align.reslice import reslice

from os.path import expanduser
home = expanduser('~')

subject = 'S00490'
dname = join(home, '.dipy', 'sample_files', str(subject))
dname2 = join(home, '.spyder-py3')

#Loading sample data
fdwi = join(dname, 'bia6_00490_003.nii')
fbvec = join(dname, str(subject) + '_bvecs.txt')
fbval = join(dname, str(subject) + '_bvals.txt')

data, affine, img, vox_size = load_nifti(fdwi, return_img = True, return_voxsize=True)
data = np.squeeze(data[:,:,:])
print(data.shape)

def resample_nifti(data, affine, img, vox_size, new_vox_size, new_fname):
    
    print(data.shape)
    data2, affine2 = reslice(data, affine, vox_size, new_vox_size)
    print(data2.shape)
    
    sli = data.shape[2] // 2
    
    plt.figure('Normal versus ResampledAxial')
    plt.subplot(1, 2, 1).set_axis_off()
    plt.imshow(histeq(data[:, :,sli].astype('float')).T,
               cmap='gray', origin='lower')

    sli = data2.shape[2] // 2

    plt.subplot(1, 2, 2).set_axis_off()
    plt.imshow(histeq(data2[:, :, sli].astype('float')).T,
               cmap='gray', origin='lower')
    
    if new_fname:
        save_nifti(new_fname, data2, affine)
        
    return data2

data = resample_nifti(data, affine, img, vox_size, (2.,2.,2.), new_fname = None)

#Now median otsu
b0_mask, mask = median_otsu(data, median_radius=4, numpass=2)
b0_mask_crop, mask_crop = median_otsu(data, median_radius=4, numpass=4, autocrop=True)

save_nifti('bia_S00490_003_mask_resample.nii.gz', b0_mask.astype(np.float32), affine)
#save_nifti(fname + '_mask.nii.gz', b0_mask.astype(np.float32), affine)

sli = data.shape[2] // 2

plt.figure('Median Otsu stuff')
plt.subplot(1, 2, 1).set_axis_off()
plt.imshow(histeq(data[:, :, sli].astype('float')).T,
           cmap='gray', origin='lower')

plt.subplot(1, 2, 2).set_axis_off()
plt.imshow(histeq(b0_mask[:, :, sli].astype('float')).T,
           cmap='gray', origin='lower')

#plt.savefig('median_otsu.png')"""
mask = mask.astype('uint8')

print(b0_mask_crop.shape)
save_nifti("S00490_t1", b0_mask, affine)