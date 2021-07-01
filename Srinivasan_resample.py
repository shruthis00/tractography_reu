# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:46:40 2021

@author: shrut
"""
import dipy
from dipy.io.image import load_nifti, save_nifti


from os.path import expanduser, join
home = expanduser('~')

dname = join(home, '.dipy', 'sample_files', 'S00393')
f_nii = join(dname, 'S00393gm.nii')

data, affine, vox_size = load_nifti(f_nii, return_voxsize=True)
print(data.shape)

from dipy.align.reslice import reslice
new_vox_size = (2.,2.,2.)
data2, affine2 = reslice(data, affine, vox_size, new_vox_size)
print(data2.shape)

save_nifti('S00393_gm_resample', data2, affine)
"""
import matplotlib.pyplot as plt
from dipy.core.histeq import histeq

sli = data2.shape[2] // 2
plt.figure('Brain segmentation with resample')
plt.subplot(1, 2, 1).set_axis_off()
plt.imshow(histeq(data2[:, :, sli,8].astype('float')).T,
           cmap='gray', origin='lower')

sli = data.shape[2] // 2
plt.figure('Brain segmentation')
plt.subplot(1, 2, 2).set_axis_off()
plt.imshow(histeq(data[:, :, sli,8].astype('float')).T,
           cmap='gray', origin='lower')
"""
