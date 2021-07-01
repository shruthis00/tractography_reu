# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 17:55:24 2021

@author: shrut
"""
import dipy
from dipy.io.image import load_nifti, save_nifti
import numpy as np
from os.path import join, expanduser

home = expanduser('~')

dname = join(home, '.dipy', 'sample_files', 'S00393')
f_nii = join(dname, 'S00393wm.nii')

data, affine, vox_size = load_nifti(f_nii, return_voxsize=True)
print(data.shape)

from dipy.align.reslice import reslice
new_vox_size = (2.,2.,2.)
data2, affine2 = reslice(data, affine, vox_size, new_vox_size)
print(data2.shape)


data2_np = np.array(data2)
#np.savetxt('wm_mask', data_np, delimiter=',')

data2_np = (data2_np >= 0.20)
data2_np = data2_np.astype('uint8')

save_nifti('S00393wm_resample', data2_np, affine)


