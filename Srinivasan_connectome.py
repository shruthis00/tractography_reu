# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:18:50 2021

@author: shrut
"""
import numpy as np
from scipy.ndimage.morphology import binary_dilation

from dipy.direction import peaks
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.segment.mask import median_otsu

from os.path import expanduser, join
home = expanduser('~')

subject = 'S00393'
dname = join(home, '.dipy', 'sample_files', subject)

flabel = join(dname, str(subject) + 'wm_resample.nii')
flabels = join(dname, str(subject) + '_label_resample.nii')
fdwi = join(dname, str(subject) + '_resample.nii')
fbval = join(dname, str(subject) + '_bvals.txt')
fbvec = join(dname, str(subject) + '_bvecs.txt')

data, affine, img = load_nifti(fdwi, return_img=True)
label = load_nifti_data(flabel)
labels = load_nifti_data(flabels)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

print("Data_loaded!")

data2 = data[:,:,:,8]
b0_mask, mask = median_otsu(data2, vol_idx = None, median_radius=4, numpass=2)
#create FA
from dipy.reconst.dti import TensorModel
tensor_model = TensorModel(gtab)
tensor_fit = tensor_model.fit(data, mask)
save_nifti(str(subject) + '_fa.nii', tensor_fit.fa, affine)

fa_right = tensor_fit.fa
fa_right[0:(tensor_fit.shape[0] // 2),:,:] = 0

fa_left = tensor_fit.fa
fa_left[(tensor_fit.shape[0] // 2):,:,:] = 0

print('FA output saved!')

import matplotlib.pyplot as plt
from dipy.core.histeq import histeq

sli = data.shape[2] // 2

plt.figure('Brain segmentation')
plt.subplot(1, 2, 1).set_axis_off()
plt.imshow(histeq(data[:, :, sli,0].astype('float')).T,
           cmap='gray', origin='lower')

wm = binary_dilation(label == 1)
csamodel = CsaOdfModel(gtab, 6)
csapeaks = peaks.peaks_from_model(model=csamodel,
                                  data=data,
                                  sphere=peaks.default_sphere,
                                  relative_peak_threshold=.8,
                                  min_separation_angle=45,
                                  mask=wm)

print('CSA done, mask loading')

import pandas as pd

df = pd.read_excel(join(dname, 'IITmean_RPI_index.xlsx'), sheet_name='Sheet1')

roi = []
roi.append(df['Index2'])
roi = np.asanyarray(roi)

roi_left = []
roi_left = np.concatenate([roi[:,0:7], roi[:,16:49]], axis = 1)

roi_right = []
roi_right = np.concatenate([roi[:,8:15], roi[:,50:83]], axis = 1)

#roi = [1032, 1033]
roi_mask = np.zeros(shape = label.shape, dtype = np.bool_)

def filter_streamlines (roi_real, fa_mask, labels, seed_input):
    
    roi_mask = np.zeros(shape = labels.shape, dtype=np.bool_)
    
    for i in np.arange(data.shape[0]):
        for j in np.arange(data.shape[1]):
            for k in np.arange(data.shape[2]):
                if labels[i][j][k] in roi_real:
                    roi_mask[i][j][k] = True
                  
                else:
                    roi_mask[i][j][k] = False
    print('ROI mask created')
    """              
    if seed_input.shape != label.shape:
        
        seed_mask = np.zeros(shape = labels.shape, dtype=np.bool_)
        for i in np.arange(data.shape[0]):
            for j in np.arange(data.shape[1]):
                for k in np.arange(data.shape[2]):
                    if labels[i][j][k] in seed_mask:
                        seed_mask[i][j][k] = True
                      
                    else:
                        seed_mask[i][j][k] = False
    else:
        seed_mask = seed_input
        """
                        
    affine = np.eye(4)
    seeds = utils.seeds_from_mask(seed_input, affine, density=1)

    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    FA_threshold = 0.05
    stopping_criterion=ThresholdStoppingCriterion(fa_mask, FA_threshold)
    #stopping_criterion=BinaryStoppingCriterion(wm == 1)
    
    streamline_generator = LocalTracking(csapeaks, stopping_criterion, seeds,
                                         affine=affine, step_size=0.5)
    streamlines = Streamlines(streamline_generator)
            
    roi_streamlines = utils.target(streamlines, affine, roi_mask)
    roi_streamlines = Streamlines(roi_streamlines)
    
    from dipy.viz import window, actor, colormap as cmap
    
    # Enables/disables interactive visualization
    interactive = True
    
    # Make display objects
    color = cmap.line_colors(roi_streamlines)
    roi_streamlines_actor = actor.line(roi_streamlines, color)
    ROI_actor = actor.contour_from_roi(roi_mask, color=(1., 1., 0.), opacity=0.1)
    
    r = window.Scene()
    
    r.add(roi_streamlines_actor)
    r.add(ROI_actor)
    
    # Save figures
    #window.record(r, n_frames=1, out_path='corpuscallosum_axial.png',
               #   size=(800, 800))
    if interactive:
        window.show(r)
    r.set_camera(position=[-1, 0, 0], focal_point=[0, 0, 0], view_up=[0, 0, 1])
    #window.record(r, n_frames=1, out_path='corpuscallosum_sagittal.png',
     #             size=(800, 800))
     
    return roi_streamlines
    return roi_real

sft = StatefulTractogram(roi_streamlines, img, Space.RASMM)
save_tractogram(sft, 'whole_brain' + str(subject) + '.trk', bbox_valid_check= False)
