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
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.io.image import load_nifti, load_nifti_data
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram

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

import matplotlib.pyplot as plt
from dipy.core.histeq import histeq

sli = data.shape[2] // 2

plt.figure('Brain segmentation')
plt.subplot(1, 2, 1).set_axis_off()
plt.imshow(histeq(data[:, :,sli].astype('float')).T,
           cmap='gray', origin='lower')


wm = binary_dilation(label == 1)
csamodel = CsaOdfModel(gtab, 6)
csapeaks = peaks.peaks_from_model(model=csamodel,
                                  data=data,
                                  sphere=peaks.default_sphere,
                                  relative_peak_threshold=.8,
                                  min_separation_angle=45,
                                  mask=wm)

hipp = np.zeros(shape = label.shape, dtype = np.bool_)

"""
import pandas as pd

df = pd.read_excel (join(dname, 'IITmean_RPI_index.xlsx'), sheet_name='Sheet1')
roi = []
roi.append(df['Index2'])
roi = np.asanyarray(roi)
"""
roi = [17,18,53,54]

#ROI gets: Hippocampus, amygdala, inferior temporal and middle temporal, superior temporal, also accumbens
for i in np.arange(103):
    for j in np.arange(103):
        for k in np.arange(50):
            if labels[i][j][k] in roi:
                hipp[i][j][k] = True
                
            else:
                hipp[i][j][k] = False
                
affine = np.eye(4)
seeds = utils.seeds_from_mask(wm, affine, density=1)
stopping_criterion = BinaryStoppingCriterion(wm)
                
streamline_generator = LocalTracking(csapeaks, stopping_criterion, seeds,
                                     affine=affine, step_size=0.5)
streamlines = Streamlines(streamline_generator)
        
hipp_streamlines = utils.target(streamlines, affine, hipp)
hipp_streamlines = Streamlines(hipp_streamlines)

other_streamlines = utils.target(streamlines, affine, hipp,
                                 include=False)
other_streamlines = Streamlines(other_streamlines)
assert len(other_streamlines) + len(hipp_streamlines) == len(streamlines)

from dipy.viz import window, actor, colormap as cmap

# Enables/disables interactive visualization
interactive = True

# Make display objects
color = cmap.line_colors(streamlines)
hipp_streamlines_actor = actor.line(streamlines,
                                  cmap.line_colors(streamlines))
hipp_ROI_actor = actor.contour_from_roi(hipp, color=(1., 1., 0.),
                                      opacity=0.5)

# Add display objects to canvas
r = window.Scene()

r.add(hipp_streamlines_actor)
r.add(hipp_ROI_actor)

# Save figures
#window.record(r, n_frames=1, out_path='corpuscallosum_axial.png',
           #   size=(800, 800))
if interactive:
    window.show(r)
r.set_camera(position=[-1, 0, 0], focal_point=[0, 0, 0], view_up=[0, 0, 1])
#window.record(r, n_frames=1, out_path='corpuscallosum_sagittal.png',
 #             size=(800, 800))


#sft = StatefulTractogram(streamlines, img, Space.RASMM)
#save_tractogram(sft, 'whole_brain' + str(subject) + '.trk', bbox_valid_check= False)