"""
Created on Tue Jul  6 12:18:50 2021

@author: shrut
"""
from fury import window, actor, colormap as cmap

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
from dipy.io.streamline import save_trk
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.segment.mask import median_otsu

from os.path import expanduser, join

home = expanduser('~')

subject = 'subj'
dname = join(home, '.dipy', 'sample_files', subject)

#you NEED a white matter mask, constructed by dti function in dipy. I have a function to do this already, and it is in the github under 
            #T1_class.py, all code there
flabel = join(dname, str(subject) + '_wm.nii')
            
#you don't need to resample the label files or the main dwi if you don't need to, just make sure everything is the same size.
flabels = join(dname, str(subject) + '_label_resampled.nii')
fdwi = join(dname, str(subject) + '_resample.nii')
fbval = join(dname, str(subject) + '_bvals.txt')
fbvec = join(dname, str(subject) + '_bvecs.txt')

data, affine, img = load_nifti(fdwi, return_img=True)
label = load_nifti_data(flabel)
labels = load_nifti_data(flabels)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

print("Data_loaded!")

b0_mask, mask = median_otsu(data, vol_idx = np.arange(8, 12), median_radius=4, numpass=2)

#create FA
from dipy.reconst.dti import TensorModel
tensor_model = TensorModel(gtab)
tensor_fit = tensor_model.fit(data, mask)
save_nifti(str(subject) + '_fa.nii.nii', tensor_fit.fa, affine)

#fa_left = np.copy(tensor_fit.fa)
#fa_left[0:(tensor_fit.shape[0] // 2),:,:] = 0

#fa_right = np.copy(tensor_fit.fa)
#fa_right[(tensor_fit.shape[0] // 2):,:,:] = 0

print('FA output generated and saved!')

import matplotlib.pyplot as plt
from dipy.core.histeq import histeq

wm = binary_dilation(label == 1)

sli = data.shape[2] // 2

plt.figure('Brain segmentation')
plt.subplot(1, 2, 1).set_axis_off()
plt.imshow(histeq(data[:, :, sli, 10].astype('float')).T,
           cmap='gray', origin='lower')

fig = plt.figure()
a = fig.add_subplot(1, 3, 3)
plt.imshow(wm[:,:,sli], cmap = 'gray')
a.axis('off')

csamodel = CsaOdfModel(gtab, 6)
csapeaks = peaks.peaks_from_model(model=csamodel,
                                  data=data,
                                  sphere=peaks.default_sphere,
                                  relative_peak_threshold=.8,
                                  min_separation_angle=45,
                                  mask=wm)
interactive = True

scene = window.Scene()
scene.add(actor.peak_slicer(csapeaks.peak_dirs,
                          csapeaks.peak_values,
                          colors=None))

if interactive:
    window.show(scene)
        
FA_map = csapeaks.gfa
print('CSA done, mask loading')
print('GFA output generated')

import pandas as pd

df = pd.read_excel(join(home, '.dipy', 'sample_files', 'IITmean_RPI_index.xlsx'), sheet_name='Sheet1')

roi_whole = []
            
#!!!! Make sure you change the case on this if need be
roi_whole.append(df['Index2'])
roi = np.asanyarray(roi_whole, dtype=object)

roi_left = []
roi_left = np.concatenate([roi[:,0:7], roi[:,16:49]], axis = 1)

roi_right = []
roi_right = np.concatenate([roi[:,8:15], roi[:,50:83]], axis = 1)

#ok I got it, h_r is the hippocampus right, h_l is hipp left, this is the seed
#a_r and a_l is amygdala left and right, this is the target
#seed to target tractography
h_r = (labels == 53)
h_l = (labels == 17)

a_r = [54]
a_l = [18]

def filter_streamlines (roi_real, fa_mask, labels, seed_input):
    
    #roi mask takes the data for the defined roi and creates a boolean array,
    #this is why I started with an array of 0, so I coule populate with true false to select for the region of interest within the brain, using the LABEL file.
    roi_mask = np.zeros(shape = labels.shape, dtype=np.bool_)
    
    for i in np.arange(data.shape[0]):
        for j in np.arange(data.shape[1]):
            for k in np.arange(data.shape[2]):
                if labels[i][j][k] in roi_real:
                    roi_mask[i][j][k] = True
                  
                else:
                    roi_mask[i][j][k] = False
                    
    print(np.mean((roi_mask.astype(int))))
    print('ROI mask created')

    #I had already defined the seed array above in the lines "h_r = (labels == 53)", making it a boolean array
    #however, if the seed array was not the same as the white matter mask I made earlier (meaning I resampled it wrong or something)
    #then I redefine it here based on the size of the white matter mask, using the labeling file.
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
    FA_threshold = 0.02
    stopping_criterion=ThresholdStoppingCriterion(fa_mask, FA_threshold)
    
    streamline_generator = LocalTracking(csapeaks, stopping_criterion, seeds,
                                         affine=affine, step_size=0.5)
    streamlines = Streamlines(streamline_generator)
            
    roi_streamlines = utils.target(streamlines, affine, roi_mask)
    roi_streamlines = Streamlines(roi_streamlines)
    
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

#whole_brain = filter_streamlines(roi, FA_map, labels, wm)
#What i have commented above ^^^ is to generate a streamline map of ALL of the streamlines in the brain

#What is below is generating the region of interest "bundles" that are self defined
#They seed from the hippocampus and target the amygdala, this is why [seed_input is h_l and h_r] and [roi_real is a_l and a_r]
hipp_to_amyg_l = filter_streamlines(a_l, FA_map, labels, h_l)
hipp_to_amyg_r = filter_streamlines(a_r, FA_map, labels, h_r)
"""
#Saving tractograms for left and right separately
sft = StatefulTractogram(hipp_to_amyg_l, img, Space.RASMM)
save_trk(sft, 'ha_l_' + str(subject) + '.trk', bbox_valid_check= False)

sft = StatefulTractogram(hipp_to_amyg_r, img, Space.RASMM)
save_trk(sft, 'ha_r_' + str(subject) + '.trk', bbox_valid_check= False)
"""
