# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 23:29:36 2021

@author: shrut
"""

import dipy
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
import numpy as np

label_fname = get_fnames('stanford_labels')
labels = load_nifti_data(label_fname)

from os.path import expanduser, join
home = expanduser('~')

dname = join(home, '.dipy', 'sample_files', 'S00393')

flabel = join(dname, 'S00393_label_resample.nii')
#flabel = join(dname, 'S00393_label_wm_resample.nii')
fdwi = join(dname, 'S00393_resample.nii')
fbval = join(dname, 'S00393_bvals.txt')
fbvec = join(dname, 'S00393_bvecs.txt')

data, affine, img, vox_size = load_nifti(fdwi, return_img=True, return_voxsize=True)
label = load_nifti_data(flabel, as_ndarray=True)

bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

"""
from dipy.segment.mask import crop
Crop just in case
data = crop(data, (22,5,0, 0), (224,250,124, 16))
label = crop(label, (22,5,0), (224,250,124))
"""
interactive = True
mid_temp = ((label == 1015) | (label == 2015))

dshape = data.shape[:-1]
xa, xb, ya, yb, za, zb = [15, 42, 10, 65, 20, 45]
data_small = data[xa:xb, ya:yb, za:zb]
selectionmask = np.zeros(dshape, 'bool')
selectionmask[xa:xb, ya:yb, za:zb] = True

from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model

csa_model = CsaOdfModel(gtab, sh_order=6)
csa_peaks = peaks_from_model(csa_model, data, default_sphere,
                             relative_peak_threshold=.6,
                             min_separation_angle=45,
                             mask=selectionmask)

# Stopping Criterion
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, 0.25)

import matplotlib.pyplot as plt

sli = csa_peaks.gfa.shape[2] // 2
plt.figure('GFA')
plt.subplot(1, 2, 1).set_axis_off()
plt.imshow(csa_peaks.gfa[:, :, sli].T, cmap='gray', origin='lower')

plt.subplot(1, 2, 2).set_axis_off()
if interactive:
    plt.imshow((csa_peaks.gfa[:, :, 28] > 0.25).T, cmap='gray', origin='lower')
    
from dipy.reconst.csdeconv import auto_response_ssst
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response)
csd_fit = csd_model.fit(data_small)
csd_fit_shm = np.lib.pad(csd_fit.shm_coeff, ((xa, dshape[0]-xb),
                                             (ya, dshape[1]-yb),
                                             (za, dshape[2]-zb),
                                             (0, 0)), 'constant')

from dipy.direction import ProbabilisticDirectionGetter

prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit_shm,
                                                    max_angle=30.,
                                                    sphere=default_sphere)
#Seeding
from dipy.tracking import utils

mask = np.zeros(data.shape[:-1], 'bool')
rad = 3
mask[26-rad:26+rad, 29-rad:29+rad, 31-rad:31+rad] = True
seeds = utils.seeds_from_mask(mask, affine, density=[4, 4, 4])

#Tracking
from dipy.tracking.local_tracking import LocalTracking

streamlines_generator = LocalTracking(prob_dg, stopping_criterion, seeds,
                                      affine, step_size=.5)

# Compute streamlines.
from dipy.tracking.streamline import Streamlines
streamlines = Streamlines(streamlines_generator)

#Mask for lateral geniculate nucleus
mask_lgn = np.zeros(data.shape[:-1], 'bool')
rad = 5
mask_lgn[35-rad:35+rad, 42-rad:42+rad, 28-rad:28+rad] = True

# Select all the fibers that enter the LGN and discard all others
filtered_fibers2 = utils.near_roi(streamlines, affine, mask_lgn, tol=1.8)

sfil = []
for i in range(len(streamlines)):
    if filtered_fibers2[i]:
        sfil.append(streamlines[i])
streamlines = Streamlines(sfil)


from dipy.viz import colormap, has_fury, actor, window

if has_fury:
    # Prepare the display objects.
    color = colormap.line_colors(streamlines)

    streamlines_actor = actor.line(streamlines,
                                   colormap.line_colors(streamlines))

    # Create the 3D display.
    scene = window.Scene()
    scene.add(streamlines_actor)

    # Save still images for this static example. Or for interactivity use
    window.record(scene, out_path='tractogram_EuDX.png', size=(800, 800))
    if interactive:
        window.show(scene)
"""
from dipy.denoise.enhancement_kernel import EnhancementKernel

D33 = 1.0
D44 = 0.02
t = 1
k = EnhancementKernel(D33, D44, t)

# Apply FBC measures
from dipy.tracking.fbcmeasures import FBCMeasures

fbc = FBCMeasures(streamlines, k)

from dipy.viz import window, actor

# Calculate LFBC for original fibers
fbc = memoryview(fbc)
fbc_sl_orig, clrs_orig, rfbc_orig = \
  fbc.get_points_rfbc_thresholded(0.2, emphasis=0.01)


# Create scene
scene = window.Scene()

# Original lines colored by LFBC
lineactor = actor.line(fbc_sl_orig, clrs_orig, linewidth=0.2)
scene.add(lineactor)

# Show original fibers
scene.set_camera(position=(-264, 285, 155),
                 focal_point=(0, -14, 9),
                 view_up=(0, 0, 1))
window.record(scene, n_frames=1, size=(900, 900))
if interactive:
    window.show(scene)
"""