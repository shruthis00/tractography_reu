# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 23:29:36 2021

@author: shrut
"""
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data

from os.path import expanduser, join
home = expanduser('~')

dname = join(home, '.dipy', 'sample_files', 'S00393')

flabel = join(dname, 'S00393wm_resample.nii')
flabels = join(dname, 'S00393_label_resample.nii')
fdwi = join(dname, 'S00393_resample.nii')
fbval = join(dname, 'S00393_bvals.txt')
fbvec = join(dname, 'S00393_bvecs.txt')

data, affine, img = load_nifti(fdwi, return_img=True)
label = load_nifti_data(flabel)
labels = load_nifti_data(flabels)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

white_matter = (label == 1)

from dipy.reconst.csdeconv import auto_response_ssst
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
csa_model = CsaOdfModel(gtab, sh_order=6)
csa_peaks = peaks_from_model(csa_model, data, default_sphere,
                             relative_peak_threshold=.8,
                             min_separation_angle=45,
                             mask=white_matter)

interactive = True

from dipy.viz import window, actor, has_fury

if has_fury:
    scene = window.Scene()
    scene.add(actor.peak_slicer(csa_peaks.peak_dirs,
                                csa_peaks.peak_values,
                                colors=None))

    window.record(scene, out_path='csa_direction_field.png', size=(900, 900))

    if interactive:
        window.show(scene, size=(800, 800))
        
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, .25)


from dipy.tracking import utils

seed_mask = (label == 1)
seeds = utils.seeds_from_mask(seed_mask, affine, density=[2, 2, 2])


from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines

# Initialization of LocalTracking. The computation happens in the next step.
streamlines_generator = LocalTracking(csa_peaks, stopping_criterion, seeds,
                                      affine=affine, step_size=.5)
# Generate streamlines object
streamlines = Streamlines(streamlines_generator)


from dipy.core.histeq import histeq
import matplotlib.pyplot as plt
sli = data.shape[2] // 2
plt.subplot(1, 2, 2).set_axis_off()
plt.imshow(histeq(white_matter[:, :, sli].astype('float')).T,
           cmap='gray', origin='lower')


from dipy.viz import colormap

if has_fury:
    # Prepare the display objects.
    color = colormap.line_colors(streamlines)

    streamlines_actor = actor.line(streamlines,
                                   colormap.line_colors(streamlines))

    # Create the 3D display.
    scene = window.Scene()
    scene.add(streamlines_actor)

    # Save still images for this static example. Or for interactivity use
    if interactive:
        window.show(scene)
        
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk

sft = StatefulTractogram(streamlines, img, Space.RASMM)
save_trk(sft, "tractogram_S00393.trk", streamlines)
        
csd_model = ConstrainedSphericalDeconvModel(gtab, response)


# Stopping Criterion


from dipy.reconst.shm import CsaOdfModel

csa_model = CsaOdfModel(gtab, sh_order=6)
gfa = csa_model.fit(data, mask=wm).gfa
stopping_criterion = ThresholdStoppingCriterion(gfa, .25)






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