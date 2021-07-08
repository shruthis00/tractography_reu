# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 18:12:26 2021

@author: shrut
"""

from dipy.tracking.streamline import Streamlines
from dipy.io.image import load_nifti, load_nifti_data
import numpy as np
from dipy.segment.bundles import RecoBundles
from dipy.align.streamlinear import whole_brain_slr
from fury import actor, window
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_trk, save_trk
from dipy.io.utils import create_tractogram_header

from dipy.data.fetcher import get_two_hcp842_bundles
from dipy.data.fetcher import (fetch_target_tractogram_hcp,
                               fetch_bundle_atlas_hcp842,
                               get_bundle_atlas_hcp842,
                               get_target_tractogram_hcp)
from dipy.io.vtk import transform_streamlines
from os.path import expanduser, join
home = expanduser('~')

subject = 'S00393'
dname = join(home, '.dipy', 'sample_files', subject)
dname2 = join(home, '.dipy', 'bundle_atlas_hcp842', 'Atlas_80_Bundles', 'whole_brain')

ftarget_tract = join(dname, 'whole_brainS00393.trk')
flabels = join(dname, 'S00393_label_resample.nii')
fatlas = join(dname2, 'whole_brain_MNI.trk')

#atlas_file, atlas_folder = fetch_bundle_atlas_hcp842()
sft_atlas = load_trk(fatlas, "same", bbox_valid_check=False, )
atlas = sft_atlas.streamlines
atlas_header = create_tractogram_header(fatlas,
                                        *sft_atlas.space_attributes)

sft_target = load_trk(ftarget_tract, "same", bbox_valid_check=False)
target = sft_target.streamlines
target_header = create_tractogram_header(fatlas,
                                        *sft_atlas.space_attributes)

#flip_mat = np.array([[1, 0, 0, 0],[0, -1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]], np.float64)
#target = transform_streamlines(target, flip_mat, in_place = False)

interactive = True
scene = window.Scene()
scene.SetBackground(1, 1, 1)
#scene.add(actor.line(np.array(atlas, dtype = object), colors=(1, 0, 1)))
scene.add(actor.line(np.array(target, dtype = object), colors=(1, 1, 0)))
window.record(scene, out_path='tractograms_initial.png', size=(600, 600))
if interactive:
    window.show(scene)
    
#Carry out affine transforms to get our brain and the atlas to line up a little
moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
    atlas, target, x0='affine', verbose=True, progressive=True,
    rng=np.random.RandomState(1984))

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(atlas, colors=(1, 0, 1)))
scene.add(actor.line(moved, colors=(1, 1, 0)))
window.record(scene, out_path='tractograms_after_registration.png',
              size=(600, 600))
if interactive:
    window.show(scene)
    
np.save("slr_transform.npy", transform)

model_c_r_file, model_cst_l_file = get_two_hcp842_bundles()

sft_af_l = load_trk(model_c_r_file, "same", bbox_valid_check=False)
model_af_l = sft_af_l.streamlines

rb = RecoBundles(moved, verbose=True, rng=np.random.RandomState(2001))

extracted_afl, af_l_labels = rb.recognize(model_bundle=model_af_l,
                                            model_clust_thr=5,
                                            reduction_thr=15,
                                            pruning_thr=7,
                                            reduction_distance='mdf',
                                            slr=True,
                                            slr_metric='asymmetric',
                                            pruning_distance='mdf')
if len(af_l_labels):
    print("Recobundles working")
    
else:
    print("Recobundles did not work)")
    exit()


scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(model_af_l, colors=(.1, .7, .26)))
scene.add(actor.line(extracted_afl, colors=(.1, .1, 6)))
scene.set_camera(focal_point=(320.21296692, 21.28884506,  17.2174015),
                 position=(2.11, 200.46, 250.44), view_up=(0.1, -1.028, 0.18))
window.record(scene, out_path='AF_L_recognized_bundle.png',
              size=(600, 600))
if interactive:
    window.show(scene)
