# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 18:12:26 2021

@author: shrut
"""

from dipy.io.image import load_nifti
import numpy as np
from dipy.segment.bundles import RecoBundles
from dipy.align.streamlinear import whole_brain_slr
from fury import actor, window
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_trk, save_trk
from dipy.io.utils import create_tractogram_header
from time import sleep
from dipy.io.vtk import transform_streamlines
from os.path import expanduser, join
home = expanduser('~')

subject = 'S00393'
dname = join(home, '.dipy', 'sample_files', subject)
dname2 = join(home, '.dipy', 'bundle_atlas_hcp842', 'Atlas_80_Bundles')

fref = join(dname, 'S00393_Reg_LPCA_nii3D.nii')
ref, affine, img = load_nifti(fref, return_img = True)
ftarget_tract = join(dname, '00393_stepsize_2_all_wholebrain_pruned_100.trk')
flabels = join(dname, 'S00393_label_resample.nii')
fatlas = join(dname2, 'whole_brain', 'whole_brain_MNI.trk')

sft_atlas = load_trk(fatlas, "same", bbox_valid_check=False, )
atlas = sft_atlas.streamlines
atlas_header = create_tractogram_header(fatlas,
                                        *sft_atlas.space_attributes)

sft_target = load_trk(ftarget_tract, "same", bbox_valid_check=False)
target = sft_target.streamlines
target_header = create_tractogram_header(fatlas,
                                        *sft_atlas.space_attributes)
print("Data loaded")
    
#Carry out affine transforms to get our brain and the atlas to line up a little
moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
    atlas, target, x0='affine', verbose=True, progressive=True,
    rng=np.random.RandomState(1984))

np.save("slr_transform.npy", transform)

interactive = True

"""
#flip_mat = np.array([[1, 0, 0, 0],[0, -1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]], np.float64)
#target = transform_streamlines(target, flip_mat, in_place = False)

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(np.array(atlas, dtype = object), colors=(1, 0, 1)))
scene.add(actor.line(np.array(target, dtype = object), colors=(1, 1, 0)))
window.record(scene, out_path='tractograms_initial.png', size=(600, 600))
if interactive:
    window.show(scene)
    
scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(np.array(atlas, dtype = object), colors=(1, 0, 1)))
scene.add(actor.line(np.array(moved, dtype = object), colors=(1, 1, 0)))
#window.record(scene, out_path='tractograms_after_registration.png',
           #   size=(600, 600))
if interactive:
    window.show(scene)
"""
#Model bundle loading
boi = 'C'

fbundle_l = join(dname2, 'bundles', str(boi) + '_L.trk')
fbundle_r = join(dname2, 'bundles', str(boi) + '_R.trk')

sft_bundle_l = load_trk(fbundle_l, "same", bbox_valid_check=False)
model_l = sft_bundle_l.streamlines
bundle_header = create_tractogram_header(fbundle_l,
                                        *sft_bundle_l.space_attributes)

sft_bundle_r = load_trk(fbundle_r, "same", bbox_valid_check=False)
model_r = sft_bundle_r.streamlines
bundle_header_r = create_tractogram_header(fbundle_r,
                                        *sft_bundle_r.space_attributes)

print("Model bundle data loaded!")
rb = RecoBundles(moved, verbose=True, rng=np.random.RandomState(2001))

extracted_bundle_l, labels_l = rb.recognize(model_bundle=model_l,
                                            model_clust_thr=.01,
                                            reduction_thr=20,
                                            pruning_thr=10,
                                            reduction_distance='mdf',
                                            slr=True,
                                            slr_metric='asymmetric',
                                            pruning_distance='mdf')

extracted_bundle_r, labels_r = rb.recognize(model_bundle=model_r,
                                            model_clust_thr=.01,
                                            reduction_thr=20,
                                            pruning_thr=10,
                                            reduction_distance='mdf',
                                            slr=True,
                                            slr_metric='asymmetric',
                                            pruning_distance='mdf')


if len(labels_l) and len(labels_r):
    print("Recobundles working")
    
    scene = window.Scene()
    scene.SetBackground(1, 1, 1)
    scene.add(actor.line(extracted_bundle_l, colors=(.93, .66, .3)))
    scene.add(actor.line(extracted_bundle_r, colors=(.44, .75, .8)))
    scene.set_camera(focal_point=(320, 21,  17),
                     position=(2.11, 200.46, 250.44), view_up=(0.1, -1.028, 0.18))
    window.record(scene, out_path= str(boi) + 'bundles.png',
              size=(400, 400))
    if interactive:
        window.show(scene)
        
    scene = window.Scene()
    scene.SetBackground(1, 1, 1)
    scene.add(actor.line(model_l, colors=(.8, .5, .3)))
    scene.add(actor.line(model_r, colors=(.3, .55, .4)))
    scene.set_camera(focal_point=(320, 21,  17),
                     position=(2.11, 200.46, 250.44), view_up=(0.1, -1.028, 0.18))
    window.record(scene, out_path= str(boi) + '_modelbundles.png',
              size=(400, 400))
    if interactive:
        window.show(scene)       

else:
    import sys
    print("Recobundles did not work")
    print('Len(labels_r) = %1.1f, Len(labels_l) = %1.1f' % (len(labels_r), len(labels_l)))
    sys.exit()
    
sft_r = StatefulTractogram(extracted_bundle_r, img, Space.RASMM)
sft_l = StatefulTractogram(extracted_bundle_l, img, Space.RASMM)
save_trk(sft_r, str(boi) + "_R.trk", bbox_valid_check = False)
save_trk(sft_l, str(boi) + "_L.trk", bbox_valid_check = False)
