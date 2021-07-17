# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 02:31:50 2021

@author: shrut
"""
import numpy as np
from dipy.viz import window, actor
from dipy.io.streamline import load_trk
from time import sleep
from dipy.io.utils import create_tractogram_header
from dipy.segment.bundles import bundle_shape_similarity
from dipy.tracking.utils import length
from dipy.tracking.metrics import downsample
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.io.vtk import transform_streamlines
from dipy.tracking.streamline import set_number_of_points

#load data
from os.path import expanduser, join
home = expanduser('~')

subject = 'S00393'
dname = join(home, '.spyder-py3')

boi = 'C'

f_boi_r = join(dname, str(boi) + '_L.trk')
f_boi_l = join(dname, str(boi) + '_R.trk')

sft_r = load_trk(f_boi_r, "same", bbox_valid_check=False, )
r_extract = sft_r.streamlines
r_header = create_tractogram_header(f_boi_r,
                                        *sft_r.space_attributes)

sft_l = load_trk(f_boi_l, "same", bbox_valid_check=False, )
l_extract = sft_l.streamlines
l_header = create_tractogram_header(f_boi_l,
                                        *sft_r.space_attributes)

r_extract = set_number_of_points(r_extract, 60)
l_extract = set_number_of_points(l_extract, 60)
print('Extracted bundles loaded!')
#Length Calculation
#Calculate number of voxels in each tract

#Mirror flip one of the bundles for shape similarity assessment
#flip_mat = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, -1, 0],[0, 0, 0, 1]], np.float64)
#l_extract = transform_streamlines(l_extract, flip_mat, in_place = False)

srr = StreamlineLinearRegistration()
srm = srr.optimize(static=r_extract, moving=l_extract)
rng = np.random.RandomState()

l_aligned = srm.transform(l_extract)

scene = window.Scene()
scene.SetBackground(1., 1, 1)

def show_both_bundles(bundles, colors=None, show=True, fname=None):

    scene = window.Scene()
    scene.SetBackground(1., 1, 1)
    for (i, bundle) in enumerate(bundles):
        color = colors[i]
        lines_actor = actor.streamtube(bundle, color, linewidth=0.3)
        lines_actor.RotateX(-90)
        lines_actor.RotateZ(90)
        scene.add(lines_actor)
    if show:
        window.show(scene)
    if fname is not None:
        sleep(1)
        window.record(scene, n_frames=1, out_path=fname, size=(900, 900))

#show_both_bundles([r_extract, l_aligned], colors=[window.colors.orange, window.colors.red],
 #       show=True, fname=None)

print('Affine transform and alignment completed!')

clust_thr = [0]
threshold = 10

ba_score = bundle_shape_similarity(r_extract, l_aligned, rng, clust_thr, threshold)
print("Extracted shape similarity score = ", ba_score)

#Testing similarity of the model bundles to see whether that even makes sense
dname2 = join(home, '.dipy', 'bundle_atlas_hcp842', 'Atlas_80_Bundles')
fmodel_l = join(dname2, 'bundles', str(boi) + '_L.trk')
fmodel_r = join(dname2, 'bundles', str(boi) + '_R.trk')

sft_model_l = load_trk(fmodel_l, "same", bbox_valid_check=False)
model_l = sft_model_l.streamlines
bundle_header_l = create_tractogram_header(fmodel_l,
                                        *sft_model_l.space_attributes)

sft_model_r = load_trk(fmodel_r, "same", bbox_valid_check=False)
model_r = sft_model_r.streamlines
bundle_header_r = create_tractogram_header(fmodel_r,
                                        *sft_model_r.space_attributes)

model_r = set_number_of_points(model_r, 40)
model_l = set_number_of_points(model_l, 40)
print('Model bundles loaded!')

srr = StreamlineLinearRegistration()
srm = srr.optimize(static=model_r, moving=model_l)
rng = np.random.RandomState()

model_l_aligned = srm.transform(model_l)
        
show_both_bundles([model_r, model_l_aligned], colors=[window.colors.orange, window.colors.red],
     show=True, fname=None)

clust_thr = [0]
threshold = 10
ba_score = bundle_shape_similarity(model_r, model_l_aligned, rng, clust_thr, threshold)
print("Model shape similarity score = ", ba_score)


print(str(boi) + 'R has %d streamlines' % len(r_extract))
print(str(boi) + 'L has %d streamlines' % len(l_extract))

length_r = list(length(r_extract))
length_l = list(length(l_extract))

avg_right = np.average(length_r)
avg_left = np.average(length_l)

print('Length of streamlines in both bundles computed')

import matplotlib.pyplot as plt

def length_hist (length_list, color):
    
    fig_hist, ax = plt.subplots(1)
    ax.hist(length_list, color=color)
    ax.set_xlabel('Length')
    ax.set_ylabel('Count')
    plt.show()
    #plt.savefig(' Length Distribution.png')

r_hist = length_hist(length_r, (.44, .75, .8))
l_list = length_hist(length_l, (.93, .66, .3))

l_r = 0
l_c = 0

#S_rc = 1 - abs((l_r - l_c)/(l_r+l_c))
