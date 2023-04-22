"""
@author: shrut
"""
from fury import window, actor
interactive = True

import numpy as np
import pandas as pd
from dipy.io.streamline import load_tractogram
from dipy.io.utils import create_tractogram_header
from dipy.io.vtk import transform_streamlines
from dipy.io.image import load_nifti
from time import sleep
from dipy.segment.bundles import bundle_shape_similarity
from dipy.tracking.utils import length
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.streamline import cluster_confidence, Streamlines
import matplotlib.pyplot as plt

plt.close('all')

from os.path import expanduser, join
home = expanduser('~')
subjects = ['S00393', 'S00490', 'S00613', 'S00680', 'S00699', 'S00795', 'S01952']

subject = 'S00393'

print('Subject is ' + subject)

dname = join(home, '.dipy', 'sample_files', subject)
dname2 = join(home, '.spyder-py3')

fdwi = join(dname, str(subject) + '_resample.nii')
data, affine, img = load_nifti(fdwi, return_img=True)

#b0_mask, mask = median_otsu(data, vol_idx = np.arange(8, 12), median_radius=4, numpass=2)
#create fa
#from dipy.reconst.dti import TensorModel
#tensor_model = TensorModel(gtab)
#tensor_fit = tensor_model.fit(data, mask)

fa_map, affine = load_nifti(join(dname2, str(subject) + '_fa.nii.nii'))

f_l = join(dname2, 'ha_l_' + str(subject) + '.trk')
f_r = join(dname2, 'ha_r_' + str(subject) + '.trk')

sft_l = load_tractogram(f_l, "same", bbox_valid_check = False)
sft_r = load_tractogram(f_r, "same", bbox_valid_check = False)

hipp_amyg_l = sft_l.streamlines
hipp_amyg_r = sft_r.streamlines
target_header = create_tractogram_header(f_l,
                                        *sft_l.space_attributes)

def cci_filter(streamlines, cci_thresh):
    
    lengths = list(length(streamlines))
    
    len_filter = Streamlines()
    for i, sl in enumerate(streamlines):
        if lengths[i] > 20:
            len_filter.append(sl)
    
    cci = cluster_confidence(len_filter)
    
    #cci = cluster_confidence(streamlines, override = True)

    """ 
    fig, ax = plt.subplots(1)
    ax.hist(cci, bins=100, histtype='step')
    ax.set_xlabel('CCI')
    ax.set_ylabel('# streamlines')
    fig.savefig('cci_histogram.png')
    """
    keep_streamlines = Streamlines()
    for i, sl in enumerate(len_filter):
        if cci[i] > cci_thresh:
            keep_streamlines.append(sl)
                
    return keep_streamlines
   
hipp_amyg_l = cci_filter(hipp_amyg_l, 0)
hipp_amyg_r = cci_filter(hipp_amyg_r, 0)
 
def show_both_bundles(bundles, colors=None, show=True, fname=None):
    
    scene = window.Scene()
    scene.SetBackground(1., 1, 1)
    for (i, bundle) in enumerate(bundles):
        color = colors[i]
        lines_actor = actor.streamtube(bundle, color, linewidth=0.1)
        lines_actor.RotateX(-90)
        lines_actor.RotateZ(90)
        scene.add(lines_actor)
    if show:
        window.show(scene)
    if fname is not None:
        sleep(1)
        window.record(scene, n_frames=1, out_path=fname, size=(900, 900))
        
def left_right_sim(static, moving, threshold, flip):
       
    show_both_bundles([static, moving], colors=[window.colors.blue, window.colors.orange],
         show=True, fname=None)   
    
    if flip:
        flip_mat = np.array([[-1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]], np.float64)
        moving = transform_streamlines(moving, flip_mat, in_place = False)
        
    srr = StreamlineLinearRegistration()
    srm = srr.optimize(static=static, moving=moving)
    rng = np.random.RandomState()
    
    l_aligned = srm.transform(moving)
    
    scene = window.Scene()
    scene.SetBackground(1., 1, 1)
    
    show_both_bundles([static, l_aligned], colors=[window.colors.blue, window.colors.orange],
         show=True, fname=None)
    
    print('Affine transform and alignment completed!')
    
    clust_thr = [0]
    
    ba_score = bundle_shape_similarity(static, l_aligned, rng, clust_thr, threshold)
    return ba_score

points = 60

#left strm hs points number of points
hipp_amyg_l = set_number_of_points(hipp_amyg_l, points)
hipp_amyg_r = set_number_of_points(hipp_amyg_r, points)

#find all of the coordinate points for the streamlines
l_coord = []
for streamline in hipp_amyg_l:
    for (x,y,z) in streamline:
        l_coord.append([int(x), int(y), int(z)])
        
r_coord = []
for streamline in hipp_amyg_r:
    for (x,y,z) in streamline:
        r_coord.append([int(x), int(y), int(z)])
        
mid_sag = (data.shape[0]) // 2

def fa_lesion(fa_map, coord):
    fa_diff = []
    for i in range(len(coord)):
        sagittal_offset = abs(coord[i][0] - mid_sag)
        x = coord[i][0]
        y = coord[i][1]
        if coord[i][0] > mid_sag:
            x_new = coord[i][0] - 2*sagittal_offset
        else:
            x_new = coord[i][0] + 2*sagittal_offset

        z = coord[i][2]
        
        fa_around= np.array(fa_map[x_new-1:x_new+1, y-1:y+1, z-1:z+1])
        fa_opp_side = np.average(fa_around)

        diff = fa_map[x,y,z] - fa_opp_side
        fa_diff.append(diff)

    return fa_diff
        
fa_off_left = fa_lesion(fa_map, l_coord)
fa_off_right = fa_lesion(fa_map, r_coord)

def fa_lesion_visual(streamlines, fa, ref, orientation = 0):
    
    scene = window.Scene()
    fa_cmap = actor.colormap_lookup_table(scale_range = (min(fa), max(fa)))
    scene.add(actor.line(streamlines, fake_tube=False, colors=fa,
                     linewidth=2))
        
    cbar = actor.scalar_bar(fa_cmap)
    scene.add(cbar)
    scene.set_camera(focal_point=(51, 51, 20.1),
                         position=(46, 277, -200), view_up=(0.1, -.8, -.07))
    
    if data.ndim == 3:
        vol_actor = actor.slicer(ref, opacity = 4.)
    elif data.ndim == 4:
        vol_actor = actor.slicer(ref[:,:,:,10], opacity = 4.)

    if orientation == "sagittal":
        vol_actor.display(x = ref.shape[0] // 2)
    elif orientation == "coronal":
        vol_actor.display(y = ref.shape[1] // 2)
    elif orientation == "axial":
        vol_actor.display(z = ref.shape[2] // 2)
    elif orientation == 0:
        vol_actor.display(x = 0)
        
    scene.add(vol_actor)
    
    if interactive:
        window.show(scene)
        
        
fa_lesion_visual(hipp_amyg_l, fa_off_left, data, 'sagittal')
fa_lesion_visual(hipp_amyg_r, fa_off_right, data, 'sagittal')

def fa_streamlines (fa_map, streamlines):
    df_fornow = pd.DataFrame(data = None)
    
    for i, streamline in enumerate(streamlines):
        fa_streamline = [fa_map[int(p[0]), int(p[1]), int(p[2])] for p in streamline]
        df_fornow = df_fornow.append(pd.Series(fa_streamline), ignore_index = True)

    return df_fornow

def fa_point_visual(streamlines, fa, ref, orientation = 0):
        
        fa_cmap = actor.colormap_lookup_table(scale_range = (0, 0.5))
        scene = window.Scene()
        
        for (i, streamline) in enumerate(streamlines):
            scene.add(actor.line(streamline, fa, lookup_colormap=(fa_cmap)))
            
        cbar = actor.scalar_bar(fa_cmap)
        scene.add(cbar)
        scene.set_camera(focal_point=(51, 51, 20.1),
                             position=(46, 277, -200), view_up=(0.1, -.8, -.07))
        
        if data.ndim == 3:
            vol_actor = actor.slicer(ref, opacity = 4.)
        elif data.ndim == 4:
            vol_actor = actor.slicer(ref[:,:,:,10], opacity = 4.)
    
        if orientation == "sagittal":
            vol_actor.display(x = ref.shape[0] // 2)
        elif orientation == "coronal":
            vol_actor.display(y = ref.shape[1] // 2)
        elif orientation == "axial":
            vol_actor.display(z = ref.shape[2] // 2)
        elif orientation == 0:
            vol_actor.display(x = 0)
            
        scene.add(vol_actor)
        
        if interactive:
            window.show(scene)
            #window.save_image(scene, 'Subject' + str(subject) + 'L and R points by FA')
            
fa_point_visual([hipp_amyg_l, hipp_amyg_r], fa_map, data, 'none')