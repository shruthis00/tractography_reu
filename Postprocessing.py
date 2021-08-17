# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 11:25:02 2021

@author: shrut

Shape similarity, length distribution, and fa mapping for both sides addressed here
"""

from fury import window, actor, colormap as cmap
interactive = True

import numpy as np
import pandas as pd
from dipy.io.streamline import save_trk, load_tractogram
from dipy.io.utils import create_tractogram_header
from dipy.io.vtk import transform_streamlines
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from time import sleep
from dipy.segment.bundles import bundle_shape_similarity
from dipy.tracking.utils import length
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.streamline import cluster_confidence, Streamlines
import matplotlib.pyplot as plt
from dipy.io.stateful_tractogram import Space, StatefulTractogram

from os.path import expanduser, join
home = expanduser('~')

df_tot = pd.DataFrame(data = None, columns = ['Subject', 'Number of Streamlines',
                      'Mean FA', 'Mean Length of Streamlines'])

ba = []
subject = [1, 2, 3] #Subject name here if intending to iterate through a for loop
for i, subject in enumerate(subjects):
    
    print('Subject is ' + subject)
    
    dname = join(home, '.dipy', 'sample_files', subject)
    dname2 = join(home, '.spyder-py3')
    
    fdwi = join(dname, str(subject) + '_resample.nii')
    data, affine, img = load_nifti(fdwi, return_img=True)
    
    #b0_mask, mask = median_otsu(data, vol_idx = np.arange(8, 12), median_radius=4, numpass=2)
    #create FA
    #from dipy.reconst.dti import TensorModel
    #tensor_model = TensorModel(gtab)
    #tensor_fit = tensor_model.fit(data, mask)
    
    FA_map, affine = load_nifti(str(subject) + '_fa.nii.nii')
    
    f_l = join(dname2, 'ha_l_' + str(subject) + '.trk')
    f_r = join(dname2, 'ha_r_' + str(subject) + '.trk')
    
    sft_l = load_tractogram(f_l, "same", bbox_valid_check = False)
    sft_r = load_tractogram(f_r, "same", bbox_valid_check = False)
    
    hipp_to_amyg_l = sft_l.streamlines
    hipp_to_amyg_r = sft_r.streamlines
    target_header = create_tractogram_header(f_l,
                                            *sft_l.space_attributes)
    
    def cci_filter(streamlines, cci_thresh):
    
        """
        lengths = list(length(streamlines))
        print(len(streamlines))
        
        long_out = Streamlines()
        for i, sl in enumerate(streamlines):
            if lengths[i] > 5:
                long_out.append(sl)
        print(len(long_out))
        """
        cci = cluster_confidence(streamlines, override = True)
        """
        fig, ax = plt.subplots(1)
        ax.hist(cci, bins=100, histtype='step')
        ax.set_xlabel('CCI')
        ax.set_ylabel('# streamlines')
        fig.savefig('cci_histogram.png')
        """
        keep_streamlines = Streamlines()
        for i, sl in enumerate(streamlines):
            if cci[i] > cci_thresh:
                keep_streamlines.append(sl)
                    
        return keep_streamlines
    
    hipp_to_amyg_l = cci_filter(hipp_to_amyg_l, 0)
    hipp_to_amyg_r = cci_filter(hipp_to_amyg_r, 0)
    
    df_l = [str(subject) + '_l', len(hipp_to_amyg_l)]
    df_r = [str(subject) + '_r', len(hipp_to_amyg_r)]
    
    #sft = StatefulTractogram(hipp_to_amyg_new, img, Space.RASMM)
    #save_trk(sft, str(subject) + 'cci_thresholding_after.trk', bbox_valid_check= False)
    
    #sft = StatefulTractogram(hipp_to_amyg_l, img, Space.RASMM)
    #save_trk(sft, str(subject) + 'cci_thresholding_before.trk', bbox_valid_check= False)
    
    def FA_streamlines (FA_map, streamlines):
        fa_streamlines = []
        
        for i, streamline in enumerate(streamlines):
            fa_streamline = [FA_map[int(p[0]), int(p[1]), int(p[2])] for p in streamline]
            
            #mean FA in streamline
            fa_streamline = np.mean(fa_streamline)
            #print('Mean of streamline is ' + str(fa_streamline))
            fa_streamlines.append(fa_streamline)
        return fa_streamlines
    
    hipp_amyg_fal = np.array(FA_streamlines(FA_map, hipp_to_amyg_l))
    hipp_amyg_far = np.array(FA_streamlines(FA_map, hipp_to_amyg_r))
    
    #np.save(str(subject) + 'ha_fa_l', hipp_amyg_fal)
    #np.save(str(subject) + 'ha_fa_r', hipp_amyg_far)

    def fa_stream_visual(streamlines, fa, ref, orientation):
        interactive = True
        
        fa_cmap = actor.colormap_lookup_table(scale_range = (0, 0.5))
        
        scene = window.Scene()
        scene.add(actor.line(streamlines, fa, lookup_colormap=(fa_cmap)))
        scene.set_camera(focal_point=(51, 51, 20.1),
                         position=(46, 277, -200), view_up=(0.1, -.8, -.07))
        cbar = actor.scalar_bar(fa_cmap)
        scene.add(cbar)
        
        if data.ndim == 3:
            vol_actor = actor.slicer(ref, opacity = 2.)
        elif data.ndim == 4:
            vol_actor = actor.slicer(ref[:,:,:,10], opacity = 2.)
    
        if orientation == "sagittal":
            vol_actor.display(x = ref.shape[0] // 2)
        elif orientation == "coronal":
            vol_actor.display(y = ref.shape[1] // 2)
        elif orientation == "axial":
            vol_actor.display(z = ref.shape[2] // 2)
    
        scene.add(vol_actor)
        
        if interactive:
            window.show(scene)
    
    fa_stream_visual(hipp_to_amyg_l, hipp_amyg_fal, np.asanyarray(data), 'axial')
    fa_stream_visual(hipp_to_amyg_r, hipp_amyg_far, np.asanyarray(data), 'axial')
    
    #load data
    hipp_to_amyg_l = set_number_of_points(hipp_to_amyg_l, 60)
    hipp_to_amyg_r = set_number_of_points(hipp_to_amyg_r, 60)
    print('Extracted bundles loaded!')
    #Length Calculation
    #Calculate number of voxels in each tract
    
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
   
    def left_right_sim (static, moving, threshold, flip):
        
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
    
    ba_score = left_right_sim(hipp_to_amyg_r, hipp_to_amyg_l, 10, flip = True)
    #print("Similarity score = ", ba_score)
    ba.append(ba_score)
    
    #length
    #print('R has %d streamlines' % len(hipp_to_amyg_r))
    #print('L has %d streamlines' % len(hipp_to_amyg_l))
    
    length_l = list(length(hipp_to_amyg_l))
    length_r = list(length(hipp_to_amyg_r))
    
    avg_r = np.average(length_r)
    avg_l = np.average(length_l)
      
    df_l.extend([np.mean(hipp_amyg_fal), avg_l])
    df_r.extend([np.mean(hipp_amyg_far), avg_r])
    
    def length_hist (length_list, color):
        """
        fig_hist, ax = plt.subplots(1)
        print(ax)
        print(type(color))
        print(type(length_list))
        ax.hist(length_list, color=color)
        ax.set_xlabel('Length')
        ax.set_ylabel('Count')
        plt.show()
        #plt.savefig(' Length Distribution.png')
        """
    r_hist = length_hist(length_r, (.44, .75, .8))
    l_list = length_hist(length_l, (.93, .66, .3))
    
    
    def length_visual (bundle_l, bundle_r, length_list_l, length_list_r):
        #length_list_l_cmap = actor.colormap_lookup_table(scale_)
        
        length_cmap = actor.colormap_lookup_table(scale_range = (3, 100))
        
        scene = window.Scene()
        scene.add(actor.line(bundle_l, length_list_l, lookup_colormap=(length_cmap)))
        scene.add(actor.line(bundle_r, length_list_r, lookup_colormap=(length_cmap)))

        scene.set_camera(focal_point=(51, 51, 20.1),
                         position=(46, 277, -200), view_up=(0.1, -.8, -.07))
        
        cbar = actor.scalar_bar(length_cmap)
        scene.add(cbar) 
        
        if interactive:
            window.show(scene)
        
    #r_hist = length_hist(length_r, (.44, .75, .8))
    #l_list = length_hist(length_l, (.93, .66, .3))
    
    length_visual(hipp_to_amyg_l, hipp_to_amyg_r, np.asanyarray(length_l), np.array(length_r))
    
    print('Length of streamlines in both bundles computed')
    
    df_tot = df_tot.append((pd.Series(df_l, index = df_tot.columns)), ignore_index = True)
    df_tot = df_tot.append((pd.Series(df_r, index = df_tot.columns)), ignore_index = True)
   
ba = pd.Series(ba, index = subjects)

#df_tot.to_csv(r'C:\Users\shrut\.dipy\sample_files\df_tot.csv', columns = df_tot.columns)
#ba.to_csv(r'C:\Users\shrut\.dipy\sample_files\ba.csv')
