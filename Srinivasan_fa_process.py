# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 20:17:04 2021

@author: shrut
"""
from fury import window, actor, colormap as cmap
interactive = True

import pandas as pd
from dipy.io.streamline import load_tractogram
from dipy.io.utils import create_tractogram_header
from dipy.io.vtk import transform_streamlines
from dipy.io.image import load_nifti
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.streamline import cluster_confidence, Streamlines

from os.path import expanduser, join
home = expanduser('~')
subjects = ['1', '2', '3', '4'] # Insert subjects here

for i, subject in enumerate(subjects):
    
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

    hipp_to_amyg_l = set_number_of_points(hipp_to_amyg_l, 60)
    hipp_to_amyg_r = set_number_of_points(hipp_to_amyg_r, 60)
    
    def FA_streamlines (FA_map, streamlines):
        
        df_fornow = pd.DataFrame(data = None)
        
        for i, streamline in enumerate(streamlines):
            fa_streamline = [FA_map[int(p[0]), int(p[1]), int(p[2])] for p in streamline]
            df_fornow = df_fornow.append(pd.Series(fa_streamline), ignore_index = True)
    
        return df_fornow
    
    #Saving datasets
    path = join(home, '.dipy', 'sample_files', 'FA_datasets', str(subject) + '_l.csv')
    df_fa_l = FA_streamlines(FA_map, hipp_to_amyg_l)
    
    da_fa_l_mean = df_fa_l.mean(axis = 0)
    df_fa_l.append(pd.Series(da_fa_l_mean), ignore_index = True)
    df_fa_l.to_csv(path, index = False)

    path = join(home, '.dipy', 'sample_files', 'FA_datasets', str(subject) + '_r.csv')
    df_fa_r = FA_streamlines(FA_map, hipp_to_amyg_r)
    
    da_fa_r_mean = df_fa_l.mean(axis = 0)
    df_fa_r.append(pd.Series(da_fa_r_mean), ignore_index = True)
    df_fa_r.to_csv(path, index = False)
