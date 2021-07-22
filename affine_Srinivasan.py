# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 00:56:40 2021

@author: shrut
"""

import dipy
import numpy as np
from os.path import join, expanduser
from dipy.viz import regtools
from dipy.io.image import load_nifti
from dipy.align import (affine_registration, 
                        center_of_mass, translation,
                        rigid, affine, register_dwi_to_template)
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
home = expanduser('~')

dname = join(home, '.dipy', 'sample_files')

fdwi1 = join(dname, 'subj', 'subj_static_file.nii')
fdwi2 = join(dname, 'subj_moving', 'subj_moving_file.nii')

s_data, s_affine, s_img = load_nifti(fdwi1, return_img = True)
m_data, m_affine, m_img = load_nifti(fdwi2, return_img = True)

identity = np.eye(4)
affine_map = AffineMap(identity,
                       s_data.shape, s_affine,
                       m_data.shape, m_affine)
resampled = affine_map.transform(m_data)
regtools.overlay_slices(s_data, resampled, None, 0,
                        "Static", "Moving", "resampled_0.png")
regtools.overlay_slices(s_data, resampled, None, 1,
                        "Static", "Moving", "resampled_1.png")
regtools.overlay_slices(s_data, resampled, None, 2,
                        "Static", "Moving", "resampled_2.png")

nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)

#three levels of iterations, 10000 coarse, 1000 medium, and 100 fine. Feel free to change
level_iters = [10000, 1000, 100]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]
#All of this is default ^^ feel free to change

"""
affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)

transform = AffineTransform3D()
params0 = None
starting_affine = rigid.affine
affine = affreg.optimize(s_data, m_data, transform, params0,
                         static_affine, moving_affine,
                         starting_affine=starting_affine)
"""

pipeline = [center_of_mass, translation, rigid, affine]

transform, reg_affine = affine_registration(m_data, s_data, moving_affine = m_affine,
                                            static_affine = s_affine, nbins = 32,
                                            metric = 'MI', pipeline = pipeline,)

regtools.overlay_slices(s_data, transform, None, 0,
                        "Static", "Transformed", "xformed_affine_0.png")
regtools.overlay_slices(s_data, transform, None, 1,
                        "Static", "Transformed", "xformed_affine_1.png")
regtools.overlay_slices(s_data, transform, None, 2, 
                        "Static", "Transformed", "xformed_affine_2.png")

