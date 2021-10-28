interactive = True

import numpy as np
from dipy.io.streamline import load_tractogram
# importing save nifti
from dipy.io.image import load_nifti, save_nifti
from dipy.tracking.streamline import set_number_of_points
import matplotlib.pyplot as plt
from dipy.core.histeq import histeq

# This file is meant to convert trk file into nii file, or mask an nii
# file to show only the voxels where the tractogram streamlines are located

from os.path import expanduser, join
home = expanduser('~')
dname = join(home, '.dipy', 'sample_files')
dname2 = join(home, '.spyder-py3')

# load trk file and subject nifti file which should correspond
f_trk = join(dname2, '(subject data)' + '.trk')
f_ref = join(dname, subject, subject + '_resampled.nii')

# load data
ref, affine, img = load_nifti(f_ref, return_img = True)
sft_original = load_tractogram(f_trk, 'same', bbox_valid_check = False)

# current resampled data has 4 axes, removed one by making it static
ref = np.array(ref[:,:,:], dtype = 'float64')
orig_strm = sft_original.streamlines

orig_strm = set_number_of_points(orig_strm, 70)
b_mask = np.zeros(shape = (ref.shape[0], ref.shape[1], ref.shape[2]), )

count_overlap = 0
for streamline in orig_strm:
    for (x,y,z) in streamline:
        x = round(x)
        y = round(y)
        z = round(z)
        if ref[x,y,z] != 0:
            b_mask[x,y,z] = 1
            if b_mask[x,y,z] > 1:
                count_overlap += 1

# viz the mask
sagittal = ref.shape[0] // 2
coronal = ref.shape[1] // 2;
axial = ref.shape[2] // 2
plt.figure('Brain segmentation')
plt.subplot().set_axis_off()
plt.imshow(histeq(b_mask[:, coronal, :].astype('float')).T,
           cmap='gray', origin='lower')

save_nifti('trk_bmask_' + subject, b_mask, affine)
# Ok, now asymmetry indices?
# We've already hashed out a method for determining asymmetry
# find method for this in the Srinivasan_LesionID file
