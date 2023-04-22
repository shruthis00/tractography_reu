# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:20:13 2021
@author: shrut
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.tissue import TissueClassifierHMRF
from os.path import join, expanduser

home = expanduser('~')
subject = 'S00393'
dname = join(home, '.dipy', 'sample_files', subject)

fdwi = join(dname, 'Full_DWI_subj.nii')
fbval = join(dname, 'bvals.txt')
fbvec = join(dname, 'bvecs.txt')

data, affine, img= load_nifti(fdwi, return_img=True)
tissue_data = np.asanyarray(img.dataobj)

print(tissue_data.shape)

fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
img_ax = np.rot90(tissue_data[..., 89])
imgplot = plt.imshow(img_ax, cmap="gray")
a.axis('off')
a.set_title('Axial')
a = fig.add_subplot(1, 2, 2)
img_cor = np.rot90(tissue_data[:, 128, :])
imgplot = plt.imshow(img_cor, cmap="gray")
a.axis('off')
a.set_title('Coronal')
plt.savefig('t1_image.png', bbox_inches='tight', pad_inches=0)

#specifying types of classes
nclass = 3

#smoothness factor of segmentation
beta = 0.1

#Set number of iterations
t0 = time.time()
hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(tissue_data, nclass, beta)

t1 = time.time()
total_time = t1-t0
print('Total time:' + str(total_time))

fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
img_ax = np.rot90(final_segmentation[..., 89])
imgplot = plt.imshow(img_ax)
a.axis('off')
a.set_title('Axial')
a = fig.add_subplot(1, 2, 2)
img_cor = np.rot90(final_segmentation[:, 128, :])
imgplot = plt.imshow(img_cor)
a.axis('off')
a.set_title('Coronal')
plt.savefig('final_seg.png', bbox_inches='tight', pad_inches=0)

csf = PVE[:,:,:,0]
gm = PVE[:,:,:,1]
wm = PVE[:,:,:,2]

fig = plt.figure()
a = fig.add_subplot(1, 3, 3)
img_cor = np.rot90(PVE[:, :, 89, 2])
imgplot = plt.imshow(img_cor, cmap="gray")
a.axis('off')
a.set_title('White Matter')

wm = (wm >= 0.20)
wm = wm.astype(int)

save_nifti(str(subject) + '_wm', wm, affine)
