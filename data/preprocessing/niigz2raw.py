import glob
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb

nii_files = glob.glob('./*.nii.gz')

for nii_file in nii_files:
	#nii_file = 'IterativeFCN_best_train_70295.pthCase2_pred.nii.gz2107092133.nii.gz'
	nii_input = nib.load(nii_file)
	header = nii_input.header
	arr = np.transpose(np.array(nii_input.dataobj), axes=[2, 1, 0])
	arr = np.transpose(arr, axes=[0, 2, 1])
	unique = np.unique(arr)	
	
	for i in unique[1:]:
		mask = np.where(arr==i, 1, 0)
		mask = np.transpose(mask, axes=[2, 1, 0])
		fileobj = open(f'{nii_file[:-7]}_gt{i}.raw', mode='wb')
		off = np.array(mask, dtype=np.uint8)
		off.tofile(fileobj)
		fileobj.close()

		print(f'{nii_file[:-7]}_gt{i}.raw', 'saved')
	#pdb.set_trace()
