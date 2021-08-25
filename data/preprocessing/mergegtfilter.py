import nibabel as nib
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb

nii_files = glob.glob('./datanii/*.nii')
nii_files.sort()

for nii_file in nii_files:
	print(nii_file)
	nii_input = nib.load(nii_file)
	id_ = os.path.basename(nii_file)[:-4]
	raw_files = glob.glob('./gtraw/' + id_ + '_*') # multiple class gt per id
	raw_files.sort()
	mask_out = np.zeros((nii_input.shape[2], nii_input.shape[1], nii_input.shape[0]))
	for i, raw_file in enumerate(raw_files, 1):
		mask = np.fromfile(raw_file, dtype='uint8', sep="")
		mask = mask.reshape(mask_out.shape)
		mask_out += mask
	mask_out = np.where(mask_out > 0, 1, 0)
	fileobj = open(os.path.join(nii_file[:-4] + '_gt1.raw'), mode='wb')
	off = np.array(mask_out, dtype=np.uint8)
	off.tofile(fileobj)
	fileobj.close()
	#pdb.set_trace()
