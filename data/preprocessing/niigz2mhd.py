import glob
import SimpleITK as sitk
import numpy as np
import pdb

nii_files = glob.glob('./dataniigz/*.nii.gz')

for nii_file in nii_files:
	img = sitk.ReadImage(nii_file)
	#arr = sitk.GetArrayFromImage(img)
	#sagital = np.transpose(arr, axes=[2,0,1])
	#sitk.WriteImage(sitk.GetImageFromArray(sagital), nii_file[:-6] + 'mhd')
	sitk.WriteImage(img, nii_file[:-6] + 'mhd')
	print(nii_file[:-6] + 'mhd')

