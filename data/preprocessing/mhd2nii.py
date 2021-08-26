import SimpleITK as sitk
import numpy as np
import glob

mhd_files = glob.glob("./img/*.mhd")

for mhd_file in mhd_files:
	img = sitk.GetArrayFromImage(sitk.ReadImage(mhd_file))
	img = np.transpose(img, axes=[1,2,0])


	sitk.WriteImage(sitk.GetImageFromArray(img), mhd_file[:-3] + 'nii')
	print(mhd_file)
