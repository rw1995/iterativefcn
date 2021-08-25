import SimpleITK as sitk
import numpy as np

img = sitk.GetArrayFromImage(sitk.ReadImage("CaseLspine2.mhd"))
img = np.transpose(img, axes=[1,2,0])


sitk.WriteImage(sitk.GetImageFromArray(img), "CaseLspine2.nii")
