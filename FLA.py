# FLA stands for First Level Analysis, including GLM and contrast analysis
# import nilearn
import numpy as np
from sklearn.utils import Bunch
import warnings
from nilearn.image import concat_imgs, mean_img, resample_img

# what is a bunch object? ok it's just a dictionary with field-like dot operator access in addition to the standard key access
# our bunches have func1:string list of paths to nifti files for session 1, func2: same as func1 for session 2
                    # trials_ses1: string list to onset files for session 1, should just be events.tsv
                    # trials_ses2: same as above for session 2
                    # anat: string, path to anat file, should this be the preprocessed version?
template1 = "/Volumes/External/enculture/preproc/sub-sid002548.ses-A005505/dt-neuro-func-task.tag-enculture.tag-preprocessed.run-"
template2 = "/Volumes/External/enculture/preproc/sub-sid002548.ses-A005540/dt-neuro-func-task.tag-enculture.tag-preprocessed.run-"
sess1_paths = []
sess2_paths = []

for run in range(1,9):
    runpath1 = template1+str(run)+"/bold.nii.gz"
    runpath2 = template2+str(run)+"/bold.nii.gz"
    sess1_paths.append(runpath1)
    sess2_paths.append(runpath2)


fmri_img = [
    concat_imgs(sess1_paths, auto_resample=False),
    concat_imgs(sess2_paths, auto_resample=False)
]
