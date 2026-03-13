import random
import nibabel as nib

# cliplist = list(range(1,61))
# n_runs=4
#
# random.shuffle(cliplist)
# print([cliplist[i::n_runs] for i in range(n_runs)])
roidir = "/Volumes/External/enculture/temp/"
folders = ["roisrun1","roisrun2"]
rois = ["ROIfreesurfer-1030","ROIfreesurfer-2030","ROIfreesurfer-9000","ROIfreesurfer-9500","ROIsubcortical-26","ROIsubcortical-58"]

run1_dir = roidir+"roisrun1/"
run2_dir = roidir+"roisrun2/"

for roi in rois:
    roi1_path = run1_dir+roi+".nii.gz"
    roi2_path = run2_dir+roi+".nii.gz"

    roi1_img = nib.load(roi1_path)
    roi1_data = roi1_img.get_fdata()

    roi2_img = nib.load(roi2_path)
    roi2_data = roi2_img.get_fdata()

    for x in range(0, 193):
        for y in range(0, 229):
            for z in range(0, 193):
                roi1val = roi1_data[x][y][z]
                roi2val = roi2_data[x][y][z]

                if roi1val >= 0.23:



    for x in range(0, 193):
        for y in range(0, 229):
            for z in range(0, 193):
                roi1val = roi1_data[x][y][z]
                roi2val = roi2_data[x][y][z]

                if roi1val != roi2val:
                    print("at {0},{1},{2} we got {3} and {4}".format(x, y, z, roi1val, roi2val))


