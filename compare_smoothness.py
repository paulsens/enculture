import nibabel as nib
import numpy as np


dir0_path = "/Volumes/External/enculture/ROItesting/proj-642a49df73d2685502e73bfb/sub-sid001401.ses-A005515/dt-neuro-rois.run-1.id-64385323a6409dfa9289a2d3/rois/ROI1.nii.gz"
dir1_path = "/Volumes/External/enculture/ROItesting/proj-642a49df73d2685502e73bfb/sub-sid001401.ses-A005515/dt-neuro-rois.run-1.id-6436d6a1a6409dfa9250b6ce/rois/ROI1.nii.gz"

run1base_path = "/Volumes/External/enculture/temp/proj-642a49df73d2685502e73bfb/sub-sid001401.ses-A005515/dt-neuro-func-task.tag-enculture.run-1.id-642a4a0773d2685502e73dc8/bold.nii.gz"

run1preproc_path = "/Volumes/External/enculture/temp/proj-642a49df73d2685502e73bfb/sub-sid001401.ses-A005515/dt-neuro-func-task.tag-enculture.tag-preprocessed.run-1.id-642c4e035f63e51e545d8030/bold.nii.gz"

t1base_path = "/Volumes/External/enculture/temp/proj-642a49df73d2685502e73bfb/sub-sid001401.ses-A005515/dt-neuro-anat-t1w.id-642a49e873d2685502e73db9/t1.nii.gz"
t1preproc_path = "/Volumes/External/enculture/temp/proj-642a49df73d2685502e73bfb/sub-sid001401.ses-A005515/dt-neuro-anat-t1w.tag-preprocessed.run-1.id-642c4e055f63e51e545d8236/t1.nii.gz"

# t1base_img = nib.load(t1base_path)
# t1base_data = t1base_img.get_fdata()
#
# t1preproc_img = nib.load(t1preproc_path)
# t1preproc_data = t1preproc_img.get_fdata()
#
# this_data = t1preproc_data
# print(len(this_data))
# print(len(this_data[0]))
# print(len(this_data[0][0]))
# quit(0)
#
# run1base_img = nib.load(run1base_path)
# run1base_data = run1base_img.get_fdata()
#
# run1preproc_img = nib.load(run1preproc_path)
# run1preproc_data = run1preproc_img.get_fdata()
#
# this_data = run1preproc_data
# print(len(this_data))
# print(len(this_data[0]))
# print(len(this_data[0][0]))
# quit(0)

smooth0_img = nib.load(dir0_path)
smooth0_data = smooth0_img.get_fdata()

smooth1_img = nib.load(dir1_path)
smooth1_data = smooth1_img.get_fdata()

this_data = smooth0_data
print(len(this_data))
print(len(this_data[0]))
print(len(this_data[0][0]))
quit(0)

this_smooth = smooth0_data
other_smooth = smooth1_data
ones = 0
for x in range(0, len(this_smooth)):
    for y in range(0, len(this_smooth[0])):
        for z in range(0, len(this_smooth[0][0])):
            value = this_smooth[x][y][z]
            if value==1:
                ones+=1
                if other_smooth[x][y][z]!=1:
                    print("smooth0 at {0},{1},{2} was {3}".format(x,y,z,value))
                    print("smooth1 at {0},{1},{2} was {3}".format(x,y,z,other_smooth[x][y][z]))
            elif value!=0:
                print("wtf got "+str(value))
print("got {0} many ones".format(ones))
