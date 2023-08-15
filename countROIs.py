import random
import nibabel as nib
import numpy as np
import os,sys
import pickle

enculture_sessions = ["sub-sid001401.ses-A005515","sub-sid001401.ses-A005552","sub-sid002548.ses-A005505","sub-sid002548.ses-A005540","sub-sid002564.ses-A005538","sub-sid002564.ses-A005567","sub-sid002566.ses-A005542","sub-sid002566.ses-A005572","sub-sid002589.ses-A005590","sub-sid002589.ses-A005615"]

genre_sessions = ["sub-001","sub-002", "sub-003","sub-004","sub-005"]

def count_binarymask():
    #roifile = "/Volumes/External/enculture/preproc/sub-sid001401.ses-A005515/rois.tag-warped/rois/ROIfreesurfer-1030.nii.gz"
    roifile = "/Volumes/External/enculture/preproc/sub-sid001401.ses-A005515/rois.tag-warped/rois/ROIsubcortical-58.nii.gz"
    roi_img = nib.load(roifile)
    roi_data = roi_img.get_fdata()

    roi_indices, roi_counts = np.unique(roi_data, return_counts=True)
    print(roi_indices, roi_counts)

atlasROI_path = "/Volumes/External/enculture/ROIs/"
def search_3d(data, threshold):
    width = len(data)
    height = len(data[0])
    depth = len(data[0][0])
    count =  0
    locations=[]

    # these are probabilities so we need to compare everything to the threshold
    for x in range(0, width):
        for y in range(0, height):
            for z in range(0, depth):
                value = data[x][y][z]

                if value > threshold:
                    count+=1
                    location = (x,y,z)
                    locations.append(location)

    return count, locations

def remove_overlap(list_of_lists):
    location_dict = {}
    for location_list in list_of_lists:
        for location in location_list:
            if location not in location_dict:
                location_dict[location] = 1
            else:
                location_dict[location] += 1
    unique_locations = []
    for coords in location_dict.keys():
        unique_locations.append(coords)

    overlapping_locations = []
    for coords in location_dict.keys():
        if location_dict[coords] >= len(list_of_lists):
            overlapping_locations.append(coords)
    return unique_locations, overlapping_locations, location_dict

# a midpoint to split the bilateral coordinates, the coords list passed in should have no repetitions
def lateralize(midpoint, coords):
    left_brain = []
    right_brain = []

    # if x coordinate is >= midpoint then it's in left brain (unintuitive based on FSL eyes but triple confirmed to be correct)
    # else right brain

    for coord in coords:
        x_coor = coord[0]
        if x_coor < midpoint:
            left_brain.append(x_coor)

        else:
            right_brain.append(x_coor)

    return left_brain, right_brain



def count_NAcc_fromatlas(threshold, midpoint):

    nacc_left = "nacc_left_HO.nii.gz"
    nacc_right = "nacc_right_HO.nii.gz"

    leftfile = atlasROI_path+nacc_left
    leftimg = nib.load(leftfile)
    leftdata = leftimg.get_fdata()
    rightfile = atlasROI_path+nacc_right
    rightimg = nib.load(rightfile)
    rightdata = rightimg.get_fdata()

    leftcount, leftlocations = search_3d(leftdata, threshold)
    rightcount, rightlocations = search_3d(rightdata, threshold)
    unique, overlapping, _ = remove_overlap([leftlocations, rightlocations])
    left_brain, right_brain = lateralize(midpoint, unique)

    print("NAcc voxels for threshold "+str(threshold)+":")
    print("Left: "+str(leftcount))
    print("Right: "+str(rightcount))
    print("Total: "+str(leftcount+rightcount))
    print("Total with repeats removed: "+str(len(unique)))
    print("-> Left Brain: "+str(len(left_brain)))
    print("-> Right Brain: "+str(len(right_brain)))


def count_STG_fromatlas(threshold, midpoint):

    stg_ant = "stg_ant_HO.nii.gz"
    stg_post = "stg_post_HO.nii.gz"

    antfile = atlasROI_path+stg_ant
    antimg = nib.load(antfile)
    antdata = antimg.get_fdata()
    postfile = atlasROI_path+stg_post
    postimg = nib.load(postfile)
    postdata = postimg.get_fdata()

    antcount, antlocations = search_3d(antdata, threshold)
    postcount, postlocations = search_3d(postdata, threshold)
    unique, overlapping, _ = remove_overlap([antlocations, postlocations])
    left_brain, right_brain = lateralize(midpoint, unique)

    print("STG voxels for threshold "+str(threshold)+":")
    print("Anterior: "+str(antcount))
    print("Posterior: "+str(postcount))
    print("Total: "+str(antcount+postcount))
    print("Total with repeats removed: "+str(len(unique)))
    print("-> Left Brain: "+str(len(left_brain)))
    print("-> Right Brain: "+str(len(right_brain)))

def count_STG_warped(threshold, midpoint, include_genre):
    # path is /Volumes/External/enculture_or_genrenew/preproc/sessiondir/rois.tag-warped/rois/
    leftROI = "ROIfreesurfer-1030.nii.gz"
    rightROI = "ROIfreesurfer-2030.nii.gz"

    leftcounts = []
    rightcounts = []
    leftlocs = [] # each element of these two lists is the locations for an entire session
    rightlocs = []

    # loop through enculture sessions
    for session in enculture_sessions:
        roidir = "/Volumes/External/enculture/preproc/"+session+"/rois.tag-warped/rois/"

        leftfile = roidir+leftROI
        leftimg = nib.load(leftfile)
        leftdata = leftimg.get_fdata()
        rightfile = roidir+rightROI
        rightimg = nib.load(rightfile)
        rightdata = rightimg.get_fdata()

        # threshold doesn't matter here because these are binary masks
        leftcount, leftlocations = search_3d(leftdata, threshold)
        leftcounts.append(leftcount)
        leftlocs.append(leftlocations)

        rightcount, rightlocations = search_3d(rightdata, threshold)
        rightcounts.append(rightcount)
        rightlocs.append(rightlocations)


    if include_genre:
        for session in genre_sessions:
            roidir = "/Volumes/External/genrenew/preproc/"+session+"/rois.tag-warped/rois/"

            leftfile = roidir+leftROI
            leftimg = nib.load(leftfile)
            leftdata = leftimg.get_fdata()
            rightfile = roidir+rightROI
            rightimg = nib.load(rightfile)
            rightdata = rightimg.get_fdata()

            # threshold doesn't matter here because these are binary masks
            leftcount, leftlocations = search_3d(leftdata, threshold)
            leftcounts.append(leftcount)
            leftlocs.append(leftlocations)

            rightcount, rightlocations = search_3d(rightdata, threshold)
            rightcounts.append(rightcount)
            rightlocs.append(rightlocations)

    leftunique, leftoverlapping, _ = remove_overlap(leftlocs)
    rightunique, rightoverlapping, _ = remove_overlap(rightlocs)

    print("For STG with the warped ROIS:")
    print("Left brain has the following voxel counts (enculture 10, 2 per subject, then genre 5):")
    print(leftcounts)
    print("The union of them all has "+str(len(leftunique))+" voxels.")
    print("The intersection of them all has "+str(len(leftoverlapping))+" voxels.")
    print()
    print("Right brain has the following voxel counts (enculture 10, 2 per subject, then genre 5):")
    print(rightcounts)
    print("The union of them all has "+str(len(rightunique))+" voxels.")
    print("The intersection of them all has "+str(len(rightoverlapping))+" voxels.")


def count_NAcc_warped(threshold, midpoint, include_genre):
    # path is /Volumes/External/enculture_or_genrenew/preproc/sessiondir/rois.tag-warped/rois/
    leftROI = "ROIsubcortical-26.nii.gz"
    rightROI = "ROIsubcortical-58.nii.gz"

    leftcounts = []
    rightcounts = []
    leftlocs = []  # each element of these two lists is the locations for an entire session
    rightlocs = []

    # loop through enculture sessions
    for session in enculture_sessions:
        roidir = "/Volumes/External/enculture/preproc/" + session + "/rois.tag-warped/rois/"

        leftfile = roidir + leftROI
        leftimg = nib.load(leftfile)
        leftdata = leftimg.get_fdata()
        rightfile = roidir + rightROI
        rightimg = nib.load(rightfile)
        rightdata = rightimg.get_fdata()

        # threshold doesn't matter here because these are binary masks
        leftcount, leftlocations = search_3d(leftdata, threshold)
        leftcounts.append(leftcount)
        leftlocs.append(leftlocations)

        rightcount, rightlocations = search_3d(rightdata, threshold)
        rightcounts.append(rightcount)
        rightlocs.append(rightlocations)

    if include_genre:
        for session in genre_sessions:
            roidir = "/Volumes/External/genrenew/preproc/" + session + "/rois.tag-warped/rois/"

            leftfile = roidir + leftROI
            leftimg = nib.load(leftfile)
            leftdata = leftimg.get_fdata()
            rightfile = roidir + rightROI
            rightimg = nib.load(rightfile)
            rightdata = rightimg.get_fdata()

            # threshold doesn't matter here because these are binary masks
            leftcount, leftlocations = search_3d(leftdata, threshold)
            leftcounts.append(leftcount)
            leftlocs.append(leftlocations)

            rightcount, rightlocations = search_3d(rightdata, threshold)
            rightcounts.append(rightcount)
            rightlocs.append(rightlocations)

    leftunique, leftoverlapping, loc_dictleft = remove_overlap(leftlocs)
    rightunique, rightoverlapping, loc_dictright = remove_overlap(rightlocs)

    print("For NAcc with the warped ROIS:")
    print("Left brain has the following voxel counts (enculture 10, 2 per subject, then genre 5):")
    print(leftcounts)
    print("The union of them all has " + str(len(leftunique)) + " voxels.")
    print("The intersection of them all has " + str(len(leftoverlapping)) + " voxels.")
    print()
    print("Right brain has the following voxel counts (enculture 10, 2 per subject, then genre 5):")
    print(rightcounts)
    print("The union of them all has " + str(len(rightunique)) + " voxels.")
    print("The intersection of them all has " + str(len(rightoverlapping)) + " voxels.")
    print("Length of loc_dictleft is "+str(len(loc_dictleft.keys())))
    print("Length of loc_dictright is "+str(len(loc_dictright.keys())))
    print("Width is "+str(len(rightdata)))
    print("Height is "+str(len(rightdata[0])))
    print("Depth is "+str(len(rightdata[0][0])))
    # now loop over the space and check if the coordinates are in the dict (since it's O(1))
    # a good question to ask is whether to go all the way over x as we flatten or concatenate the two flattened halves
    flattened = []
    width = len(rightdata)
    height = len(rightdata[0])
    depth = len(rightdata[0][0])
    for x in range(0, midpoint):
        for y in range(0, height):
            for z in range(0, depth):
                temp_coord = (x,y,z)
                if temp_coord in loc_dictleft.keys():
                    flattened.append(temp_coord)
    for x in range(midpoint, width):
        for y in range(0, height):
            for z in range(0, depth):
                temp_coord = (x,y,z)
                if temp_coord in loc_dictright.keys():
                    flattened.append(temp_coord)

    print("Flattened NAcc warped Union ROI has "+str(len(flattened))+" voxels.")
    # the lists of coordinates were already 1D, so maybe calling this flattened is imprecise
    # what this really is, is a list of coordinates ordered such that iterating through and grabbing values from a nifti file
    #    will perform the desired flattening of the 3d image. at the same time, order really shouldn't matter since all the components
    #    of a transformer are agnostic to the order of dimensions. there are no filters commonly seen in CNNs for example

    with open("/Volumes/External/enculture/ROIs/NAccWarpedFlatUnionROI.p", "wb") as roi_fp:
        pickle.dump(flattened, roi_fp)


include_genre = True # sometimes we might want to just count the enculturation ROIs out of curiosity
thresh = 23
midpoint = 48 # the left-and-right (i.e the subject's left and right sides) axis has length 97 in our MNI space, so 48 is the midpoint
            # we arbitrarily choose to include voxels with x coordinate 48 in the right side, but this shouldn't affect anything.
# count_NAcc_fromatlas(thresh, midpoint)
# count_STG_fromatlas(thresh, midpoint)
# count_STG_warped(0, midpoint, include_genre)
count_NAcc_warped(0, midpoint, include_genre)

