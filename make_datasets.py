import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import itertools
import os
import pickle
import pandas as pd
from random import randint
from datetime import date
from torch.utils.data import Dataset
from helpers import standardize_flattened, detrend_flattened
import random
import copy as coppy

enculture_subs = ["sub-sid001401.ses-A005515","sub-sid001401.ses-A005552","sub-sid002548.ses-A005505","sub-sid002548.ses-A005540","sub-sid002564.ses-A005538","sub-sid002564.ses-A005567","sub-sid002566.ses-A005542","sub-sid002566.ses-A005572","sub-sid002589.ses-A005590","sub-sid002589.ses-A005615"]
enculture_sessions = {"sub-sid001401":["ses-A005515", "ses-A005552"],
                      "sub-sid002548":["ses-A005505", "ses-A005540"],
                      "sub-sid002564":["ses-A005538", "ses-A005567"],
                      "sub-sid002566":["ses-A005542", "ses-A005572"],
                      "sub-sid002589":["ses-A005590", "ses-A005615"]
                    }

enculture_holdouts = {"sub-sid001401":3,
                      "sub-sid002548":7,
                      "sub-sid002564":6,
                      "sub-sid002566":4,
                      "sub-sid002589":1
                      }
# the lists in these dictionaries are the TRIAL NUMBERS (in blocks of 3) corresponding to each condition. The numbers for train index into the compiled list with the validation run already removed.
# similarly, the numbers for val index into the held out run on its own
sid1401_conditions = {"bach_train":[0, 2, 4, 6, 7, 8, 15, 16, 17, 18, 20, 23, 24, 25],
                       "bach_val":[0, 2],
                       "shanxi_train":[1, 3, 5, 9, 10, 11, 12, 13, 14, 19, 21, 22, 26, 27],
                       "shanxi_val":[1, 3],
                      "heldout_run":3
}

sid2548_conditions = {"bach_train":[0, 1, 2, 3, 4, 6, 14, 15, 16, 17, 18, 19, 23, 25],
                       "bach_val":[0, 3],
                       "shanxi_train":[5, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 24, 26, 27],
                       "shanxi_val":[1, 2],
                      "heldout_run":7
}

sid2564_conditions = {"bach_train":[0, 4, 5, 6, 7, 10, 11, 12, 13, 14, 19, 20, 24, 26],
                       "bach_val":[2, 3],
                       "shanxi_train":[1, 2, 3, 8, 9, 15, 16, 17, 18, 21, 22, 23, 25, 27],
                       "shanxi_val":[0, 1],
                      "heldout_run":6
}

sid2566_conditions = {"bach_train":[0, 1, 8, 9, 10, 11, 12, 15, 17, 20, 21, 22, 24, 25],
                       "bach_val":[1, 2],
                       "shanxi_train":[2, 3, 4, 5, 6, 7, 13, 14, 16, 18, 19, 23, 26, 27],
                       "shanxi_val":[0, 3],
                      "heldout_run":4
}

sid2589_conditions = {"bach_train":[0, 3, 5, 7, 8, 10, 12, 14, 17, 18, 22, 24, 25, 26],
                       "bach_val":[1, 3],
                       "shanxi_train":[1, 2, 4, 6, 9, 11, 13, 15, 16, 19, 20, 21, 23, 27],
                       "shanxi_val":[0, 2],
                      "heldout_run":1
}

enculture_conditions = {"sub-sid001401":sid1401_conditions,
                        "sub-sid002548":sid2548_conditions,
                        "sub-sid002564":sid2564_conditions,
                        "sub-sid002566":sid2566_conditions,
                        "sub-sid002589":sid2589_conditions
}

genre_subs = ["sub-001","sub-002", "sub-003","sub-004","sub-005"]


# makes NTP and/or SameSession from the enculture dataset
# 30 TRs (45s) per trial, 12 trials per run (360 TRs, 540s), 8 runs per session (2880 TRs, 4320s)
# note that dummy TRs have already been removed from all runs by detrendstandardize.py
def make_Enc(CLS, SEP, ROI_name, task="both"):

    if ROI_name=="NAccUnion":
        allruns_name = "allruns_NAccUnion_detrendedstandardized.p"
        ROIvoxels = 417 # this is known a priori from creating the ROI
        reservedDims = 3 # CLS, SEP, and MSK, even though we aren't using MSK right now.
        TRsperrun = 360
        seq_len = 5
        NTP_stride = 2
        SS_stride = 5
    # possible elif cases in the future

    X_train_NTP = []
    y_train_NTP = []
    X_val_NTP = []
    y_val_NTP = []

    bach_X_train_SS = []
    bach_y_train_SS = []
    bach_X_val_SS = []
    bach_y_val_SS = []
    shanxi_X_train_SS = []
    shanxi_y_train_SS = []
    shanxi_X_val_SS = []
    shanxi_y_val_SS = []

    for subid in enculture_sessions.keys():
        session_list = enculture_sessions[subid]
        session1 = session_list[0]
        session2 = session_list[1]
        heldout_run = enculture_holdouts[subid] # the same run is held out for both sessions of each subject
        heldout_startTR = (heldout_run-1)*TRsperrun
        heldout_endTR = heldout_startTR + TRsperrun #technically the last included TR+1

        allruns1_path = "/Volumes/External/enculture/preproc/"+subid+"."+session1+"/"+allruns_name
        allruns2_path = "/Volumes/External/enculture/preproc/"+subid+"."+session2+"/"+allruns_name

        with open(allruns1_path, "rb") as allruns1_fp:
            allruns_data1 = pickle.load(allruns1_fp)
        with open(allruns2_path, "rb") as allruns2_fp:
            allruns_data2 = pickle.load(allruns2_fp)

        # so allruns has length 2880. Each run is 360 TRs.
        training_data1 = []
        heldout_data1 = []
        training_data2 = []
        heldout_data2 = []

        for training_TR in range(0, heldout_startTR):
            training_data1.append(allruns_data1[training_TR])
        for training_TR in range(heldout_endTR, 2880):
            training_data1.append(allruns_data1[training_TR])
        for heldout_TR in range(heldout_startTR, heldout_endTR):
            heldout_data1.append(allruns_data1[heldout_TR])

        for training_TR in range(0, heldout_startTR):
            training_data2.append(allruns_data2[training_TR])
        for training_TR in range(heldout_endTR, 2880):
            training_data2.append(allruns_data2[training_TR])
        for heldout_TR in range(heldout_startTR, heldout_endTR):
            heldout_data2.append(allruns_data2[heldout_TR])


        if task in ["NTP","both"]:
            X_train1, y_train1 = make_NTP(training_data1, 7, TRsperrun, seq_len, NTP_stride, CLS, SEP) # pass in the collection of data we're using for training, which is comprised of 7 runs
            X_val1, y_val1 = make_NTP(heldout_data1, 1, TRsperrun, seq_len, NTP_stride, CLS, SEP)
            X_train_NTP.extend(X_train1)
            y_train_NTP.extend(y_train1)
            X_val_NTP.extend(X_val1)
            y_val_NTP.extend(y_val1)

            X_train2, y_train2 = make_NTP(training_data2, 7, TRsperrun, seq_len, NTP_stride, CLS, SEP) # pass in the collection of data we're using for training, which is comprised of 7 runs
            X_val2, y_val2 = make_NTP(heldout_data2, 1, TRsperrun, seq_len, NTP_stride, CLS, SEP)
            X_train_NTP.extend(X_train2)
            y_train_NTP.extend(y_train2)
            X_val_NTP.extend(X_val2)
            y_val_NTP.extend(y_val2)


        if task in ["SameSession", "both"]:
            bach_X_train, bach_y_train, shanxi_X_train, shanxi_y_train = make_SS(subid, training_data1, training_data2, 7, TRsperrun, seq_len, SS_stride, CLS, SEP)
            bach_X_val, bach_y_val, shanxi_X_val, shanxi_y_val = make_SS(subid, heldout_data1, heldout_data2, 1, TRsperrun, seq_len, SS_stride, CLS, SEP)

            bach_X_train_SS.extend(bach_X_train)
            bach_y_train_SS.extend(bach_y_train)
            bach_X_val_SS.extend(bach_X_val)
            bach_y_val_SS.extend(bach_y_val)

            shanxi_X_train_SS.extend(shanxi_X_train)
            shanxi_y_train_SS.extend(shanxi_y_train)
            shanxi_X_val_SS.extend(shanxi_X_val)
            shanxi_y_val_SS.extend(shanxi_y_val)

    # all sessions done, save the data for NTP
    if task in ["NTP", "both"]:
        with open("/Volumes/External/enculture/preproc/datasets/X_trainNTP_fold" + str(heldout_run) + ".p", "wb") as fold_fp:
            pickle.dump(X_train_NTP, fold_fp)
            print("Saved X_train_NTP for fold " + str(heldout_run) + " with length " + str(len(X_train_NTP)))

        with open("/Volumes/External/enculture/preproc/datasets/y_trainNTP_fold" + str(heldout_run) + ".p", "wb") as fold_fp:
            pickle.dump(y_train_NTP, fold_fp)
            print("Saved y_train_NTP for fold " + str(heldout_run) + " with length " + str(len(y_train_NTP)))

        with open("/Volumes/External/enculture/preproc/datasets/X_valNTP_fold" + str(heldout_run) + ".p", "wb") as fold_fp:
            pickle.dump(X_val_NTP, fold_fp)
            print("Saved X_val_NTP for fold " + str(heldout_run) + " with length " + str(len(X_val_NTP)))

        with open("/Volumes/External/enculture/preproc/datasets/y_valNTP_fold" + str(heldout_run) + ".p", "wb") as fold_fp:
            pickle.dump(y_val_NTP, fold_fp)
            print("Saved y_val_NTP for fold " + str(heldout_run) + " with length " + str(len(y_val_NTP)))

    # now save for SS
    if task in ["SameSession", "both"]:
        # save for bach
        with open("/Volumes/External/enculture/preproc/datasets/bach_X_trainSS.p", "wb") as fold_fp:
            pickle.dump(bach_X_train_SS, fold_fp)
            print("Saved bach_X_train with length " + str(len(bach_X_train_SS)))
        with open("/Volumes/External/enculture/preproc/datasets/bach_y_trainSS.p", "wb") as fold_fp:
            pickle.dump(bach_y_train_SS, fold_fp)
            print("Saved bach_y_train with length " + str(len(bach_y_train_SS)))
        with open("/Volumes/External/enculture/preproc/datasets/bach_X_valSS.p", "wb") as fold_fp:
            pickle.dump(bach_X_val_SS, fold_fp)
            print("Saved bach_X_val with length " + str(len(bach_X_val_SS)))
        with open("/Volumes/External/enculture/preproc/datasets/bach_y_valSS.p", "wb") as fold_fp:
            pickle.dump(bach_y_val_SS, fold_fp)
            print("Saved bach_y_val with length " + str(len(bach_y_val_SS)))

        # save for shanxi
        with open("/Volumes/External/enculture/preproc/datasets/shanxi_X_trainSS.p", "wb") as fold_fp:
            pickle.dump(shanxi_X_train_SS, fold_fp)
            print("Saved shanxi_X_train with length " + str(len(shanxi_X_train_SS)))
        with open("/Volumes/External/enculture/preproc/datasets/shanxi_y_trainSS.p", "wb") as fold_fp:
            pickle.dump(shanxi_y_train_SS, fold_fp)
            print("Saved shanxi_y_train with length " + str(len(shanxi_y_train_SS)))
        with open("/Volumes/External/enculture/preproc/datasets/shanxi_X_valSS.p", "wb") as fold_fp:
            pickle.dump(shanxi_X_val_SS, fold_fp)
            print("Saved shanxi_X_val with length " + str(len(shanxi_X_val_SS)))
        with open("/Volumes/External/enculture/preproc/datasets/shanxi_y_valSS.p", "wb") as fold_fp:
            pickle.dump(shanxi_y_val_SS, fold_fp)
            print("Saved shanxi_y_val with length " + str(len(shanxi_y_val_SS)))

# makes NTP dataset from the music genre dataset
# 400 TRs per run
def make_Genre(CLS, SEP, ROI_name):

    if ROI_name=="NAccUnion":
        allruns_name = "allruns_NAccUnion_detrendedstandardized.p"
        ROIvoxels = 417 # this is known a priori from creating the ROI
        reservedDims = 3 # CLS, SEP, and MSK, even though we aren't using MSK right now.
        TRsperrun = 400
        seq_len = 5
        stride = 2
    # possible elif cases in the future

    # let's do 12 fold crossval again, each fold holding out one of the 12 training runs
    for heldout_run in range(1, 13):
        X_train_NTP = []
        y_train_NTP = []
        X_val_NTP = []
        y_val_NTP = []


        for session in genre_subs:


            allruns_path = "/Volumes/External/genrenew/preproc/"+session+"/"+allruns_name

            with open(allruns_path, "rb") as allruns_fp:
                allrunswithtest_data = pickle.load(allruns_fp)

            # cut out the test runs because fuck it
            allruns_data = allrunswithtest_data[2400:]
            allruns_data = np.array(allruns_data)
            # now the holdout indices start where they should
            heldout_startTR = (heldout_run-1)*TRsperrun
            heldout_endTR = heldout_startTR + TRsperrun #technically the last included TR+1
            # so allruns has length 4800. Each run is 400 TRs.
            training_data = []
            heldout_data = []


            for training_TR in range(0, heldout_startTR):
                training_data.append(allruns_data[training_TR])
            for training_TR in range(heldout_endTR, 4800):
                training_data.append(allruns_data[training_TR])
            for heldout_TR in range(heldout_startTR, heldout_endTR):
                heldout_data.append(allruns_data[heldout_TR])


            X_train, y_train = make_NTP(training_data, 11, TRsperrun, seq_len, stride, CLS, SEP) # pass in the collection of data we're using for training, which is comprised of 7 runs
            X_val, y_val = make_NTP(heldout_data, 1, TRsperrun, seq_len, stride, CLS, SEP)
            X_train_NTP.extend(X_train)
            y_train_NTP.extend(y_train)
            X_val_NTP.extend(X_val)
            y_val_NTP.extend(y_val)

        # all sessions done, save the data for this fold
        with open("/Volumes/External/genrenew/preproc/datasets/X_train_fold"+str(heldout_run)+".p", "wb") as fold_fp:
            pickle.dump(X_train_NTP, fold_fp)
            print("Saved X_train_NTP for fold "+str(heldout_run)+" with length "+str(len(X_train_NTP)))

        with open("/Volumes/External/genrenew/preproc/datasets/y_train_fold"+str(heldout_run)+".p", "wb") as fold_fp:
            pickle.dump(y_train_NTP, fold_fp)
            print("Saved y_train_NTP for fold "+str(heldout_run)+" with length "+str(len(y_train_NTP)))

        with open("/Volumes/External/genrenew/preproc/datasets/X_val_fold"+str(heldout_run)+".p", "wb") as fold_fp:
            pickle.dump(X_val_NTP, fold_fp)
            print("Saved X_val_NTP for fold "+str(heldout_run)+" with length "+str(len(X_val_NTP)))

        with open("/Volumes/External/genrenew/preproc/datasets/y_val_fold"+str(heldout_run)+".p", "wb") as fold_fp:
            pickle.dump(y_val_NTP, fold_fp)
            print("Saved y_val_NTP for fold "+str(heldout_run)+" with length "+str(len(y_val_NTP)))


def make_NTP(voxel_data, num_runs, TRsperrun, seq_len, stride, CLS, SEP):
    # we don't want positive NTP pairs to cross run boundaries, so let's do each run as a chunk
    X = [] # samples
    y = [] # labels
    for chunk in range(0, num_runs):
        chunk_start = TRsperrun*chunk
        chunk_end = chunk_start + TRsperrun
        last_sample_start = chunk_end - seq_len*2 # this avoids having seq2 cross over into the next run

        for sample_start in range(chunk_start, last_sample_start+1, stride):
            # create positive sample starting at sample_start
            temp_pos = [coppy.deepcopy(CLS)]
            for seq1_idx in range(sample_start, sample_start + seq_len):
                temp_pos.append(voxel_data[seq1_idx])
            temp_pos.append(coppy.deepcopy(SEP))
            for seq2_idx in range(sample_start+seq_len, sample_start+seq_len+seq_len):
                temp_pos.append(voxel_data[seq2_idx])
            X.append(temp_pos)
            y.append([0,1]) # True, yes it is the next thought

            # create negative sample starting at sample_start
            # choose a place for seq2 to begin, doesn't need to be in this chunk
            seq2_start = randint(0,len(voxel_data)-seq_len)
            nearby = list(range(sample_start-10, sample_start+10)) # let's keep the negative partner from being too near to seq2
            while seq2_start in nearby: # find a seq2_start that's not nearby
                seq2_start = randint(0, len(voxel_data) - seq_len)
            temp_neg = [coppy.deepcopy(CLS)]
            for seq1_idx in range(sample_start, sample_start+seq_len):
                temp_neg.append(voxel_data[seq1_idx])
            temp_neg.append(coppy.deepcopy(SEP))
            for seq2_idx in range(seq2_start, seq2_start+seq_len):
                temp_neg.append(voxel_data[seq2_idx])
            X.append(temp_neg)
            y.append([1,0]) # False, no it is not the next thought

    return X, y

def make_SS(subid, voxel_data1, voxel_data2, num_runs, TRsperrun, seq_len, stride, CLS, SEP):
    X = []  # samples
    y = []  # labels
    # each block has the same condition on its three trials, let's write out the start points for each block of 90 TRs (30*3)
    block_startpoints = [4, 9, 14, 19, 24, 34, 39, 44, 49, 54, 64, 69, 74, 79, 84]
    this_sub_conditions = enculture_conditions[subid]
    if num_runs == 1:
        bach_blocks = this_sub_conditions["bach_val"]
        shanxi_blocks = this_sub_conditions["shanxi_val"]
    else:
        bach_blocks = this_sub_conditions["bach_train"]
        shanxi_blocks = this_sub_conditions["shanxi_train"]

    bach_samples = []
    bach_labels = []
    shanxi_samples = []
    shanxi_labels = []

    # samples  where left-seq is from voxel_data1
    for block_n in bach_blocks:
        for startpoint in block_startpoints:
            pos_sample = [coppy.deepcopy(CLS)]
            neg_sample = [coppy.deepcopy(CLS)]
            sample_start = 90*block_n + startpoint
            for TR in range(0, seq_len):
                pos_sample.append(voxel_data1[sample_start+TR])
                neg_sample.append(voxel_data1[sample_start+TR])
            pos_sample.append(coppy.deepcopy(SEP))
            neg_sample.append(coppy.deepcopy(SEP))

            # get partner for positive sample
            pos_partner_block = block_n
            while pos_partner_block == block_n:
                pos_partner_block = random.choice(bach_blocks)
            pos_partner_startpoint = random.choice(block_startpoints)
            pos_partner_startTR = pos_partner_block*90 + pos_partner_startpoint

            # get partner for negative sample
            neg_partner_block = block_n
            while neg_partner_block == block_n:
                neg_partner_block = random.choice(bach_blocks)
            neg_partner_startpoint = random.choice(block_startpoints)
            neg_partner_startTR = neg_partner_block*90 + neg_partner_startpoint

            for TR in range(0, seq_len):
                pos_sample.append(voxel_data1[pos_partner_startTR+TR])
                neg_sample.append(voxel_data2[neg_partner_startTR+TR])

            # pos_sample and neg_sample are filled, add them to running lists
            bach_samples.append(pos_sample)
            bach_labels.append([0,1])
            bach_samples.append(neg_sample)
            bach_labels.append([1,0])

    # samples  where left-seq is from voxel_data2
    for block_n in bach_blocks:
        for startpoint in block_startpoints:
            pos_sample = [coppy.deepcopy(CLS)]
            neg_sample = [coppy.deepcopy(CLS)]
            sample_start = 90*block_n + startpoint
            for TR in range(0, seq_len):
                pos_sample.append(voxel_data2[sample_start+TR])
                neg_sample.append(voxel_data2[sample_start+TR])
            pos_sample.append(coppy.deepcopy(SEP))
            neg_sample.append(coppy.deepcopy(SEP))

            # get partner for positive sample
            pos_partner_block = block_n
            while pos_partner_block == block_n:
                pos_partner_block = random.choice(bach_blocks)
            pos_partner_startpoint = random.choice(block_startpoints)
            pos_partner_startTR = pos_partner_block*90 + pos_partner_startpoint

            # get partner for negative sample
            neg_partner_block = block_n
            while neg_partner_block == block_n:
                neg_partner_block = random.choice(bach_blocks)
            neg_partner_startpoint = random.choice(block_startpoints)
            neg_partner_startTR = neg_partner_block*90 + neg_partner_startpoint

            for TR in range(0, seq_len):
                pos_sample.append(voxel_data2[pos_partner_startTR+TR])
                neg_sample.append(voxel_data1[neg_partner_startTR+TR])

            # pos_sample and neg_sample are filled, add them to running lists
            bach_samples.append(pos_sample)
            bach_labels.append([0,1])
            bach_samples.append(neg_sample)
            bach_labels.append([1,0])

    print("After getting left-seqs from both sessions, we have "+str(len(bach_samples))+" samples from the bach clips. Moving on to shanxi...")
    print("Each sample has size "+str(len(bach_samples[0])))
    print("Each element of the sample has size "+str(len(bach_samples[0][0])))

    # samples  where left-seq is from voxel_data1
    for block_n in shanxi_blocks:
        for startpoint in block_startpoints:
            pos_sample = [coppy.deepcopy(CLS)]
            neg_sample = [coppy.deepcopy(CLS)]
            sample_start = 90*block_n + startpoint
            for TR in range(0, seq_len):
                pos_sample.append(voxel_data1[sample_start+TR])
                neg_sample.append(voxel_data1[sample_start+TR])
            pos_sample.append(coppy.deepcopy(SEP))
            neg_sample.append(coppy.deepcopy(SEP))

            # get partner for positive sample
            pos_partner_block = block_n
            while pos_partner_block == block_n:
                pos_partner_block = random.choice(shanxi_blocks)
            pos_partner_startpoint = random.choice(block_startpoints)
            pos_partner_startTR = pos_partner_block*90 + pos_partner_startpoint

            # get partner for negative sample
            neg_partner_block = block_n
            while neg_partner_block == block_n:
                neg_partner_block = random.choice(shanxi_blocks)
            neg_partner_startpoint = random.choice(block_startpoints)
            neg_partner_startTR = neg_partner_block*90 + neg_partner_startpoint

            for TR in range(0, seq_len):
                pos_sample.append(voxel_data1[pos_partner_startTR+TR])
                neg_sample.append(voxel_data2[neg_partner_startTR+TR])

            # pos_sample and neg_sample are filled, add them to running lists
            shanxi_samples.append(pos_sample)
            shanxi_labels.append([0,1])
            shanxi_samples.append(neg_sample)
            shanxi_labels.append([1,0])

    # samples  where left-seq is from voxel_data2
    for block_n in shanxi_blocks:
        for startpoint in block_startpoints:
            pos_sample = [coppy.deepcopy(CLS)]
            neg_sample = [coppy.deepcopy(CLS)]
            sample_start = 90*block_n + startpoint
            for TR in range(0, seq_len):
                pos_sample.append(voxel_data2[sample_start+TR])
                neg_sample.append(voxel_data2[sample_start+TR])
            pos_sample.append(coppy.deepcopy(SEP))
            neg_sample.append(coppy.deepcopy(SEP))

            # get partner for positive sample
            pos_partner_block = block_n
            while pos_partner_block == block_n:
                pos_partner_block = random.choice(shanxi_blocks)
            pos_partner_startpoint = random.choice(block_startpoints)
            pos_partner_startTR = pos_partner_block*90 + pos_partner_startpoint

            # get partner for negative sample
            neg_partner_block = block_n
            while neg_partner_block == block_n:
                neg_partner_block = random.choice(shanxi_blocks)
            neg_partner_startpoint = random.choice(block_startpoints)
            neg_partner_startTR = neg_partner_block*90 + neg_partner_startpoint

            for TR in range(0, seq_len):
                pos_sample.append(voxel_data2[pos_partner_startTR+TR])
                neg_sample.append(voxel_data1[neg_partner_startTR+TR])

            # pos_sample and neg_sample are filled, add them to running lists
            shanxi_samples.append(pos_sample)
            shanxi_labels.append([0,1])
            shanxi_samples.append(neg_sample)
            shanxi_labels.append([1,0])

    print("After getting left-seqs from both sessions, we have "+str(len(shanxi_samples))+" samples from the shanxi clips.")
    print("Each sample has size "+str(len(shanxi_samples[0])))

    return bach_samples, bach_labels, shanxi_samples, shanxi_labels

# define parameters and call the making functions
if __name__ == '__main__':

    voxel_dim = 420
    CLS = [1] + ([0] * (voxel_dim - 1))  # first dimension is reserved for cls_token flag
    MSK = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
    SEP = [0, 0, 1] + ([0] * (voxel_dim - 3))  # third dimension is reserved for sep_token flag
    # this ROI needs to be a flattened list of coordinates to loop through, rather than a 3D binary mask.
    # such a list should be obtained from countROIs.py
    ROI_name = "NAccUnion"
    Enc_task = "SameSession" # task is one of [NTP, SameSession, both]


    # comment or uncomment and/or change the Enc task as desired
    make_Enc(CLS, SEP, ROI_name=ROI_name, task=Enc_task)
    #make_Genre(CLS, SEP, ROI_name=ROI_name) # currently only does NTP