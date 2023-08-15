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

enculture_sessions = ["sub-sid001401.ses-A005515","sub-sid001401.ses-A005552","sub-sid002548.ses-A005505","sub-sid002548.ses-A005540","sub-sid002564.ses-A005538","sub-sid002564.ses-A005567","sub-sid002566.ses-A005542","sub-sid002566.ses-A005572","sub-sid002589.ses-A005590","sub-sid002589.ses-A005615"]
enculture_subs = ["sub-sid001401", "sub-sid002548", "sub-sid002564", "sub-sid002566", "sub-sid002589"]
enculture_session_dict = {"sub-sid001401":["ses-A005515", "ses-A005552"],
                      "sub-sid002548":["ses-A005505", "ses-A005540"],
                      "sub-sid002564":["ses-A005538", "ses-A005567"],
                      "sub-sid002566":["ses-A005542", "ses-A005572"],
                      "sub-sid002589":["ses-A005590", "ses-A005615"]
}
genre_subs = ["sub-001","sub-002", "sub-003","sub-004","sub-005"]

ROI_path = "/Volumes/External/enculture/ROIs/NAccWarpedFlatUnionROI.p" # make sure to set this for the desired ROI, note this is not a binary mask but a list of coordinates
ROI_name = "NAccUnion"
with open(ROI_path, "rb") as ROI_fp:
    ROI_coordinates = pickle.load(ROI_fp)
do_enculturation = False
do_genre = True
# =============== Ready to start the work =================
# for each session, this code first excises the ROI from each TR of a run, concatenates all the excised runs, then detrends and standardizes.
# that data is then saved to be used by make_datasets.py
if do_enculturation:
    enculture_sessions = ["sub-sid002566.ses-A005542","sub-sid002566.ses-A005572","sub-sid002589.ses-A005590","sub-sid002589.ses-A005615"]
    for session in enculture_sessions:
        base_path = "/Volumes/External/enculture/preproc/"
        session_dir = base_path+session+"/"
        all_runs_data = []

        for run_n in range(1, 9):
            run_dir = session_dir+"func-task.tag-enculture.tag-preprocessed.run-"+str(run_n)+"/"
            run_img = nib.load(run_dir+"bold.nii.gz")
            run_data = run_img.get_fdata()

            # run_data is 97x115x97x362. recall that the first two TRs of each run are dummy data.
            start_TR=2
            end_TR = len(run_data[0][0][0])

            for TR in range(start_TR, end_TR):
                if ROI_name == "NAccUnion":
                    this_TR_ROI = [0, 0, 0] # use the loaded coordinates to fill this list, three dimensions reserved for tokens
                    NUM_TOKENS = 3
                else:
                    print("ROI name "+str(ROI_name)+" not implemented, quitting...")
                    quit(0)

                for coordinates in ROI_coordinates: # this list was flattened to be in the desired order back in countROIs.py
                    x = coordinates[0]
                    y = coordinates[1]
                    z = coordinates[2]
                    this_TR_ROI.append(run_data[x][y][z][TR])

                # ROI has now been excised for this TR. add to running list
                this_TR_ROI = np.array(this_TR_ROI)
                all_runs_data.append(this_TR_ROI)

            # done with this run, nothing to do here, there are no run-specific data structures
            print("finished run "+str(run_n))

        # all runs are done
        all_runs_data = np.array(all_runs_data)
        print("all runs data has shape "+str(all_runs_data.shape))
        # print("some samples:")
        # print(all_runs_data[0])
        # print(all_runs_data[2700])
        # print(all_runs_data[1200])

        with open(session_dir+"allruns_"+ROI_name+".p", "wb") as allruns_fp:
            pickle.dump(all_runs_data, allruns_fp)

        all_runs_detrended = np.array(detrend_flattened(all_runs_data, num_tokens = NUM_TOKENS))
        all_runs_detrended_standardized = np.array(standardize_flattened(all_runs_detrended, num_tokens = NUM_TOKENS))

        with open(session_dir+"allruns_"+ROI_name+"_detrendedstandardized.p", "wb") as allruns_detstand_fp:
            pickle.dump(all_runs_detrended_standardized, allruns_detstand_fp)

        print("all runs detrended standardized has shape "+str(all_runs_detrended_standardized.shape))
        # 2880 timesteps by the way (dummy TRs already removed)
        # print("some samples:")
        # print(all_runs_detrended_standardized[0])
        # print(all_runs_detrended_standardized[2700])
        # print(all_runs_detrended_standardized[1200])


if do_genre:
    genre_subs = ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005"]

    for session in genre_subs:
        base_path = "/Volumes/External/genrenew/preproc/"
        session_dir = base_path+session+"/"
        all_runs_data = []


        # 6 test runs, 12 training runs, each has 40 clips, each clip is 15 seconds or 10 TRs
        # each run has 10 dummy TRs at the start
        # only take the first 10 clips (100 TRs) of each Test run
        # for run_n in range(1, 7):
        #     run_dir = session_dir+"func-task.tag-test.tag-preprocessed.run-0"+str(run_n)+"/"
        #     run_img = nib.load(run_dir+"bold.nii.gz")
        #     #run_data = run_img.get_fdata()
        #
        #     # run_data is 97x115x97x410. recall that the first ten TRs of each run are dummy data.
        #     start_TR=10
        #     end_TR = 410
        #
        #     for TR in range(start_TR, end_TR):
        #         TR_data = run_img.dataobj[...,TR]
        #
        #         if ROI_name == "NAccUnion":
        #             this_TR_ROI = [0, 0, 0] # use the loaded coordinates to fill this list, three dimensions reserved for tokens
        #             NUM_TOKENS = 3
        #         else:
        #             print("ROI name "+str(ROI_name)+" not implemented, quitting...")
        #             quit(0)
        #
        #         for coordinates in ROI_coordinates: # this list was flattened to be in the desired order back in countROIs.py
        #             x = coordinates[0]
        #             y = coordinates[1]
        #             z = coordinates[2]
        #             #this_TR_ROI.append(run_data[x][y][z][TR])
        #             this_TR_ROI.append(TR_data[x][y][z])
        #
        #         # ROI has now been excised for this TR. add to running list
        #         this_TR_ROI = np.array(this_TR_ROI)
        #         all_runs_data.append(this_TR_ROI)
        #
        #     # done with this run, nothing to do here, there are no run-specific data structures
        #
        #     print("finished run "+str(run_n))


        # do it all again for the 12 training runs
        for run_n in range(1, 13):
            if run_n < 10:
                run_str = "0"+str(run_n)
            else:
                run_str = str(run_n)
            run_dir = session_dir+"func-task.tag-training.tag-preprocessed.run-"+run_str+"/"
            run_img = nib.load(run_dir+"bold.nii.gz")
            #run_data = run_img.get_fdata()

            # run_data is 97x115x97x410. recall that the first ten TRs of each run are dummy data.
            start_TR=10
            end_TR = 410

            for TR in range(start_TR, end_TR):
                TR_data = run_img.dataobj[...,TR]

                if ROI_name == "NAccUnion":
                    this_TR_ROI = [0, 0, 0] # use the loaded coordinates to fill this list, three dimensions reserved for tokens
                    NUM_TOKENS = 3
                else:
                    print("ROI name "+str(ROI_name)+" not implemented, quitting...")
                    quit(0)

                for coordinates in ROI_coordinates: # this list was flattened to be in the desired order back in countROIs.py
                    x = coordinates[0]
                    y = coordinates[1]
                    z = coordinates[2]
                    this_TR_ROI.append(TR_data[x][y][z])

                # ROI has now been excised for this TR. add to running list
                this_TR_ROI = np.array(this_TR_ROI)
                all_runs_data.append(this_TR_ROI)

            # done with this run, nothing to do here, there are no run-specific data structures
            print("finished run "+str(run_n))

        # all runs are done
        all_runs_data = np.array(all_runs_data)
        print("all runs data has shape " + str(all_runs_data.shape))
        print("some samples:")
        print(all_runs_data[0])
        print(all_runs_data[2700])
        print(all_runs_data[1200])

        with open(session_dir + "alltrainingruns_" + ROI_name + ".p", "wb") as allruns_fp:
            pickle.dump(all_runs_data, allruns_fp)

        all_runs_detrended = np.array(detrend_flattened(all_runs_data, num_tokens=NUM_TOKENS))
        all_runs_detrended_standardized = np.array(standardize_flattened(all_runs_detrended, num_tokens=NUM_TOKENS))

        with open(session_dir + "alltrainingruns_" + ROI_name + "_detrendedstandardized.p", "wb") as allruns_detstand_fp:
            pickle.dump(all_runs_detrended_standardized, allruns_detstand_fp)

        print("all runs detrended standardized has shape " + str(all_runs_detrended_standardized.shape))
        print("some samples:")
        print(all_runs_detrended_standardized[0])
        print(all_runs_detrended_standardized[2700])
        print(all_runs_detrended_standardized[1200])