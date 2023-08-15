import torch
import torch.nn as nn
import random
from random import randint
import numpy as np
from helpers import *
from transfer_transformer import *
#from pitchclass_data import *
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.optim as optim
from Constants import *
import sys
import os
import datetime

null_model = False # wildly important that this is set to False when training real models
null_labels  = np.ones(7548) # number of training samples on sametimbre with runs 5-8 held out
null_labels[:3774]=0
np.random.shuffle(null_labels)

printed_labels = 0
debug=1
val_flag=1
seed=3
#random.seed(seed)
#torch.manual_seed(seed)
#np.random.seed(seed)
#mask_variation=True
valid_accuracy=True

LR_def=0.00001 #defaults, should normally be set by command line
printed_count=0
val_printed_count=0
EPOCHS=10
#torch.use_deterministic_algorithms(True)
if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        thiscount=None #gets changed if a count is passed as a command line argument
        #get command line arguments and options
        opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
        args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
        if "-m" in opts:
            # -m "this is the description of the run" will be at the end of the command line call
            idx = opts.index("-m")
            run_desc = args[idx]
        else:
            run_desc = None
        # text description of this job


        if "-fold" in opts:
            # which fold of crossvalidation is this, or empty string
            idx = opts.index("-fold")
            fold = args[idx] # the actual number
            fold_str = "_fold"+str(fold) # for file pathing
        else:
            fold_str = "" # for file pathing
            fold = None


        # train on gpu or not, not implemented yet
        if "-gpu" in opts:
            idx = opts.index("-gpu")
            gpunum = args[idx]  # currently only works if only one gpu is given
            device = torch.device("cuda:" + str(gpunum))
        else:
            device = "cpu"

        # index in job submission script, indicates the heldout run
        if "-heldout_run" in opts:
            # count and thiscount can be read as the index of the heldout run
            idx = opts.index("-heldout_run")
            thiscount = int(args[idx])
        else:
            thiscount = 0


        if "-LR" in opts:
            idx = opts.index("-LR")
            LR = args[idx]
            if LR == "default":
                LR = LR_def  # default value if nothing is passed by command line
            LR = float(LR)
        else:
            LR = 0.00001


        # whether or not we want to save the model after training, defaults to False if not provided
        if "-save_model" in opts:
            idx = opts.index("-save_model")
            save_model = args[idx]
            save_model = True if save_model == "True" else False
        else:
            save_model = False

        if "-attention_heads" in opts:
            idx = opts.index("-attention_heads")
            attn_heads = int(args[idx])
        else:
            attn_heads = ATTENTION_HEADS  # defined in Constants.py

        if "-forward_expansion" in opts:
            idx = opts.index("-forward_expansion")
            f_exp = int(args[idx])
        else:
            f_exp = 4  # arbitrary default value

        if "-num_layers" in opts:
            idx = opts.index("-num_layers")
            n_layers = int(args[idx])
        else:
            n_layers = 2  # arbitrary default value

        if "-task" in opts:
            idx = opts.index("-task")
            task = str(args[idx])
        else:
            task="both"  # arbitrary default value

        if "-seq_len" in opts:
            idx = opts.index("-seq_len")
            seq_len = int(args[idx])
        else:
            seq_len = 5  # arbitrary default value

        if "-dataset" in opts:
            idx = opts.index("-dataset")
            orig_dataset = str(args[idx])
        else:
            orig_dataset = "genre_NTP"  # arbitrary default value



        today = datetime.date.today()
        now = datetime.datetime.now()
        ##############################  SET PARAMETERS  ##############################
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #dictionary of hyperparameters, eventually should probably come from command line
        hp_dict={
            "orig_dataset":orig_dataset,
            "task":task,
            "CLS_task":"samesession", #same_genre or nextseq
            "num_CLS_labels": 2,
            "MSK_task":"reconstruction", #legacy name for masked brain modeling
            "COOL_DIVIDEND" : COOL_DIVIDEND,
            "ATTENTION_HEADS" : attn_heads,
            "num_layers" : n_layers,
            "device" : str(device),
            "MSK_flag" : 1,
            "CLS_flag" : 1,
            "BATCH_SIZE" : 1,
            "EPOCHS" : EPOCHS,
            "LEARNING_RATE" : LR, #set at top of this file or by command line argument
            #Have to manually set the name of the folder whose training data you want to use, since there will be many
            "count" : str(thiscount),
            #manually set max_seq_length used in data creation, in the input CLS+seq+SEP+seq this is the max length of seq
            "max_sample_length":seq_len,
            #"mask_variation":mask_variation,
            "within_subject":1,
            "num_subjects":5,
            "heldout_run":thiscount,
            "forward_expansion":f_exp
        }


        torch.set_default_dtype(torch.float32)

        #set up logfile, PRETRAIN_LOG_PATH is defined in Constants.py
        today_dir = THESIS_PATH+"pairedpretrain/logs/"+str(today)+"/"
        if not (os.path.exists(today_dir)):
            os.mkdir(today_dir)

        if(thiscount!=None):
            logcount=thiscount
        else:
            logcount=0
        logfile = today_dir + "pairedpretrainlog_"+str(logcount)+".txt"
        while (os.path.exists(logfile)):
            logcount+=1
            logfile = today_dir + "pretrainlog_" + str(logcount) + ".txt"
        log = open(logfile,"w")
        log.write(str(now)+"\n")

        # run_desc is potentially given in command line call
        if(run_desc is not None):
            log.write(run_desc+"\n\n")
            print(run_desc+"\n\n")
        #write hyperparameters to log
        for hp in hp_dict.keys():
            log.write(str(hp)+" : "+str(hp_dict[hp])+"\n")

        seed = hp_dict["heldout_run"]
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)


        #load samples and labels
        if orig_dataset=="genre_NTP":
            hp_dict["heldout_run"]=hp_dict["heldout_run"]+1
            with open("/isi/music/auditoryimagery2/seanthesis/enculturation/datasets/genre/X_train_fold"+str(hp_dict["heldout_run"])+".p", "rb") as samples_fp:
                train_X = pickle.load(samples_fp)
            with open("/isi/music/auditoryimagery2/seanthesis/enculturation/datasets/genre/y_train_fold"+str(hp_dict["heldout_run"])+".p", "rb") as labels_fp:
                train_Y = pickle.load(labels_fp)
            with open("/isi/music/auditoryimagery2/seanthesis/enculturation/datasets/genre/X_val_fold"+str(hp_dict["heldout_run"])+".p","rb") as samples_fp:
                val_X = pickle.load(samples_fp)
            with open("/isi/music/auditoryimagery2/seanthesis/enculturation/datasets/genre/y_val_fold"+str(hp_dict["heldout_run"])+".p","rb") as labels_fp:
                val_Y = pickle.load(labels_fp)

        elif orig_dataset=="enc_NTP":
            with open("/isi/music/auditoryimagery2/seanthesis/enculturation/datasets/enc/X_trainNTP_fold1.p", "rb") as samples_fp:
                train_X = pickle.load(samples_fp)
            with open("/isi/music/auditoryimagery2/seanthesis/enculturation/datasets/enc/y_trainNTP_fold1.p", "rb") as labels_fp:
                train_Y = pickle.load(labels_fp)
            with open("/isi/music/auditoryimagery2/seanthesis/enculturation/datasets/enc/X_valNTP_fold1.p","rb") as samples_fp:
                val_X = pickle.load(samples_fp)
            with open("/isi/music/auditoryimagery2/seanthesis/enculturation/datasets/enc/y_valNTP_fold1.p","rb") as labels_fp:
                val_Y = pickle.load(labels_fp)
        elif orig_dataset=="enc_bachSS":
            with open("/isi/music/auditoryimagery2/seanthesis/enculturation/datasets/enc/bach_X_trainSS.p", "rb") as samples_fp:
                train_X = pickle.load(samples_fp)
            with open("/isi/music/auditoryimagery2/seanthesis/enculturation/datasets/enc/bach_y_trainSS.p", "rb") as labels_fp:
                train_Y = pickle.load(labels_fp)
            with open("/isi/music/auditoryimagery2/seanthesis/enculturation/datasets/enc/bach_X_valSS.p","rb") as samples_fp:
                val_X = pickle.load(samples_fp)
            with open("/isi/music/auditoryimagery2/seanthesis/enculturation/datasets/enc/bach_y_valSS.p","rb") as labels_fp:
                val_Y = pickle.load(labels_fp)

        elif orig_dataset=="enc_shanxiSS":
            with open("/isi/music/auditoryimagery2/seanthesis/enculturation/datasets/enc/shanxi_X_trainSS.p", "rb") as samples_fp:
                train_X = pickle.load(samples_fp)
            with open("/isi/music/auditoryimagery2/seanthesis/enculturation/datasets/enc/shanxi_y_trainSS.p", "rb") as labels_fp:
                train_Y = pickle.load(labels_fp)
            with open("/isi/music/auditoryimagery2/seanthesis/enculturation/datasets/enc/shanxi_X_valSS.p","rb") as samples_fp:
                val_X = pickle.load(samples_fp)
            with open("/isi/music/auditoryimagery2/seanthesis/enculturation/datasets/enc/shanxi_y_valSS.p","rb") as labels_fp:
                val_Y = pickle.load(labels_fp)
        #load valsamples and vallabels



        #train_X has shape (timesteps, max_length, voxel_dim)
        num_samples = len(train_X)
        max_length = len(train_X[0]) #should be max_sample_length*2 + 2
        #assert (max_length == (hp_dict["max_sample_length"]*2 +2))

        voxel_dim = len(train_X[0][0])
        print("voxel dim is "+str(voxel_dim))
        print("num samples is "+str(num_samples))
        #convert to numpy arrays
        train_X = np.array(train_X)
        train_Y = np.array(train_Y)
        print("train x has shape "+str(train_X.shape))
        val_X = np.array(val_X)
        val_Y = np.array(val_Y)

        #convert to tensors
        train_X = torch.from_numpy(train_X)
        train_Y = torch.from_numpy(train_Y)
        val_X = torch.from_numpy(val_X)
        val_Y = torch.from_numpy(val_Y)

        all_data = TrainData(train_X, train_Y) #make the TrainData object
        val_data = TrainData(val_X, val_Y) #make TrainData object for validation data

        #train_val_dataset defined in helpers, val_split defined in Constants
        #datasets = train_val_dataset(all_data, val_split)

        train_loader = DataLoader(dataset=all_data, batch_size=hp_dict["BATCH_SIZE"], shuffle=True) #make the DataLoader object
        val_loader = DataLoader(dataset=val_data, batch_size=hp_dict["BATCH_SIZE"], shuffle=False)
        log.write("voxel dim is "+str(voxel_dim)+"\n\n")
        # MSK_token = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
        # MSK_token = np.array(MSK_token)
        # MSK_token = torch.from_numpy(MSK_token)

        #CLS_task_weight = hp_dict["CLS_task_weight"]
        #MSK_task_weight = 1-CLS_task_weight

        CLS_task_labels = 2 #two possible labels for same genre task, yes or no
        #num_genres = 10 #from the training set

        src_pad_sequence = [0]*voxel_dim

        #model = Transformer(next_sequence_labels=binary_task_labels, num_genres=num_genres, src_pad_sequence=src_pad_sequence, max_length=max_length, voxel_dim=voxel_dim, ref_samples=ref_samples, mask_task=hp_dict["MSK_task"], print_flag=0).to(hp_dict["device"])
        model = Transformer(num_CLS_labels=hp_dict["num_CLS_labels"], num_genres=10, src_pad_sequence=src_pad_sequence, max_length=max_length, voxel_dim=voxel_dim, ref_samples=None, mask_task=hp_dict["MSK_task"], print_flag=0, heads = hp_dict["ATTENTION_HEADS"], num_layers=hp_dict["num_layers"], forward_expansion=hp_dict["forward_expansion"]).to(hp_dict["device"])
        model = model.float()
        model.to(hp_dict["device"])

        criterion_bin = nn.CrossEntropyLoss()
        if hp_dict["MSK_task"]=="genre_decode":
            criterion_multi = nn.CrossEntropyLoss()
            get_multi_acc = True
        elif hp_dict["MSK_task"]=="reconstruction":
            criterion_multi = nn.MSELoss()
            get_multi_acc = False
        optimizer = optim.Adam(model.parameters(), lr=hp_dict["LEARNING_RATE"], betas=(0.9,0.999), weight_decay=0.0001)

        best_avg_val_acc = 0
        for e in range(1, hp_dict["EPOCHS"]+1):
            epoch_val_masks=[]
            #0'th index is the number of times the model was correct when the ground truth was 0, and when ground truth was 1
            bin_correct_train = [0,0]
            bin_correct_val = [0,0]
            bin_neg_count_train = 0 #count the number of training samples where 0 was the correct answer
            bin_neg_count_val = 0 #count the number of validation samples where 0 was the correct answer

            random.seed(seed+e)
            torch.manual_seed(seed+e)
            np.random.seed(seed+e)
            model.train()  # sets model status, doesn't actually start training
                #need the above every epoch because the validation part below sets model.eval()
            epoch_loss = 0
            epoch_acc = 0
            epoch_acc2 = 0
            epoch_tbin_loss = 0
            epoch_vbin_loss = 0
            epoch_tmulti_loss = 0
            epoch_vmulti_loss = 0

            batch_count = 0
            for X_batch, y_batch in train_loader:
                if null_model:
                    null_label = null_labels[batch_count]
                    y_batch = [[null_label]]
                    y_batch=np.array(y_batch)
                    y_batch=torch.from_numpy(y_batch)
                batch_mask_indices = []
                X_batch=X_batch.float()
                y_batch=y_batch.float()
                X_batch, y_batch = X_batch.to(hp_dict["device"]), y_batch.to(hp_dict["device"])
                #ytrue_bin_batch = [] #list of batch targets for binary classification task
                #ytrue_multi_batch = [] #list of batch targets for multi-classification task
                #print("before apply masks, first input is "+str(X_batch[0]))
                optimizer.zero_grad() #reset gradient to zero before each mini-batch
                # for x in range(0,hp_dict["BATCH_SIZE"]):
                #     sample_mask_indices = []  # will either have 1 or 2 ints in it
                #     sample_dists = [] #will be appended to ytrue_multi_batch
                #     ytrue_dist_multi1 = np.zeros((10,))  # we want a one-hot probability distrubtion over the 10 genre labels
                #     ytrue_dist_multi2 = np.zeros((10,))  # only used when this sample gets two masks/replacements
                #     #no return value from apply_masks, everything is updated by reference in the lists
                #     apply_masks(X_batch[x], y_batch[x], ref_samples, hp_dict, mask_variation, ytrue_multi_batch, sample_dists, ytrue_dist_multi1, ytrue_dist_multi2, batch_mask_indices, sample_mask_indices, mask_task=hp_dict["MSK_task"], log=log, heldout=False, main_task = hp_dict["task"])

                # for y in range(0,hp_dict["BATCH_SIZE"]):
                #     # if(y_batch[y][0]):
                #     #     ytrue_dist_bin = [0,1] #true, they are the same genre
                #     # else:
                #     #     ytrue_dist_bin = [1,0] #false
                #     ytrue_bin_batch.append(y_batch[y]) #should give a list BATCHSIZE many same_genre boolean targets
                ytrue_bin_batch = y_batch
                #convert label lists to pytorch tensors
                #ytrue_bin_batch = np.array(ytrue_bin_batch)
                #ytrue_multi_batch = np.array(ytrue_multi_batch)

                #batch_mask_indices = np.array(batch_mask_indices)

                #ytrue_bin_batch = torch.from_numpy(ytrue_bin_batch).float()
                #ytrue_multi_batch = torch.from_numpy(ytrue_multi_batch).float()

                #batch_mask_indices = torch.from_numpy(batch_mask_indices).float()

                #send this stuff to device
                ytrue_bin_batch.to(hp_dict["device"])
                #batch_mask_indices.to(hp_dict["device"])

                #returns predictions for binary class and multiclass, in that order
                #print("input to model looks like:")
                #X = X_batch[0]
                # for i in range(0, len(X)):
                #     print(X[i])
                #     print("\n")
                # print("batch mask indices is "+str(batch_mask_indices))
                # quit(0)
                if orig_dataset in ["genre_NTP", "enc_NTP"]:
                    batch_mask_indices = "ntp"
                else:
                    batch_mask_indices = "samesession"
                ypred_bin_batch = model(X_batch, "samesession")
                for batch_idx, bin_pred in enumerate(ypred_bin_batch): #for each 2 dimensional output vector for the binary task
                    bin_true=ytrue_bin_batch[batch_idx]
                    true_idx=torch.argmax(bin_true)
                    pred_idx=torch.argmax(bin_pred)
                    if(true_idx==pred_idx):
                        bin_correct_train[true_idx]+=1 #either 0 or 1 was the correct choice, so count it
                    if(true_idx==0):
                        bin_neg_count_train+=1

                ypred_bin_batch = ypred_bin_batch.float()
                #ypred_multi_batch = ypred_multi_batch.float()
                #log.write("ypred_multi_batch has shape "+str(ypred_multi_batch.shape)+"\n and ytrue_multi_batch has shape "+str(ytrue_multi_batch.shape))
                #log.write("For binary classification, predictions are "+str(ypred_bin_batch)+" and true labels are "+str(ytrue_bin_batch)+"\n")
                loss_bin = criterion_bin(ypred_bin_batch, ytrue_bin_batch)
                #log.write("The loss in that case was "+str(loss_bin)+"\n")
                #log.write("For genre classification, predictions are "+str(ypred_multi_batch)+" and true labels are "+str(ytrue_multi_batch)+"\n")
                # if hp_dict["task"] != "CLS_only":
                #     loss_multi = criterion_multi(ypred_multi_batch, ytrue_multi_batch)
                # else:
                loss_multi = 0
                #log.write("The loss in that case was "+str(loss_multi)+"\n\n")
                # if printed_count < 1:
                #     print("sample "+str(printed_count)+": "+str(X_batch)+"\n\n")
                #     print("ypred "+str(printed_count)+": "+str(ypred_bin_batch)+"\n\n")
                #     print("ytrue "+str(printed_count)+": "+str(ytrue_bin_batch)+"\n\n")
                #     printed_count+=1
                #     model.print_flag=0
                loss = loss_bin #toy example for just same-genre task
                acc = get_accuracy(ypred_bin_batch, ytrue_bin_batch, hp_dict["CLS_task"], log)


                loss.backward()
                optimizer.step()
                batch_count+=1


                epoch_loss += loss.item()
                #the word valid here does not refer to validation, but rather is this task something we can obtain a valid accuracy for

                epoch_acc += acc.item()
                epoch_acc2 += 0
                # if not valid_accuracy:
                #     print("Accuracy is invalid.")
                #     log.write("Accuracy is invalid.")
                #log.write("added "+str(acc.item())+" to epoch_acc")

            # now calculate validation loss/acc, turn off gradient
            # print("Model params before val split:\n")
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name,param.data)
            if val_X is not None:
                model.eval()
                #model.print_flag=1
                random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                with torch.no_grad():
                    val_loss=0
                    val_acc=0
                    val_acc2=0
                    for X_batch_val, y_batch_val in val_loader:
                        X_batch_val=X_batch_val.float()
                        y_batch_val=y_batch_val.float()
                        #batch_mask_indices_val = []
                        #ytrue_bin_batch_val = []  # list of batch targets for binary classification task
                        #ytrue_multi_batch_val = []  # list of batch targets for multi-classification task
                        X_batch_val, y_batch_val = X_batch_val.to(hp_dict["device"]), y_batch_val.to(hp_dict["device"])

                        # for x in range(0, hp_dict["BATCH_SIZE"]):
                        #     sample_mask_indices_val = []  # will either have 1 or 2 ints in it
                        #     sample_dists_val = []  # will be appended to ytrue_multi_batch
                        #     ytrue_dist_multi1_val = np.zeros(
                        #         (10,))  # we want a one-hot probability distrubtion over the 10 genre labels
                        #     ytrue_dist_multi2_val = np.zeros(
                        #         (10,))  # only used when this sample gets two masks/replacements
                        #     # no return value from apply_masks, everything is updated by reference in the lists
                        #     apply_masks(X_batch_val[x], y_batch_val[x], ref_samples, hp_dict, mask_variation,   ytrue_multi_batch_val, sample_dists_val, ytrue_dist_multi1_val, ytrue_dist_multi2_val, batch_mask_indices_val, sample_mask_indices_val, mask_task=hp_dict["MSK_task"], log=log, heldout=True, main_task = hp_dict["task"])
                        #
                        # epoch_val_masks.append(batch_mask_indices_val)

                        # for y in range(0, hp_dict["BATCH_SIZE"]):
                        #
                        #     ytrue_bin_batch_val.append(
                        #         y_batch_val[y])  # should give a list BATCHSIZE many same_genre boolean targets

                        # convert label lists to pytorch tensors
                        #ytrue_bin_batch_val = np.array(ytrue_bin_batch_val)
                        #ytrue_multi_batch_val = np.array(ytrue_multi_batch_val)
                        #ytrue_bin_batch_val = torch.from_numpy(ytrue_bin_batch_val).float()
                        #ytrue_multi_batch_val = torch.from_numpy(ytrue_multi_batch_val).float()
                        #epoch_val_masks.append(batch_mask_indices_val)
                        # returns predictions for binary class and multiclass, in that order
                        ytrue_bin_batch_val = y_batch_val
                        ypred_bin_batch_val = model(X_batch_val, "samesession")

                        #get accuracy stats for validation samples
                        for batch_idx, bin_pred in enumerate(
                                ypred_bin_batch_val):  # for each 2 dimensional output vector for the binary task
                            bin_true = ytrue_bin_batch_val[batch_idx]
                            true_idx = torch.argmax(bin_true)
                            pred_idx = torch.argmax(bin_pred)
                            if (true_idx == pred_idx):
                                bin_correct_val[true_idx] += 1  # either 0 or 1 was the correct choice, so count it
                            if (true_idx == 0):
                                bin_neg_count_val += 1


                        ypred_bin_batch_val = ypred_bin_batch_val.float()
                        #ypred_multi_batch_val = ypred_multi_batch_val.float()
                        # log.write("ypred_multi_batch_val has shape " + str(
                        #     ypred_multi_batch_val.shape) + "\n and ytrue_multi_batch_val has shape " + str(
                        #     ytrue_multi_batch_val.shape))
                        # log.write("For binary classification, predictions are "+str(ypred_bin_batch)+" and true labels are "+str(ytrue_bin_batch)+"\n")
                        loss_bin_val = criterion_bin(ypred_bin_batch_val, ytrue_bin_batch_val)
                        # log.write("The loss in that case was "+str(loss_bin)+"\n")
                        # log.write("For genre classification, predictions are "+str(ypred_multi_batch)+" and true labels are "+str(ytrue_multi_batch)+"\n")

                        loss_multi_val = 0
                        # log.write("The loss in that case was "+str(loss_multi)+"\n\n")

                        loss = loss_bin_val  # toy example for just same-genre task
                        acc = get_accuracy(ypred_bin_batch_val, ytrue_bin_batch_val, hp_dict["CLS_task"],log)


                        val_loss += loss.item()

                        val_acc += acc.item()
                        val_acc2 += 0

            print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f} | Acc2: {epoch_acc2/len(train_loader):.3f}')
            print("Epoch bin training stats:\n")
            print("correct counts for this epoch: "+str(bin_correct_train))
            print("bin neg sample count: "+str(bin_neg_count_train))
            print("number of samples: "+str(len(train_loader)))

            log.write(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f} | Acc2: {epoch_acc2/len(train_loader):.3f}')

            if val_X is not None:
                print(f'Validation: | Loss: {val_loss/len(val_loader):.5f} | Acc: {val_acc/len(val_loader):.3f} | Acc2: {val_acc2/len(val_loader):.3f}')
                log.write(f'Validation: | Loss: {val_loss/len(val_loader):.5f} | Acc: {val_acc/len(val_loader):.3f} | Acc2: {val_acc2/len(val_loader):.3f}')

                print("Epoch bin val stats:\n")
                print("correct counts for this epoch: " + str(bin_correct_val))
                print("bin neg sample count: " + str(bin_neg_count_val))
                print("number of samples: " + str(len(val_loader)))
                #print("epoch val masks:" + str(epoch_val_masks) + "\n\n")
                avg_val_acc = val_acc/len(val_loader)
                # if not valid_accuracy:
                #     print("Accuracy is invalid.")
                #     log.write("Accuracy is invalid.")

            #print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')
            #log.write(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')

        # print("before saving model, model has params:\n")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

            if avg_val_acc > best_avg_val_acc: # save model is set by command line argument
                best_avg_val_acc = avg_val_acc
                best_val_epoch = e
                model_path = "/isi/music/auditoryimagery2/seanthesis/enculturation/trained_models/pretrain/"+str(orig_dataset)+"/states_"+str(thiscount)+".pt"


                torch.save(model.state_dict(),model_path)
                model_path = "/isi/music/auditoryimagery2/seanthesis/enculturation/trained_models/pretrain/"+str(orig_dataset)+"/full_"+str(thiscount)+".pt"
                torch.save(model,model_path)
    print("Best val epoch was "+str(best_val_epoch))
    log.close()
