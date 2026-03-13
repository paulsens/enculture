import os

# example filename sub-sid001088_task-pitchimaginedXtrumXF_run-08_space-MNI152NLin2009cAsym_desc-preproc_bold.json
#basedir = "/Volumes/External/enculture/preproc/"
basedir = "/Volumes/External/genrenew/"
#basedir = "/Volumes/External/enculture/preproc/bids/"
spacestr = "_space-MNI152NLin2009cAsym"

# for subname in os.listdir(basedir):
#     if subname=="dataset_description.json":
#         print("dataset description, skipping")
#         continue
#     if subname=="derivatives":
#         print("derivatives folder, skipping")
#         continue
#
#     subdir = os.path.join(basedir, subname)
#     #print("sessiondir is "+str(sessiondir))
#
#     for sessionname in os.listdir(subdir):
#         sessiondir = os.path.join(subdir,sessionname)
#
#         for branchname in os.listdir(sessiondir): # anat or func
#             branchpath = os.path.join(sessiondir,branchname)
#
#             for filename in os.listdir(branchpath):
#                 pieces = filename.split("_")
#                 type = pieces[-1]
#                 filepath = os.path.join(branchpath, filename)
#
#                 halves = filename.split("_desc")
#                 newname = halves[0]+"_space-MNI152NLin2009cAsym_desc-"+type
#                 newpath = os.path.join(branchpath, newname)
#                 print("newpath is "+str(newpath))
#
#                 os.rename(filepath, newpath)

genresubs = ["sub-001","sub-002","sub-003","sub-004","sub-005"]
# this half of the file renames the non-bids structure downloaded from brainlife

for subname in genresubs:

    subdir = os.path.join(basedir, subname)
    #print("sessiondir is "+str(sessiondir))

    for itemname in os.listdir(subdir):
        itempath = os.path.join(subdir,itemname)

        if itemname.startswith("func"):
            print("skipping {0}".format(itemname))
            continue
        if itemname.startswith("transform"):
            print("skipping {0}".format(itemname))
            continue

        if itemname[3]=="n":
            #print(itemname)
            temp_name = itemname.split("neuro-")
            temp_name = temp_name[1]

        elif itemname[3]=="r":
            #print(itemname)
            temp_name = itemname.split("report-")
            temp_name = temp_name[1]


        else:
            print('illegal item name, quitting')
            quit(0)
        new_name = temp_name.split(".id")
        new_name = new_name[0]


        newpath = os.path.join(subdir, new_name)
        #print("renaming "+str(itempath)+" to "+str(newpath))
        os.rename(itempath, newpath)
