import sys, os, ffmpeg, wave
from mutagen.mp3 import MP3
import matplotlib.pyplot as plt

# good_old is after the synthesizer was fixed, and matches the stimuli presented to the ferrets
# good_new contains good old and some supplementary stuff to even out the distributions of durations

bad_path = "/Volumes/External/enculture/bad/"
good_old_path = "/Volumes/External/enculture/good_old/"

def count_files(path):
    for dir in os.listdir(path):
        # ignore meta files
        if dir[0]==".":
            continue
        thisdir = path+dir
        if not os.path.isdir(thisdir):
            continue
        thisdir_files = os.listdir(thisdir)

        wav_count = 0
        asd_count = 0
        mp3_count = 0
        for file in thisdir_files:
            if file[-3:]=="wav":
                wav_count+=1
            elif file[-3:]=="asd":
                asd_count+=1
            elif file[-3:]=="mp3":
                mp3_count+=1

        print("Directory "+str(dir)+" has "+str(wav_count)+" wav files, "+str(asd_count)+" asd files, and "+str(mp3_count)+" mp3 files.")

chosen_path = good_old_path
count_files(chosen_path)
dircount = 0

for dir in sorted(os.listdir(chosen_path)):
    # ignore meta files
    if dir[0] == ".":
        continue
    thisdir = chosen_path + dir
    if not os.path.isdir(thisdir):
        continue
    thisdir_files = os.listdir(thisdir)

    print("Directory: "+str(dir)+"\n")
    dir_dict = {}
    dir_lists = {}
    total_dur = 0

    for file in sorted(thisdir_files):
        if file[-3:]=="asd" or file[0]==".":
            continue
        elif file[-3:]=="wav":
            #print("file is "+str(file))
            file_path = thisdir+"/"+file

            # obj is an instance of the Wave_read class
            #
            obj = wave.open(file_path,"rb")
            nchannels = obj.getnchannels()
            sampwidth = obj.getsampwidth()
            framerate = obj.getframerate()
            nframes = obj.getnframes()

            nseconds = nframes/framerate
            total_dur+=nseconds
            if nseconds not in dir_dict.keys():
                dir_dict[nseconds]=1
                dir_lists[nseconds]=[file[-7:-4]]
            else:
                dir_dict[nseconds]+=1
                dir_lists[nseconds].append(file[-7:-4])

        elif file[-3:]=="mp3":
            #print("file is "+str(file))
            file_path = thisdir+"/"+file

            obj = MP3(file_path)
            duration = obj.info.length
            total_dur+=duration
            if duration not in dir_dict.keys():
                dir_dict[duration]=1

            else:
                dir_dict[duration]+=1


    print("Dict for "+str(dir)+" is "+str(dir_dict))
    print("Lists for "+str(dir)+" is ")
    for key in dir_lists.keys():
        print("{0}: {1}".format(key, dir_lists[key]))
    print("Total duration for "+str(dir)+" is "+str(total_dur))
    tuplelist = []
    for key in dir_dict.keys():
        this_tuple = (key, dir_dict[key])
        tuplelist.append(this_tuple)

    x = []
    y = []
    for tup in tuplelist:
        x.append(tup[0])
        y.append(tup[1])

    fig = plt.figure(figsize=(10,5))
    plt.bar(x, y, color='maroon', width=1.0)

    plt.xlabel("Length of track")
    plt.ylabel("Number of tracks")
    plt.title("Track Lengths for "+str(dir))
    plt.show()



