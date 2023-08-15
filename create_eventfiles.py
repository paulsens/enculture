import pandas as pd
import numpy as np
import soundfile


sublist = {
    #"1401":["5515","5552"],
    #"2548":["5505","5540"],
    #"2564":["5538","5567"],
    #"2566":["5542","5572"],
    "2589":["5590","5615"]
}
logs_path = "/Volumes/External/enculture/key_logs/sub-sid00"
clips_path = "/Volumes/External/enculture/glued/"
events_path = "/Volumes/External/enculture/BIDS_events/"

for subid in sublist.keys():
    for run_n in range(1, 9):
        log_file = logs_path+subid+"/subject_{0}_run_{1}.txt".format(subid,run_n)

        fp = open(log_file, "r")
        loglines = fp.readlines()

        onsets = []
        durations = []
        styles = []
        tracks = []

        for line in loglines:
            temp = line.strip()
            if len(temp)<3:
                continue
            extension = temp[-3:]
            info_l = temp.split(",")

            # music file played
            if extension=="wav":
                filename = info_l[-1].split("/") # split /this/long/path/name.wav
                filename = filename[-1]

                style_abbrv = filename[:2] # ba or ch
                track_n = filename[2:4] # single digit track numbers are zero padded, i.e 01 or 05 so this is fine
                if style_abbrv == "ba":
                    style = "Bach"
                    this_path = clips_path+"bach30s/"

                elif style_abbrv == "ch":
                    style = "Chinese"
                    this_path = clips_path+"chinese30s/"

                else:
                    print("illegal style abbreviation, got "+str(style_abbrv)+", quitting...")
                    quit(0)

                wav_file = soundfile.SoundFile(this_path+filename)

                duration = wav_file.frames/wav_file.samplerate
                duration = float(np.round(duration,3))

                onset = float(info_l[0])



            # behavioral component
            elif extension=="png":
                onset = float(info_l[0])
                duration = 6
                style="behavioral"
                track_n = "n/a"

            else:
                continue

            onsets.append(str(onset))
            durations.append(str(duration))
            styles.append(str(style))
            tracks.append(str(track_n))

        this_eventfile = {
            "onset": onsets,
            "duration": durations,
            "style": styles,
            "track": tracks
        }

        df = pd.DataFrame(this_eventfile)
        # the event file is the same for both sessions
        eventfile_name = "sub-sid00"+subid+"_task-enculture1_run0"+str(run_n)+"_events.tsv"
        df.to_csv(events_path+eventfile_name, sep="\t", index=False)
        eventfile_name = "sub-sid00"+subid+"_task-enculture2_run0"+str(run_n)+"_events.tsv"
        df.to_csv(events_path+eventfile_name, sep="\t", index=False)


