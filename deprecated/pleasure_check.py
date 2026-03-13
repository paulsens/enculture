from statistics import mean

sublist = {
    "1401":["5515","5552"],
    "2548":["5505","5540"],
    "2564":["5538","5567"],
    "2566":["5542","5572"],
    "2589":["5590","5615"]
}

logpath = "/Volumes/External/enculture/key_logs/"
eventspath = "/Volumes/External/enculture/BIDS_events/"
session1_Bach_pleasure = {}
session1_Chinese_pleasure = {}
session2_Bach_pleasure = {}
session2_Chinese_pleasure = {}

for shortid in sublist.keys():
    subid = "sub-sid00"+shortid
    session = 1
    session1_Bach_pleasure[subid] = []
    session1_Chinese_pleasure[subid] = []
    session2_Bach_pleasure[subid] = []
    session2_Chinese_pleasure[subid] = []

    for shortacc in sublist[shortid]:
        accession = "A00"+shortacc
        accdir = logpath+subid+"/ses-"+accession
        eventsdir = eventspath+subid+"/ses-"+accession

        for run_n in range(1, 9):
            key_filename = accession+"_run_"+str(run_n)+"_key_log.txt"
            key_path = accdir+"/"+key_filename
            events_path = subid+"_task-enculture"+str(session)+"_run0"+str(run_n)+"_events.tsv"

            with open(key_path, "r") as key_fp:
                all_lines = key_fp.readlines()
                relevant_lines = []
                position = -1 # start at end of file
                while len(relevant_lines) < 4:
                    current_line = all_lines[position]
                    current_line = current_line.strip()
                    if current_line != "" and current_line[:6]!="IGNORE":
                        #print(current_line)
                        relevant_lines.append(current_line)
                    position = position - 1
                #print("the above is for "+key_path)
            relevant_lines.reverse()


            with open(eventsdir+"/"+events_path, "r") as events_fp:
                events_lines = events_fp.readlines()
                conditions = []
                conditions.append(events_lines[2].split("\t")[2].strip())
                conditions.append(events_lines[6].split("\t")[2].strip())
                conditions.append(events_lines[10].split("\t")[2].strip())
                conditions.append(events_lines[14].split("\t")[2].strip())

                for j in range(0, len(conditions)):
                    condition = conditions[j]
                    #print("condition is "+str(condition))
                    if session==1:
                        Bach_dict = session1_Bach_pleasure
                        Chinese_dict = session1_Chinese_pleasure
                    elif session==2:
                        Bach_dict = session2_Bach_pleasure
                        Chinese_dict = session2_Chinese_pleasure

                    if condition == "Bach":
                        pleasure_rating = relevant_lines[j].split(",")[1].strip()
                        if pleasure_rating=="0":
                            print("zero rating, skipping")
                        else:
                            Bach_dict[subid].append(int(pleasure_rating))
                    elif condition=="Chinese":
                        pleasure_rating = relevant_lines[j].split(",")[1].strip()
                        if pleasure_rating=="0":
                            print("zero rating, skipping")
                        else:
                            Chinese_dict[subid].append(int(pleasure_rating))
                    #print("pleasure rating was "+str(pleasure_rating))

            # input("press enter to continue")
        session+=1

    print("Stuff for "+str(subid))
    print("session1_Bach_pleasure average: "+str(mean(session1_Bach_pleasure[subid])))
    print("session1_Chinese_pleasure average: "+str(mean(session1_Chinese_pleasure[subid])))
    print("session2_Bach_pleasure average: "+str(mean(session2_Bach_pleasure[subid])))
    print("session2_Chinese_pleasure average: "+str(mean(session2_Chinese_pleasure[subid])))
