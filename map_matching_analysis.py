import csv


if __name__ == "__main__":
    num_traj = [40]
    num_sample = [10, 40]
    edge = [0.5, 1, 2]
    shake = [0.5, 1, 2]
    hmm_result = {}
    for t in num_traj:
	for s in num_sample:
	    hmm_result[s] = {}
	    for e in edge:
		hmm_result[s][e] = {}
		for sh in shake:
		    hmm_result[s][e][sh] = {}
		    # traj_src = "syn_data/ground_truth_" + str(t) + "traj_" + str(s) + "p_" + str(int(e * s)) + "e_" + str(sh) + "shake.json"
		    hmm_src = "syn_meta/hmm_result_" + str(t) + "traj_" + str(s) + "p_" + str(int(e * s)) + "e_" + str(sh) + "shake.csv"
		    with open(hmm_src) as csv_file:
			path_length = []
			map_matching_time = []
			num_match = []
			reader = csv.DictReader(csv_file)
			for row in reader:
			    path_length.append(float(row["path_length"]))
			    map_matching_time.append(float(row["map_matching_time"]))
			    num_match.append(int(row["number_of_right_matches"]))
			hmm_result[s][e][sh]["length"] = path_length
			hmm_result[s][e][sh]["time"] = map_matching_time
			hmm_result[s][e][sh]["match"] = num_match
    for s in num_sample:
	for e in edge:
	    for sh in shake:
		print "Average runtime with " + str(s) + " points, " + str(e) + " edges, and " + str(sh) + " shakes is: " + str(sum(hmm_result[s][e][sh]["time"]) / len(hmm_result[s][e][sh]["time"]))
    for s in num_sample:
        for e in edge:
            for sh in shake:
		print "Average correct match with " + str(s) + " points, " + str(e) + " edges, and " + str(sh) + " shakes is: " + str(float(sum(hmm_result[s][e][sh]["match"])) / float(len(hmm_result[s][e][sh]["match"])))
    for s in num_sample:
        for e in edge:
            for sh in shake:
		count = 0
		for num in hmm_result[s][e][sh]["match"]:
		    if num == s:
			count += 1
                print "With " + str(s) + " points, " + str(e) + " edges, and " + str(sh) + " shakes, " + str(count) + " trajectories are matched with ground truth!"
