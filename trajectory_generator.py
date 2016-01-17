import urllib2
import json
import sys
import time


def compare_result_with_truth(result, truth):
    match_count = 0
    wrong_match = {}
    for fragment in result["road"]:
        # Skip the connected path between two samples
        if fragment["p"] is not None:
            rid = fragment["road"]["id"]
            fragment_index = fragment["p"].split(",")
            for p_index in fragment_index:
                if truth[int(p_index)] == rid:
                    #		    print p_index + ":" + rid
                    match_count += 1
                else:
                    wrong_match[int(p_index)] = rid
    return [match_count, wrong_match]


if __name__ == "__main__":
    if len(sys.argv) != 10:
        print "Input format: python trajectory_generator.py <protocol> <ip address> <port> <output repository> <city id> <num of traj> <num of sample> <num of edges> <shake>"
    else:
        # server_prefix = sys.argv[1] + "://" + sys.argv[2] + ":" + sys.argv[3] + "/avatar/"
        server_prefix = sys.argv[1] + "://" + sys.argv[2] + "/avatar/"
        output_repo = sys.argv[4]
        num_traj = sys.argv[6]
        num_sample = sys.argv[7]
        edge = sys.argv[8]
        shake = sys.argv[9]
        output_prefix = output_repo + "ground_truth_" + num_traj + "traj_" + num_sample + "p_" + str(int(edge) * int(num_sample)) + "e_" + shake + "shake"
        # Seperater trajectroies with different map matching accuracy range
        output_low_acc = open(output_repo + "_low_acc" + ".json", "a")      # map matching accuracy < 60%
        output_mid_acc = open(output_repo + "_mid_acc" + ".json", "a")      # map matching accuracy >= 60% < 80%
        output_high_acc = open(output_repo + "_high_acc" + ".json", "a")     # map matching accuracy >= 80%
        num_low_acc = 0
        num_mid_acc = 0
        num_high_acc = 0
        while num_low_acc < int(num_traj) or num_mid_acc < int(num_traj) or num_high_acc < int(num_traj):
            url = server_prefix + "simulator/generate_syn_traj/?city=" + sys.argv[5] + "&traj=1&point=" + num_sample + "&edge=" + str(int(edge) * int(num_sample)) + "&shake=" + shake
            # print "Trajectory generator url is: " + url
            try:
                # Generate a trajectory
                traj_info = urllib2.urlopen(url)
                traj_set = json.load(traj_info)
                # Build ground truth index
                true_path = {}
                for fragment in traj_set["ground_truth"][0]:
                    # Skip the connecting path between two samples
                    if len(fragment[1]) != 0:
                        for p_index in fragment[1]:
                            true_path[p_index] = fragment[0]
                # print true_path
                traj_id = traj_set["traj_id"][0]
                path_length = traj_set["path_length"][0]
                # Perform HMM map matching
                task_start = time.time()
                url_hmm = server_prefix + "map-matching/perform/?city=" + sys.argv[5] + "&id=" + traj_id
                print "Map matching url is: " + url_hmm
                map_matching_info = urllib2.urlopen(url_hmm)
                map_matching_result = json.load(map_matching_info)
                hmm_path = map_matching_result["path"]
                task_end = time.time()
                match_result = compare_result_with_truth(hmm_path, true_path)
                # Choose the right file to output
                ground_truth_str = json.dumps(traj_set)
                accuracy = float(match_result[0]) / float(num_sample)
                if accuracy < 0.6:
                    output_low_acc.write(ground_truth_str + "\n")
                    num_low_acc += 1
                elif accuracy < 0.8:
                    output_mid_acc.write(ground_truth_str + "\n")
                    num_mid_acc += 1
                # If the map matching result is exactly right, no need to perform active map matching
                elif accuracy < 1.0:
                    output_high_acc.write(ground_truth_str + "\n")
                    num_high_acc += 1
            # print "Created " + str(num_complete_traj) + " trajectories..."
            except urllib2.HTTPError, e:
                pass
                # print e.code
        output_low_acc.close()
        output_mid_acc.close()
        output_high_acc.close()
        print "Finished saving ground truth file!"
