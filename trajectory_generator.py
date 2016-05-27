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
    if len(sys.argv) != 11:
        print "Input format: python trajectory_generator.py <server type> <protocol> <ip address> <port> <output repository> <city id> <num of traj> <num of sample> <sparsity> <shake>"
    else:
        if sys.argv[1] == "celery":
            server_prefix = sys.argv[2] + "://" + sys.argv[3] + ":" + sys.argv[4] + "/"
        elif sys.argv[1] == "nginx":
            server_prefix = sys.argv[2] + "://" + sys.argv[3] + "/avatar/"
        output_repo = sys.argv[5]
        city_id = sys.argv[6]
        num_traj = sys.argv[7]
        num_sample = sys.argv[8]
        edge = sys.argv[9]
        shake = sys.argv[10]
        output_prefix = output_repo + "ground_truth_" + num_traj + "traj_" + num_sample + "p_" + str(int(edge) * int(num_sample)) + "e_" + shake + "shake"
        # Seperater trajectroies with different map matching accuracy range
        output_50acc = open(output_prefix + "_50acc" + ".json", "a")    # map matching accuracy 50% - 60%
        output_60acc = open(output_prefix + "_60acc" + ".json", "a")    # map matching accuracy 60% - 70%
        output_70acc = open(output_prefix + "_70acc" + ".json", "a")    # map matching accuracy 70% - 80%
        output_80acc = open(output_prefix + "_80acc" + ".json", "a")    # map matching accuracy 80% - 90%
        output_90acc = open(output_prefix + "_90acc" + ".json", "a")    # map matching accuracy 90% - 100%
        num_50acc = 0
        num_60acc = 0
        num_70acc = 0
        num_80acc = 0
        num_90acc = 0
        while num_50acc < int(num_traj) or num_60acc < int(num_traj) or num_70acc < int(num_traj) or num_80acc < int(num_traj) or num_90acc < int(num_traj):
            url = server_prefix + "simulator/generate_syn_traj/?city=" + city_id + "&traj=1&point=" + num_sample + "&edge=" + str(int(edge) * int(num_sample)) + "&shake=" + shake
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
                url_hmm = server_prefix + "map-matching/perform/?city=" + city_id + "&id=" + traj_id
                # print "Map matching url is: " + url_hmm
                map_matching_info = urllib2.urlopen(url_hmm)
                map_matching_result = json.load(map_matching_info)
                hmm_path = map_matching_result["path"]
                task_end = time.time()
                match_result = compare_result_with_truth(hmm_path, true_path)
                remove_url = server_prefix + "traj/remove/?id=" + traj_id
                # If the first or last point is wrong, don't use this trajectory
                if 0 in match_result[1] or int(num_sample) - 1 in match_result[1]:
                    remove_action = urllib2.urlopen(remove_url)
                else:
                    # Choose the right file to output
                    ground_truth_str = json.dumps(traj_set)
                    accuracy = (float(match_result[0]) - 2.0) / (float(num_sample) - 2.0)
                    if accuracy < 0.6:
                        if accuracy >= 0.5:
                            if num_50acc < int(num_traj):
                                output_50acc.write(ground_truth_str + "\n")
                                num_50acc += 1
                            else:
                                remove_action = urllib2.urlopen(remove_url)
                    elif accuracy < 0.7:
                        if num_60acc < int(num_traj):
                            output_60acc.write(ground_truth_str + "\n")
                            num_60acc += 1
                        else:
                            remove_action = urllib2.urlopen(remove_url)
                    # If the map matching result is exactly right, no need to perform active map matching
                    elif accuracy < 0.8:
                        if num_70acc < int(num_traj):
                            output_70acc.write(ground_truth_str + "\n")
                            num_70acc += 1
                        else:
                            remove_action = urllib2.urlopen(remove_url)
                    elif accuracy < 0.9:
                        if num_80acc < int(num_traj):
                            output_80acc.write(ground_truth_str + "\n")
                            num_80acc += 1
                        else:
                            remove_action = urllib2.urlopen(remove_url)
                    elif accuracy < 1.0:
                        if num_90acc < int(num_traj):
                            output_90acc.write(ground_truth_str + "\n")
                            num_90acc += 1
                        else:
                            remove_action = urllib2.urlopen(remove_url)
                    else:
                        remove_action = urllib2.urlopen(remove_url)
            # print "Created " + str(num_complete_traj) + " trajectories..."
            except urllib2.HTTPError, e:
                pass
                # print e.code
        output_50acc.close()
        output_60acc.close()
        output_70acc.close()
        output_80acc.close()
        output_90acc.close()
        print "Finished saving ground truth file!"
