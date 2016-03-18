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
    if len(sys.argv) != 12:
        print "Input format: python trajectory_generator.py <server type> <protocol> <ip address> <port> <output repository> <city id> <num of traj> <num of sample> <sparsity> <shake> <accuracy level>"
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
        acc_level = int(sys.argv[11])
        output_prefix = output_repo + "ground_truth_" + num_traj + "traj_" + num_sample + "p_" + str(int(edge) * int(num_sample)) + "e_" + shake + "shake"
        output_src = output_prefix + "_" + str(acc_level) + "acc" + ".json"
        output = open(output_src, "a")
        num_ready = 0
        while num_ready < int(num_traj):
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
                # Choose the right file to output
                ground_truth_str = json.dumps(traj_set)
                accuracy = float(match_result[0]) / float(num_sample)
                if accuracy >= float(acc_level) / 100.0 and accuracy < float(acc_level) / 100.0 + 0.1:
                    if num_ready < int(num_traj):
                        output.write(ground_truth_str + "\n")
                        num_ready += 1
                    else:
                        remove_action = urllib2.urlopen(remove_url)
                else:
                    remove_action = urllib2.urlopen(remove_url)
            except urllib2.HTTPError, e:
                pass
                # print e.code
        output.close()
        print "Finished saving ground truth file!"
