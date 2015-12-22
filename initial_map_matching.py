import sys
import json
import urllib2
import random
import math
import time
from decimal import Decimal


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
    if len(sys.argv) != 7:
        print "Input format: python initial_map_matching.py <protocol> <ip address> <port> <ground_truth_src> <city> <output file>"
    else:
        # server_prefix = sys.argv[1] + "://" + sys.argv[2] + ":" + sys.argv[3] + "/avatar/"
        server_prefix = sys.argv[1] + "://" + sys.argv[2] + "/avatar/"
        ground_truth_src = sys.argv[4]
        if sys.argv[5] in ["shenzhen", "Shenzhen"]:
            city = 3
        else:
            print "No map for this city!"
            exit()
        result_src = sys.argv[6]
        output = open(result_src, "a")
        output.write("trajectory_id,path_length,map_matching_time,number_of_right_matches" + "\n")
        # Build ground truth index
        ground_truth_file = open(ground_truth_src, "r")
        for line in ground_truth_file.readlines():
            ground_truth = json.loads(line)
            true_path = {}
            for fragment in ground_truth["ground_truth"][0]:
                # Skip the connecting path between two samples
                if len(fragment[1]) != 0:
                    for p_index in fragment[1]:
                        true_path[p_index] = fragment[0]
            # print true_path
            traj_id = ground_truth["traj_id"][0]
            path_length = ground_truth["path_length"][0]
            # Perform HMM map matching
            task_start = time.time()
            url_hmm = server_prefix + "map-matching/perform/?city=" + str(city) + "&id=" + traj_id
            print "Map matching url is: " + url_hmm
            map_matching_info = urllib2.urlopen(url_hmm)
            map_matching_result = json.load(map_matching_info)
            hmm_path = map_matching_result["traj"]["path"]
            task_end = time.time()
            match_result = compare_result_with_truth(hmm_path, true_path)
            # print "The trajectory contains " + str(len(trace["p"])) + " samples. After initial map matching, " + str(match_result[0]) + " has been matched to the right road!";
            output.write(
                traj_id + "," + str(path_length) + "," + str(task_end - task_start) + "," + str(match_result[0]) + "\n")
        output.close()
        print "Finished!"
