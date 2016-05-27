import sys
import json
import urllib2
import random
import math
import time
import csv
from decimal import Decimal
import question_selection_evaluation


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
    noise_traj_id = []
    leven_paths = []
    orig_traj_id = []
    # Build ground truth index
    ground_truth_file = open("test_ground_truth.json", "r")
    for line in ground_truth_file.readlines():
        ground_truth = json.loads(line)
        noise_traj_id.append(ground_truth["traj_id"][0])
        true_path = []
        for fragment in ground_truth["ground_truth"][0]:
            true_path.append(fragment[0])
        leven_paths.append(true_path)
    orig_traj_file = open("test_origin_traj_id.csv", "r")
    traj_list = csv.reader(orig_traj_file)
    for traj in traj_list:
        orig_traj_id.append(traj[0])
    for i in range(len(orig_traj_id)):
        # Perform HMM map matching
        url_hmm = "http://127.0.0.1/api/avatar/map-matching/perform/?city=3&id=" + orig_traj_id[i]
        map_matching_info = urllib2.urlopen(url_hmm)
        map_matching_result = json.load(map_matching_info)
        hmm_path = map_matching_result["path"]
        hmm_path_rids = []
        for fragment in hmm_path["road"]:
            hmm_path_rids.append(fragment["road"]["id"])
        leven_result = question_selection_evaluation.levenshtein_distance(hmm_path_rids, leven_paths[i])
        if leven_result != 0:
            print "Original map matching wrong: " + str(i + 1)
        noise_url_hmm = "http://127.0.0.1/api/avatar/map-matching/perform/?city=3&id=" + noise_traj_id[i]
        noise_map_matching_info = urllib2.urlopen(noise_url_hmm)
        noise_map_matching_result = json.load(noise_map_matching_info)
        noise_hmm_path = noise_map_matching_result["path"]
        noise_hmm_path_rids = []
        for fragment in noise_hmm_path["road"]:
            noise_hmm_path_rids.append(fragment["road"]["id"])
        noise_leven_result = question_selection_evaluation.levenshtein_distance(hmm_path_rids, noise_hmm_path_rids)
        if noise_leven_result == 0:
            print "Noise path is right: " + str(i + 1)
