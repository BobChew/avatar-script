import sys
import json
import urllib2
import random
import math
import time
import threading
from decimal import Decimal
import csv

DEBUG = False


map_matching_time = []
active_map_matching_time = []
selection_time = []
dynamic_selection_time = []
ini_acc_table = []
scan_percent = []
selection_acc = []
hit_rate = []
active_ratio = []
num_first_hit = []


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
                    # print p_index + ":" + rid
                    match_count += 1
                else:
                    wrong_match[int(p_index)] = rid
    return [match_count, wrong_match]


def compare_result_with_initial(result, initial):
    diff_index = []
    for fragment in result["road"]:
        # Skip the connected path between two samples
        if fragment["p"] is not None:
            rid = fragment["road"]["id"]
            fragment_index = fragment["p"].split(",")
            for p_index in fragment_index:
                if initial[int(p_index)] != rid:
                    # print p_index + ":" + rid
                    diff_index.append(int(p_index))
    return diff_index


def normalize_prob_table(table):
    norm_prob_table = []
    for sample in table:
        sample_prob = []
        total = sum(sample)
        for prob in sample:
            sample_prob.append(prob / total)
        norm_prob_table.append(sample_prob)
    return norm_prob_table


def get_entropy_list(prob_table):
    entropy_list = []
    # Normalize probability table
    norm_prob_table = normalize_prob_table(prob_table)
    # Calculate entropy
    for sample in norm_prob_table:
        entropy = Decimal(0.0)
        for prob in sample:
            # Shannon entropy: sum of -p * logbp, we take b = 2
            if Decimal(prob) > Decimal(1e-300):
                prob_l = math.log(Decimal(prob), 2)
                entropy -= Decimal(prob) * Decimal(prob_l)
        entropy_list.append(entropy)
    # sorted_index = sorted(range(len(entropy_list)), key=lambda k: entropy_list[k], reverse=True)
    return entropy_list


def fixed_point_hmm(emission_prob, transition_prob, p_index, c_index):
    map_matching_prob = []
    chosen_index = []
    ini_prob = []
    tmp_emission_prob = [prob[:] for prob in emission_prob]
    for t in range(len(tmp_emission_prob)):
        for i in range(len(tmp_emission_prob[t])):
            if t in p_index:
                if i == c_index[p_index.index(t)]:
                    tmp_emission_prob[t][i] = 1.0
                else:
                    tmp_emission_prob[t][i] = 0.0
    for first in tmp_emission_prob[0]:
        ini_prob.append(Decimal(first))
    map_matching_prob.append(ini_prob)
    for t in range(len(transition_prob)):
        state_prob = []
        prev_index = []
        for c in range(len(transition_prob[t])):
            candidate_prob = []
            for i in range(len(transition_prob[t][c])):
                value = Decimal(map_matching_prob[t][i]) * Decimal(transition_prob[t][c][i]) * Decimal(tmp_emission_prob[t + 1][c])
                candidate_prob.append(value)
            state_prob.append(max(candidate_prob))
            prev_index.append(candidate_prob.index(max(candidate_prob)))
        chosen_index.append(prev_index)
        map_matching_prob.append(state_prob)
    hmm_path_index = []
    final_prob = map_matching_prob[len(map_matching_prob) - 1]
    final_index = final_prob.index(max(final_prob))
    confidence = final_prob[final_index]
    hmm_path_index.append(final_index)
    for i in range(len(chosen_index), 0, -1):
        prev_index = chosen_index[i - 1][hmm_path_index[len(hmm_path_index) - 1]]
        hmm_path_index.append(prev_index)
    hmm_path_index.reverse()
    return [confidence, hmm_path_index]


def build_confidence_table(emission_prob, transition_prob, fixed_p, fixed_c):
    confidence_table = []
    for p in range(len(emission_prob)):
        sample_confidence = []
        for c in range(len(emission_prob[0])):
            if Decimal(emission_prob[p][c]) > Decimal(1e-300):
                if p not in fixed_p:
                    result = fixed_point_hmm(emission_prob, transition_prob, fixed_p + [p], fixed_c + [c])
                else:
                    result = fixed_point_hmm(emission_prob, transition_prob, fixed_p, fixed_c)
                sample_confidence.append(result)
            else:
                sample_confidence.append([0.0, None])
        confidence_table.append(sample_confidence)
    return confidence_table


def path_index_change(path1, path2):
    count = 0
    for p in range(len(path1)):
        if path1[p] != path2[p]:
            count += 1
    return count


def get_entropy_confidence_list(emission_prob, transition_prob, fixed_p, fixed_c):
    start = time.time()
    brute_force_match = build_confidence_table(emission_prob, transition_prob, fixed_p, fixed_c)
    end = time.time()
    confidence_table = []
    for sample in brute_force_match:
        sample_result = []
        for result in sample:
            if result[1] is not None:
                sample_result.append(result[0])
        confidence_table.append(sample_result)
    # print confidence_table
    weight_list = get_entropy_list(confidence_table)
    return weight_list


def model_change_table(emission_prob, transition_prob, path_index, fixed_p, fixed_c):
    model_change = []
    brute_force_match = build_confidence_table(emission_prob, transition_prob, fixed_p, fixed_c)
    for p in range(len(brute_force_match)):
        if brute_force_match[p][path_index[p]][1] is not None:
            fixed_confidence = brute_force_match[p][path_index[p]][0]
            path_change = []
            confidence_change = []
            for c in range(len(brute_force_match[p])):
                if c != path_index[p] and brute_force_match[p][c][1] is not None:
                    index_diff = path_index_change(brute_force_match[p][c][1], path_index)
                    confidence_diff = abs(brute_force_match[p][c][0] - fixed_confidence)
                    # We assume that: smaller confidence change and larger index change results in higher priority
                    if index_diff == 0:
                        path_diff = Decimal(1.0)
                    else:
                        path_diff = Decimal(confidence_diff) / Decimal(index_diff)
                    path_change.append(path_diff)
                    path_change.append(index_diff)
            if len(path_change) > 0:
                # We choose the smallest from the vector
                model_change.append(max(path_change))
            else:
                print "The target point has no other choice!"
                # print brute_force_match[p]
                # print emission_prob[p]
                model_change.append(Decimal("inf"))
        else:
            print "The chosen road is too far away from the target point!"
            # print brute_force_match[p]
    return model_change


def omit_change_table(omit_emission_prob, omit_transition_prob, path_index, fixed_p, fixed_c, traj):
    omit_change = []
    index_change = [0 for i in range(len(traj["trace"]["p"]))]
    if len(omit_emission_prob) == 0 and len(omit_transition_prob) == 0:
        query = True
    else:
        query = False
    for i in range(len(traj["trace"]["p"])):
        if query:
            # Create a new trajectory omitting the target sample
            url_omit = server_prefix + "traj/remove_point/?id=" + traj["id"] + "&pid=" + traj["trace"]["p"][i]["id"]
            omit_traj_info = urllib2.urlopen(url_omit)
            omit_traj = json.load(omit_traj_info)
            # Perform map matching and save the emission and transition probability tables in cache
            url_omit_hmm = server_prefix + "map-matching/perform/?city=" + str(city) + "&id=" + omit_traj["id"]
            omit_map_matching = urllib2.urlopen(url_omit_hmm)
            omit_map_matching_result = json.load(omit_map_matching)
            omit_emission_prob.append(omit_map_matching_result["emission_prob"])
            omit_transition_prob.append(omit_map_matching_result["transition_prob"])
            omit_path_index = omit_map_matching_result["path_index"]
            omit_path_index = [int(index) for index in omit_path_index]
            # Calculate confidence using the emission and transition probability tables
            # omit_result.append(fixed_point_hmm(omit_map_matching_result["emission_prob"], omit_map_matching_result["transition_prob"], [], []))
            # Delete the new trajectory omitting the target sample
            url_remove = server_prefix + "traj/remove/?id=" + omit_traj["id"]
            remove_traj = urllib2.urlopen(url_remove)
        else:
            fixed_p_omit = []
            fixed_c_omit = []
            for j in range(len(fixed_p)):
                # If the omitted point is before the fixed point, move the index of the fixed point ahead
                if i < fixed_p[j]:
                    fixed_p_omit.append(fixed_p[j] - 1)
                    fixed_c_omit.append(fixed_c[j])
                elif i > fixed_p[j]:
                    fixed_p_omit.append(fixed_p[j])
                    fixed_c_omit.append(fixed_c[j])
            # omit_result.append(fixed_point_hmm(omit_emission_prob[i], omit_transition_prob[i], fixed_p_omit, fixed_c_omit))
            omit_path_index = fixed_point_hmm(omit_emission_prob[i], omit_transition_prob[i], fixed_p_omit, fixed_c_omit)[1]
        # Insert the index of omitted point back, assuming the omitted point is not changed
        omit_path_index.insert(i, path_index[i])
        index_diff = path_index_change(omit_path_index, path_index)
        omit_change.append(index_diff)
        # Calculate the index change of each point
        for i in range(len(traj["trace"]["p"])):
            if omit_path_index[i] != path_index[i]:
                index_change[i] += 1
    # omit_result = [index_change[i] + omit_change[i] for i in range(len(index_change))]
    omit_result = [index_change[i] for i in range(len(index_change))]
    return omit_emission_prob, omit_transition_prob, omit_result


def single_trajectory_simulation(line, server_prefix, city, uid, selection_strategy, reduction_rate, active_strategy):
    ground_truth = json.loads(line)
    true_path = {}
    acc_vector = []
    for fragment in ground_truth["ground_truth"][0]:
        # Skip the connecting path between two samples
        if len(fragment[1]) != 0:
            for p_index in fragment[1]:
                true_path[p_index] = fragment[0]
    # print true_path
    traj_id = ground_truth["traj_id"][0]
    # Get the trace sorted by time stamp
    url_get_trace = server_prefix + "traj/get/?id=" + traj_id
    if DEBUG:
        print "Get trajectory url is: " + url_get_trace
    traj_info = urllib2.urlopen(url_get_trace)
    traj_result = json.load(traj_info)
    trace = traj_result["trace"]
    # Perform HMM map matching
    map_matching_start = time.time()
    url_hmm = server_prefix + "map-matching/perform/?city=" + str(city) + "&id=" + traj_id
    if DEBUG:
        print "Map matching url is: " + url_hmm
    map_matching_info = urllib2.urlopen(url_hmm)
    map_matching_result = json.load(map_matching_info)
    hmm_path = map_matching_result["path"]
    emission_prob = map_matching_result["emission_prob"]
    transition_prob = map_matching_result["transition_prob"]
    path_index = [int(index) for index in map_matching_result["path_index"]]
    candidate_rid = map_matching_result["candidate_rid"]
    confidence = map_matching_result["confidence"]
    map_matching_end = time.time()
    map_matching_time.append(map_matching_end - map_matching_start)
    match_result = compare_result_with_truth(hmm_path, true_path)
    acc_vector.append(match_result[0] - 2)
    ini_acc_table.append((float(match_result[0] - 2.0)) / (float(len(trace["p"])) - 2.0))
    initial_path = {}
    for fragment in hmm_path["road"]:
        # Skip the connected path between two samples
        if fragment["p"] is not None:
            rid = fragment["road"]["id"]
            fragment_index = fragment["p"].split(",")
            for p_index in fragment_index:
                initial_path[int(p_index)] = rid
    if DEBUG:
        print "The trajectory contains " + str(len(trace["p"])) + " samples. After initial map matching, " + str(match_result[0]) + " has been matched to the right road!"
    # Delete user history
    url_remove_history = server_prefix + "map-matching/remove_user_action_history_from_cache/?id=" + traj_id + "&uid=" + uid
    try:
        remove_history = urllib2.urlopen(url_remove_history)
    except urllib2.HTTPError, e:
        pass
    # Prepare question selection list
    selection_start = time.time()
    weight_list = []
    fixed_p = []
    fixed_r = []
    # Add the first and last point into fixed point list
    fixed_p.append(0)
    fixed_r.append(candidate_rid[0].index(true_path[0]))
    fixed_p.append(len(trace["p"]) - 1)
    fixed_r.append(candidate_rid[len(trace["p"]) - 1].index(true_path[len(trace["p"]) - 1]))
    # Compute the global order of question selection
    if selection_strategy == "forward":
        unit = Decimal(2) / (Decimal(len(emission_prob)) * (Decimal(len(emission_prob) + 1)))
        for i in range(len(emission_prob)):
            weight_list.append(unit * Decimal(i + 1))
    if selection_strategy == "random":
        shuffle_index = []
        for i in range(len(trace["p"])):
            shuffle_index.append(i)
        random.shuffle(shuffle_index)
        unit = Decimal(2) / (Decimal(len(emission_prob)) * (Decimal(len(emission_prob) + 1)))
        for i in range(len(emission_prob)):
            weight_list.append(unit * Decimal(shuffle_index.index(i) + 1))
    if selection_strategy == "entropy_dist":
        for i in range(len(emission_prob)):
            emission_prob[i] = filter(lambda x : x > 1e-300, emission_prob[i])
        weight_list = get_entropy_list(emission_prob)
        if DEBUG:
            print weight_list
    if selection_strategy == "entropy_confidence":
        weight_list = get_entropy_confidence_list(emission_prob, transition_prob, fixed_p, fixed_r)
        if DEBUG:
            print weight_list
    if selection_strategy == "min_change":
        # Convert string to int
        weight_list = model_change_table(emission_prob, transition_prob, path_index, fixed_p, fixed_r)
        # print weight_list
        # sorted_index = sorted(range(len(model_change)), key=lambda k: model_change[k])
        if DEBUG:
            print weight_list
    if selection_strategy == "omit":
        # Convert string to int
        omit_emission_table, omit_transition_table, weight_list = omit_change_table([], [], path_index, fixed_p, fixed_r, traj_result)
    if selection_strategy == "omit_entropy":
        omit_emission_table, omit_transition_table, weight_list = omit_change_table([], [], path_index, fixed_p, fixed_r, traj_result)
        print weight_list
        min_weight = min(weight_list)
        if max(weight_list) > min_weight:
            omit_level = True
        else:
            omit_level = False
        # If omit strategy does not work from the beginning, or the strategy is global, change to entropy confidence strategy
            # print "Change to entropy confidence based selection strategy!"
        entropy_weight_list = get_entropy_confidence_list(emission_prob, transition_prob, [], [])
        max_entropy = max(entropy_weight_list)
        # Append entropy to each weight if omit strategy cannot distinguish them
        for i in range(len(weight_list)):
            weight_list[i] = weight_list[i] + max_entropy if weight_list[i] > min_weight else entropy_weight_list[i]
    # Place the first and last point at the bottom of the list
    if selection_strategy in ["entropy_dist", "entropy_confidence", "omit", "omit_entropy"]:
        weight_list[0] = -Decimal("inf")
        weight_list[len(trace["p"]) - 1] = -Decimal("inf")
    elif selection_strategy in ["forward", "random", "min_change"]:
        weight_list[0] = Decimal("inf")
        weight_list[len(trace["p"]) - 1] = Decimal("inf")
    selection_end = time.time()
    selection_time.append(selection_end - selection_start)
    first_hit = False # When does the first selected point is actually matched wrongly
    # num_wrong = len(trace["p"]) - match_result[0] # Total number of wrong points
    num_wrong = len(match_result[1].keys())
    # Start reperform map matching process
    merged_p = trace["p"][0]["id"] + "," + trace["p"][len(trace["p"]) - 1]["id"]
    merged_r = true_path[0] + "," + true_path[len(trace["p"]) - 1]
    scanned_list = []
    num_hit = 0
    while len(trace["p"]) != match_result[0]:
        # Sequentially choose samples along the trajectory to ask user
        if selection_strategy in ["entropy_dist", "entropy_confidence", "omit", "omit_entropy"]:
            p_index = weight_list.index(max(weight_list))
        elif selection_strategy in ["forward", "random", "min_change"]:
            p_index = weight_list.index(min(weight_list))
        scanned_list.append(p_index)
        if p_index not in match_result[1]:
            reperform_launch = 0
        else:
            reperform_launch = 1
        sample_id = trace["p"][p_index]["id"]
        merged_p += "," + sample_id
        merged_r += "," + true_path[p_index]
        # Start the reperform map matching process
        if reperform_launch == 1:
            num_hit += 1
            if not first_hit:
                num_first_hit.append(len(scanned_list))
                first_hit = True
            if DEBUG:
                print "Reperform map matching at " + str(p_index) + "th point!"
            # sample_id = trace["p"][p_index]["id"]
            task_start = time.time()
            url_with_label = server_prefix + "map-matching/perform_with_label/?city=" + str(city) + "&id=" + traj_id + "&pid=" + merged_p + "&rid=" + merged_r + "&uid=" + uid
            if DEBUG:
                print str(len(scanned_list) - 1) + "th reperform map matching url is: " + url_with_label
            remap_matching_info = urllib2.urlopen(url_with_label)
            remap_matching_result = json.load(remap_matching_info)
            hmm_path_with_label = remap_matching_result["path"]
            transition_prob = remap_matching_result["transition_prob"]
            candidate_rid = remap_matching_result["candidate_rid"]
            task_end = time.time()
            active_map_matching_time.append(task_end - task_start)
            match_result = compare_result_with_truth(hmm_path_with_label, true_path)
            if DEBUG:
                print "The trajectory contains " + str(len(trace["p"])) + " samples. After " + str(
                    len(scanned_list)) + "th reperform map matching, " + str(
                    match_result[0]) + " samples has been matched to the right road!"
            # merged_p = "0" + "," + str(len(trace["p"]) - 1)
            # merged_r = true_path[0] + "," + true_path[len(trace["p"]) - 1]
            if "step" in active_strategy:
                path_index = [int(index) for index in remap_matching_result["path_index"]]
        # Record the temporary accuracy
        acc_vector.append(match_result[0] - 2)
        # If the reperform task is finished before all points are scanned, fill the rest points with max accuracy
        if match_result[0] == len(trace["p"]):
            # Total length of acc_vector should be number of points plus one as initial acc
            while len(acc_vector) < len(trace["p"]) - 1:
                acc_vector.append(acc_vector[len(acc_vector) - 1])
        # Dynamically tune the weight list
        dynamic_start = time.time()
        if "fix" in active_strategy:
            # Find the index of chosen road for the chosen point
            # find_candidate_url = server_prefix + "map-matching/find_candidates/?city=" + str(city) + "&lat=" + str(trace["p"][p_index]["p"]["lat"]) + "&lng=" + str(trace["p"][p_index]["p"]["lng"])
            # if DEBUG:
                # print "Finding the " + str(len(scanned_list) - 1) + "the point's candidate roads url is: " + find_candidate_url
            # candidate_info = urllib2.urlopen(find_candidate_url)
            # candidate_set = json.load(candidate_info)
            # r_index = None
            # if len(candidate_set) >= len(emission_prob[0]):
                # for i in range(len(emission_prob[0])):
                    # if candidate_set[i] == true_path[p_index]:
                        # r_index = i
                        # break
            # if r_index is None:
                # r_index = len(emission_prob[0]) - 1
            if true_path[p_index] in candidate_rid[p_index]:
                r_index = candidate_rid[p_index].index(true_path[p_index])
                # If the chosen road is too far away, set the emission probability to 1
                if emission_prob[p_index][r_index] == Decimal(1e-300):
                    print "The chosen road is not in top 10"
                    emission_prob[p_index][r_index] = Decimal(1.0)
            else:
                r_index = len(emission_prob[0]) - 1
                print "This case happens!"
            # Add (p_index, r_index) into fixed set
            fixed_p.append(p_index)
            fixed_r.append(r_index)
            # print fixed_p
            # print fixed_r
            if selection_strategy == "entropy_confidence":
                weight_list = get_entropy_confidence_list(emission_prob, transition_prob, fixed_p, fixed_r)
            elif selection_strategy == "min_change":
                weight_list = model_change_table(emission_prob, transition_prob, path_index, fixed_p, fixed_r)
            elif selection_strategy == "omit":
                omit_emission_table, omit_transition_table, weight_list = omit_change_table(omit_emission_table, omit_transition_table, path_index, fixed_p, fixed_r, traj_result)
            elif selection_strategy == "omit_entropy":
                if omit_level:
                    omit_emission_table, omit_transition_table, weight_list = omit_change_table(omit_emission_table, omit_transition_table, path_index, fixed_p, fixed_r, traj_result)
                    # print weight_list
                    min_weight = min(weight_list)
                    for scanned_p in scanned_list:
                        weight_list[scanned_p] = -Decimal("inf")
                    if max(weight_list) == min_weight:
                        # If omit strategy does not work from the beginning, change to entropy confidence strategy
                        print "Change to entropy confidence based selection strategy!"
                        weight_list = get_entropy_confidence_list(emission_prob, transition_prob, fixed_p, fixed_r)
                        omit_level = False
                else:
                    weight_list = get_entropy_confidence_list(emission_prob, transition_prob, fixed_p, fixed_r)
        if "influence" in active_strategy:
            if reperform_launch == 1:
                modified_list = compare_result_with_initial(hmm_path_with_label, initial_path)
                for modified_index in modified_list:
                    if selection_strategy in ["entropy_dist", "entropy_confidence", "omit"]:
                        weight_list[modified_index] *= reduction_rate
                    if selection_strategy in ["forward", "random", "min_change"]:
                        weight_list[modified_index] /= reduction_rate
        if "side" in active_strategy:
            # Right now we hard code the range of influenced points
            p_min = p_index - 1 if p_index - 1 >= 0 else 0
            p_max = p_index + 1 if p_index + 1 <= len(trace["p"]) - 1 else len(trace["p"]) - 1
            for side_index in range(p_min, p_max + 1):
                weight_list[side_index] *= reduction_rate
        for scanned_p in scanned_list:
            if selection_strategy in ["entropy_dist", "entropy_confidence", "omit", "omit_entropy"]:
                weight_list[scanned_p] = -Decimal("inf")
                weight_list[0] = -Decimal("inf")
                weight_list[len(trace["p"]) - 1] = -Decimal("inf")
            elif selection_strategy in ["forward", "random", "min_change"]:
                weight_list[scanned_p] = Decimal("inf")
                weight_list[0] = Decimal("inf")
                weight_list[len(trace["p"]) - 1] = Decimal("inf")
        if DEBUG:
            print weight_list
        dynamic_end = time.time()
        dynamic_selection_time.append(dynamic_end - dynamic_start)
        if len(scanned_list) == len(trace["p"]) and len(trace["p"]) != match_result[0]:
            print "After scanning all the points on trajectory " + traj_id + ", there is still something wrong with map matching result!"
            for wrong_p_index in match_result[1]:
                print "The " + str(wrong_p_index) + "th point is matched to road " + match_result[1][wrong_p_index]
            break
    print scanned_list
    scan_percent.append(float(len(scanned_list)) / (float(len(trace["p"])) - 2.0))
    if len(scanned_list) != 0:
        selection_acc.append(float(num_wrong) / float(len(scanned_list)))
        hit_rate.append(float(num_hit) / float(len(scanned_list)))
    else:
        selection_acc.append(1.0)
        hit_rate.append(1.0)
    if num_wrong != num_hit:
        print str(num_wrong) + " points are wrong initially, but " + str(num_hit) + " points are fixed in general!"
    active_ratio.append(float(num_hit) / float(num_wrong))
    if DEBUG:
        print str(len(scanned_list)) + " reperform map matching process are performed before the entire trajectory match the ground truth!"
    print "Finish!"
    return acc_vector


if __name__ == "__main__":
    if len(sys.argv) != 12:
        # Reperform strategy includes: scan, modify
        # Selection strategy includes: forward, backward, binary, random, entropy_dist, entropy_confidence, max_change, min_change
        print "Input format: python trajectory_generator.py <server type> <protocol> <ip address> <port> <ground_truth_src> <city id> <user id> <selection strategy> <reduction rate> <active strategy> <output prefix>"
    else:
        group_start = time.time()
        if sys.argv[1] == "celery":
            server_prefix = sys.argv[2] + "://" + sys.argv[3] + ":" + sys.argv[4] + "/"
        elif sys.argv[1] == "nginx":
            server_prefix = sys.argv[2] + "://" + sys.argv[3] + "/avatar/"
        ground_truth_src = sys.argv[5]
        city = int(sys.argv[6])
        uid = sys.argv[7]
        selection_strategy = sys.argv[8]
        reduction_rate = Decimal(sys.argv[9])
        active_strategy = sys.argv[10].split(",")
        output_prefix = sys.argv[11]
        # Record the number of right matches after each scan for each trajectory
        acc_table = []
        # Build ground truth index
        ground_truth_file = open(ground_truth_src, "r")
        threads = []
        for line in ground_truth_file.readlines():
            acc_vector = single_trajectory_simulation(line, server_prefix, city, uid, selection_strategy, reduction_rate, active_strategy)
            acc_table.append(acc_vector)
        ground_truth_file.close()
        acc_file = open(output_prefix + "_" + selection_strategy + "_" + sys.argv[10] + "_acc.json", "a")
        acc_str = json.dumps(acc_table)
        acc_file.write(acc_str)
        acc_file.close()
        # Calculate the average runtime
        avg_hmm_time = float(sum(map_matching_time)) / float(len(map_matching_time))
        avg_active_hmm_time = float(sum(active_map_matching_time)) / float(len(active_map_matching_time))
        avg_time = float(sum(selection_time)) / float(len(selection_time))
        avg_dynamic_time = float(sum(dynamic_selection_time)) / float(len(dynamic_selection_time))
        # Calculate the statstic result
        avg_ini_acc = float(sum(ini_acc_table)) / float(len(ini_acc_table))
        scan_percent.sort()
        avg_scan_percent = float(sum(scan_percent)) / float(len(scan_percent))
        med_scan_percent = scan_percent[int(len(scan_percent) * 0.5)]
        quad25_scan_percent = scan_percent[int(len(scan_percent) * 0.25)]
        quad75_scan_percent = scan_percent[int(len(scan_percent) * 0.75)]
        selection_acc.sort()
        avg_selection_acc = float(sum(selection_acc)) / float(len(selection_acc))
        med_selection_acc = selection_acc[int(len(selection_acc) * 0.5)]
        quad25_selection_acc = selection_acc[int(len(selection_acc) * 0.25)]
        quad75_selection_acc = selection_acc[int(len(selection_acc) * 0.75)]
        hit_rate.sort()
        avg_hit_rate = float(sum(hit_rate)) / float(len(hit_rate))
        med_hit_rate = hit_rate[int(len(hit_rate) * 0.5)]
        quad25_hit_rate = hit_rate[int(len(hit_rate) * 0.25)]
        quad75_hit_rate = hit_rate[int(len(hit_rate) * 0.75)]
        avg_first_hit = float(sum(num_first_hit)) / float(len(num_first_hit))
        active_ratio.sort()
        avg_active_ratio = float(sum(active_ratio)) / float(len(active_ratio))
        med_active_ratio = active_ratio[int(len(active_ratio) * 0.5)]
        quad25_active_ratio = active_ratio[int(len(active_ratio) * 0.25)]
        quad75_active_ratio = active_ratio[int(len(active_ratio) * 0.75)]
        avg_first_hit = float(sum(num_first_hit)) / float(len(num_first_hit))
        group_end = time.time()
        print "The entire group experiment takes: " + str(group_end - group_start) + " seconds!"
        print "Map matching takes " + str(avg_hmm_time) + " seconds in average!"
        print "Active map matching takes " + str(avg_active_hmm_time) + " seconds in average!"
        print selection_strategy + " selection method takes " + str(avg_time) + " seconds in average to generate an ordered sequence!"
        print selection_strategy + " selection method takes " + str(avg_dynamic_time) + " seconds in average to re-order the sequence!"
        print "The average accuracy of the initial map matching is " + str(avg_ini_acc) + "!"
        print str(avg_first_hit) + " points in average are selected before the first wrong point is found!"
        print "Scan percent: average: " + str(avg_scan_percent) + " medium: " + str(med_scan_percent) + " 25 quad: " + str(quad25_scan_percent) + " 75 quad: " + str(quad75_scan_percent)
        print "Selection accuracy: average: " + str(avg_selection_acc) + " medium: " + str(med_selection_acc) + " 25 quad: " + str(quad25_selection_acc) + " 75 quad: " + str(quad75_selection_acc)
        print "Hit rate: average: " + str(avg_hit_rate) + " medium: " + str(med_hit_rate) + " 25 quad: " + str(quad25_hit_rate) + " 75 quad: " + str(quad75_hit_rate)
        print "Active ratio: average: " + str(avg_active_ratio) + " medium: " + str(med_active_ratio) + " 25 quad: " + str(quad25_active_ratio) + " 75 quad: " + str(quad75_active_ratio)
        # Write the individual results into output file
#        if "influence" in active_strategy:
#            output_file = "../result/active_result_" + selection_strategy + "_decay.csv"
#        else:
#            output_file = "../result/active_result_" + selection_strategy + ".csv"
#        csvfile = file(output_file, "a")
#        writer = csv.writer(csvfile)
#        for i in range(len(scan_percent)):
#            result_row = [ini_acc_table[i], scan_percent[i], selection_acc[i], map_matching_time[i], avg_active_hmm_time, selection_time[i], avg_dynamic_time]
#            writer.writerow(result_row)
#        csvfile.close()
