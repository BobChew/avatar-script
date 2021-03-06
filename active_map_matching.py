import sys
import json
import urllib2
import random
import math
import time
from decimal import Decimal

DEBUG = False


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


def binary_index(index_list, low, high, height):
    if low < high:
        mid = (low + high) / 2
        if mid not in index_list:
            index_list[mid] = height
            index_list = binary_index(index_list, low, mid, height + 1)
            index_list = binary_index(index_list, mid, high, height + 1)
    return index_list


def normalize_prob_table(table):
    norm_prob_table = []
    for sample in table:
        sample_prob = []
        total = sum(sample)
        for prob in sample:
            sample_prob.append(prob / total)
        norm_prob_table.append(sample_prob)
    return norm_prob_table


def sort_by_entropy(prob_table):
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
    sorted_index = sorted(range(len(entropy_list)), key=lambda k: entropy_list[k], reverse=True)
    return sorted_index


def fixed_point_hmm(emission_prob, transition_prob, p_index, c_index):
    map_matching_prob = []
    chosen_index = []
    ini_prob = []
    if p_index == 0:
        for i in range(len(emission_prob[0])):
            if c_index == i:
                ini_prob.append(Decimal(emission_prob[0][i]) * Decimal(1.0))
            else:
                ini_prob.append(Decimal(emission_prob[0][i]) * Decimal(0.0))
    else:
        for first in emission_prob[0]:
            ini_prob.append(Decimal(first))
    map_matching_prob.append(ini_prob)
    for t in range(len(transition_prob)):
        state_prob = []
        prev_index = []
        for current in transition_prob[t]:
            candidate_prob = []
            for i in range(len(current)):
                if p_index == t:
                    if c_index == i:
                        value = Decimal(map_matching_prob[t][i]) * Decimal(current[i]) * Decimal(
                            emission_prob[t + 1][i]) * Decimal(1.0)
                    else:
                        value = Decimal(map_matching_prob[t][i]) * Decimal(current[i]) * Decimal(
                            emission_prob[t + 1][i]) * Decimal(0.0)
                else:
                    value = Decimal(map_matching_prob[t][i]) * Decimal(current[i]) * Decimal(emission_prob[t + 1][i])
                candidate_prob.append(value)
            state_prob.append(max(candidate_prob))
            prev_index.append(candidate_prob.index(max(candidate_prob)))
        chosen_index.append(prev_index)
        map_matching_prob.append(state_prob)
    hmm_path_index = []
    final_prob = map_matching_prob[len(map_matching_prob) - 1]
    if p_index == len(map_matching_prob) - 1:
        final_index = c_index
    else:
        final_index = final_prob.index(max(final_prob))
    confidence = final_prob[final_index]
    current_index = final_index
    hmm_path_index.append(final_index)
    for i in range(len(chosen_index), 0, -1):
        prev_index = chosen_index[i - 1][hmm_path_index[len(hmm_path_index) - 1]]
        hmm_path_index.append(prev_index)
        current_index = prev_index
    hmm_path_index.reverse()
    return [confidence, hmm_path_index]


def build_confidence_table(emission_prob, transition_prob):
    confidence_table = []
    for p in range(len(emission_prob)):
        sample_confidence = []
        for c in range(len(emission_prob[0])):
            if Decimal(emission_prob[p][c]) > Decimal(1e-300):
                result = fixed_point_hmm(emission_prob, transition_prob, p, c)
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


def model_change_table(emission_prob, transition_prob, path_index):
    model_change = []
    brute_force_match = build_confidence_table(emission_prob, transition_prob)
    for p in range(len(brute_force_match)):
        if brute_force_match[p][path_index[p]][1] is not None:
            fixed_confidence = brute_force_match[p][path_index[p]][0]
            path_change = []
            # confidence_change = []
            for c in range(len(brute_force_match[p])):
                if c != path_index[p] and brute_force_match[p][c][1] is not None:
                    # path_change.append(path_index_change(brute_force_match[p][c][1], path_index))
                    # confidence_change.append(brute_force_match[p][c][0] - fixed_confidence)
            # if len(path_change) > 0 and len(confidence_change) > 0:
                # avg_path_change = float(sum(path_change)) / float(len(path_change))
                # avg_confidence_change = sum(confidence_change) / Decimal(len(confidence_change))
                # model_change.append([avg_path_change, avg_confidence_change])
            # else:
                # print "The target point has no other choice!"
                # print brute_force_match[p]
		        # print emission_prob[p]
                # model_change.append([None, None])
                    index_diff = path_index_change(brute_force_match[p][c][1], path_index)
                    confidence_diff = abs(brute_force_match[p][c][0] - fixed_confidence)
                    path_change.append([index_diff, confidence_diff])
            if len(path_change) > 0:
                # We choose the smallest from the vector
                model_change.append(min(path_change))
            else:
                print "The target point has no other choice!"
                model_change.append([None, None])
        else:
            print "The chosen road is too far away from the target point!"
            # print brute_force_match[p]
    return model_change


if __name__ == "__main__":
    if len(sys.argv) != 11:
        # Reperform strategy includes: scan, modify
        # Selection strategy includes: forward, backward, binary, random, entropy_dist, entropy_confidence, max_change, min_change
        print "Input format: python trajectory_generator.py <server type> <protocol> <ip address> <port> <ground_truth_src> <city id> <user id> <reperform strategy> <selection strategy> <output prefix>"
    else:
        if sys.argv[1] == "celery":
            server_prefix = sys.argv[2] + "://" + sys.argv[3] + ":" + sys.argv[4] + "/"
        elif sys.argv[1] == "nginx":
            server_prefix = sys.argv[2] + "://" + sys.argv[3] + "/avatar/"
        ground_truth_src = sys.argv[5]
        city = int(sys.argv[6])
        uid = sys.argv[7]
        reperform_strategy = sys.argv[8]
        selection_strategy = sys.argv[9]
        output_prefix = sys.argv[10]
        if reperform_strategy not in ["scan", "modify"]:
            print "No such reperform strategy!"
            print "Reperform strategy inlcudes: scan, modify"
            exit()
        if selection_strategy not in ["forward", "binary", "random", "entropy_dist", "entropy_confidence", "max_change", "min_change"]:
            print "No such selection strategy!"
            print "Selection strategy includes: forward, binary, random, entropy_dist, entropy_confidence, max_change, min_change"
            exit()
        # Build ground truth index
        ground_truth_file = open(ground_truth_src, "r")
        result_file = open(output_prefix + "_" + reperform_strategy + "_" + selection_strategy + ".json", "a")
        time_file = open(output_prefix + "_" + reperform_strategy + "_" + selection_strategy + "_time.json", "a")
        # Record the number of right matches after each scan for each trajectory
        acc_table = []
        for line in ground_truth_file.readlines():
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
            url_hmm = server_prefix + "map-matching/perform/?city=" + str(city) + "&id=" + traj_id
            if DEBUG:
                print "Map matching url is: " + url_hmm
            map_matching_info = urllib2.urlopen(url_hmm)
            map_matching_result = json.load(map_matching_info)
            hmm_path = map_matching_result["path"]
            emission_prob = map_matching_result["emission_prob"]
            transition_prob = map_matching_result["transition_prob"]
            path_index = map_matching_result["path_index"]
            match_result = compare_result_with_truth(hmm_path, true_path)
            acc_vector.append(match_result[0])
            if DEBUG:
                print "The trajectory contains " + str(len(trace["p"])) + " samples. After initial map matching, " + str(match_result[0]) + " has been matched to the right road!"
            # Delete user history
            url_remove_history = server_prefix + "map-matching/remove_user_action_history_from_cache/?id=" + traj_id + "&uid=" + uid
            try:
                remove_history = urllib2.urlopen(url_remove_history)
            except urllib2.HTTPError, e:
                pass
            reperform_num = 0  # How many reperform map matching process should be done before the entire trajectory match the ground truth
            pass_num = 0  # How many points user think is mapped to the right road
            modify_num = 0  # How many points user thins is mapped to the wrong road
            task_time = []
            # Prepare question selection list
            if selection_strategy == "random":
                shuffle_index = []
                for i in range(len(trace["p"])):
                    shuffle_index.append(i)
                random.shuffle(shuffle_index)
                if DEBUG:
                    print shuffle_index
            if selection_strategy == "binary":
                bindex = {}
                bindex = binary_index(bindex, 1, len(trace["p"]) - 1, 1)
                shuffle_index = sorted(bindex.keys(), key=lambda k: bindex[k])
                shuffle_index = [0, len(trace["p"]) - 1] + shuffle_index
                if DEBUG:
                    print shuffle_index
            if selection_strategy == "entropy_dist":
                # url_get_emission = server_prefix + "map-matching/get_emission_table/?city=" + str(city) + "&id=" + traj_id
                # if DEBUG:
                    # print "Get emission table url is: " + url_get_emission
                # emission_info = urllib2.urlopen(url_get_emission)
                # emission_table = json.load(emission_info)
                for i in range(len(emission_prob)):
                    emission_prob[i] = filter(lambda x : x > 1e-300, emission_prob[i])
                sorted_index = sort_by_entropy(emission_prob)
                if DEBUG:
                    print sorted_index
            if selection_strategy == "entropy_confidence":
                brute_force_match = build_confidence_table(emission_prob, transition_prob)
                confidence_table = []
                for sample in brute_force_match:
                    sample_result = []
                    for result in sample:
                        if result[1] is not None:
                            sample_result.append(result[0])
                    confidence_table.append(sample_result)
                # print confidence_table
                sorted_index = sort_by_entropy(confidence_table)
                if DEBUG:
                    print sorted_index
            if selection_strategy in ["max_change", "min_change"]:
                # Convert string to int
                path_index = [int(index) for index in path_index]
                model_change = model_change_table(emission_prob, transition_prob, path_index)
                # For those points who have only one candidate road, put them at the bottom of the list
                for i in range(len(model_change)):
                    if model_change[i][0] is None:
                        if selection_strategy == "min_change":
                            model_change[i] = [float("inf"), Decimal("inf")]
                        elif selection_strategy == "max_change":
                            model_change[i] = [-float("inf"), -Decimal("inf")]
                # print model_change
                sorted_index = sorted(range(len(model_change)), key=lambda k: model_change[k])
                if selection_strategy == "max_change":
                    sorted_index.reverse()
                if DEBUG:
                    print sorted_index
            # Start reperform map matching process
            selection_num = 0
            merged_p = None
            merged_r = None
            while len(trace["p"]) != match_result[0]:
                # Sequentially choose samples along the trajectory to ask user
                if selection_strategy == "forward":
                    p_index = selection_num
                elif selection_strategy in ["random", "binary"]:
                    p_index = shuffle_index[selection_num]
                elif selection_strategy in ["entropy_dist", "entropy_confidence", "max_change", "min_change"]:
                    p_index = sorted_index[selection_num]
                # For modify strategy, if the point is mapped to the right road, no need to reperform map matching
                # For scan strategy, if the point is mapped to the right road, merge the point to reperform later
                # if reperform_strategy == "modify" and p_index not in match_result[1]:
                if p_index not in match_result[1]:
                    reperform_launch = 0
                else:
                    reperform_launch = 1
                sample_id = trace["p"][p_index]["id"]
                if reperform_strategy == "modify" or merged_p is None:
                    merged_p = sample_id
                    merged_r = true_path[p_index]
                else:
                    merged_p += "," + sample_id
                    merged_r += "," + true_path[p_index]
                if p_index not in match_result[1]:
                    pass_num += 1
                else:
                    modify_num += 1
                    # Start the reperform map matching process
                if reperform_launch == 1:
                    if DEBUG:
                        print "Reperform map matching at " + str(p_index) + "th point!"
                    if reperform_strategy == "modify":
                        reperform_num += 1
                    else:
                        reperform_num = selection_num + 1
                    # sample_id = trace["p"][p_index]["id"]
                    task_start = time.time()
                    url_with_label = server_prefix + "map-matching/perform_with_label/?city=" + str(
                        city) + "&id=" + traj_id + "&pid=" + merged_p + "&rid=" + merged_r + "&uid=" + uid
                    if DEBUG:
                        print str(reperform_num) + "th reperform map matching url is: " + url_with_label
                    remap_matching_info = urllib2.urlopen(url_with_label)
                    remap_matching_result = json.load(remap_matching_info)
                    hmm_path_with_label = remap_matching_result
                    task_end = time.time()
                    task_time.append(task_end - task_start)
                    match_result = compare_result_with_truth(hmm_path_with_label, true_path)
                    acc_vector.append(match_result[0])
                    # If the reperform task is finished before all points are scanned, fill the rest points with max accuracy
                    if reperform_strategy == "scan" and match_result[0] == len(trace["p"]):
                        while len(acc_vector) < len(trace["p"]) + 1:
                            acc_vector.append(acc_vector[len(acc_vector) - 1])
                    if DEBUG:
                        print "The trajectory contains " + str(len(trace["p"])) + " samples. After " + str(
                            reperform_num) + "th reperform map matching, " + str(
                            match_result[0]) + " samples has been matched to the right road!"
                    if reperform_strategy == "scan":
                        merged_p = None
                        merged_r = None
                # If the point is matched to the right road, copy the accuracy of the previous point
                elif reperform_strategy == "scan":
                    acc_vector.append(acc_vector[len(acc_vector) - 1])
                selection_num += 1
                if selection_num == len(trace["p"]) and len(trace["p"]) != match_result[0]:
                    if reperform_strategy == "scan":
                        print "After scanning all the points on trajectory " + traj_id + ", there is still something wrong with map matching result!"
                        for wrong_p_index in match_result[1]:
                            print "The " + str(wrong_p_index) + "th point is matched to road " + match_result[1][
                                wrong_p_index]
                        break
                    elif reperform_strategy == "modify":
                        selection_num = 0
            acc_table.append(acc_vector)
            result_vector = [traj_id, uid, reperform_num, pass_num, modify_num]
            time_vector = [traj_id, uid, task_time]
            if selection_strategy in ["random", "binary"]:
                result_vector.append(shuffle_index)
            if selection_strategy in ["entropy_dist", "entropy_confidence", "max_change", "min_change"]:
                result_vector.append(sorted_index)
            result_str = json.dumps(result_vector)
            result_file.write(result_str + "\n")
            time_str = json.dumps(time_vector)
            time_file.write(time_str + "\n")
            # output.write(traj_id + "," + uid + "," + reperform_strategy + "," + selection_strategy + "," + str(reperform_num) + "\n")
            if DEBUG:
                print str(
                    reperform_num) + " reperform map matching process are performed before the entire trajectory match the ground truth!"
        acc_file = open(output_prefix + "_" + reperform_strategy + "_" + selection_strategy + "_acc.json", "a")
        acc_str = json.dumps(acc_table)
        acc_file.write(acc_str)
        acc_file.close()
        result_file.close()
        time_file.close()
        ground_truth_file.close()
