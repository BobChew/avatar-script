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


def compare_result_in_path(result, truth):
    match_count = 0
    wrong_match = {}
    for fragment in result["road"]:
        # Skip the connected path between two samples
        if fragment["p"] is not None:
            rid = fragment["road"]["id"]
            fragment_index = fragment["p"].split(",")
            if rid in truth:
                match_count += len(fragment_index)
            else:
                for p_index in fragment_index:
                    wrong_match[int(p_index)] = rid
    return [match_count, wrong_match]


def jaccard_distance(result, truth):
    j_union = set(result).union(set(truth))
    j_inter = set(result).intersection(set(truth))
    dis = 1.0 - float(len(j_inter)) / float(len(j_union))
    return dis


def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return current[n]


def levenshtein_distance(first, second):
    """Find the Levenshtein distance between two strings."""
    if len(first) > len(second):
        first, second = second, first
    if len(second) == 0:
        return len(first)
    first_length = len(first) + 1
    second_length = len(second) + 1
    distance_matrix = [range(second_length) for x in range(first_length)]
    for i in range(1, first_length):
        for j in range(1, second_length):
            deletion = distance_matrix[i-1][j] + 1
            insertion = distance_matrix[i][j-1] + 1
            substitution = distance_matrix[i-1][j-1]
            if first[i-1] != second[j-1]:
                substitution += 1
            distance_matrix[i][j] = min(insertion, deletion, substitution)
    return distance_matrix[first_length-1][second_length-1]


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
    # map_matching_prob = []
    # chosen_index = []
    # ini_prob = []
    # tmp_emission_prob = emission_prob[:]
    # for t in range(len(tmp_emission_prob)):
    #     for i in range(len(tmp_emission_prob[t])):
    #         if p_index == t:
    #             if c_index == i:
    #                 tmp_emission_prob[t][i] = 1.0
    #             else:
    #                 tmp_emission_prob[t][i] = 0.0
    # for first in tmp_emission_prob[0]:
    #     ini_prob.append(Decimal(first))
    # map_matching_prob.append(ini_prob)
    # for t in range(len(transition_prob)):
    #     state_prob = []
    #     prev_index = []
    #     for current in transition_prob[t]:
    #         candidate_prob = []
    #         for i in range(len(current)):
    #             value = Decimal(map_matching_prob[t][i]) * Decimal(current[i]) * Decimal(tmp_emission_prob[t + 1][i])
    #             candidate_prob.append(value)
    #         state_prob.append(max(candidate_prob))
    #         prev_index.append(candidate_prob.index(max(candidate_prob)))
    #     chosen_index.append(prev_index)
    #     map_matching_prob.append(state_prob)
    # hmm_path_index = []
    # final_prob = map_matching_prob[len(map_matching_prob) - 1]
    # final_index = final_prob.index(max(final_prob))
    # confidence = final_prob[final_index]
    # hmm_path_index.append(final_index)
    # for i in range(len(chosen_index), 0, -1):
    #     prev_index = chosen_index[i - 1][hmm_path_index[len(hmm_path_index) - 1]]
    #     hmm_path_index.append(prev_index)
    # hmm_path_index.reverse()
    # return [confidence, hmm_path_index]


def build_confidence_table(emission_prob, transition_prob):
    confidence_table = []
    for p in range(len(emission_prob)):
        sample_confidence = []
        for c in range(len(emission_prob[0])):
            if Decimal(emission_prob[p][c]) != Decimal(1e-300):
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
            confidence_change = []
            for c in range(len(brute_force_match[p])):
                if c != path_index[p] and brute_force_match[p][c][1] is not None:
                    index_diff = path_index_change(brute_force_match[p][c][1], path_index)
                    confidence_diff = abs(brute_force_match[p][c][0] - fixed_confidence)
                    # We assume that: smaller confidence change and larger index change results in higher priority
                    path_diff = Decimal(confidence_diff) / Decimal(index_diff)
                    path_change.append(path_diff)
            if len(path_change) > 0:
                # We choose the smallest from the vector
                model_change.append(min(path_change))
            else:
                print "The target point has no other choice!"
                # print brute_force_match[p]
                # print emission_prob[p]
                model_change.append(Decimal("inf"))
        else:
            print "The chosen road is too far away from the target point!"
            # print brute_force_match[p]
    return model_change


def omit_change_table(traj, true_path, true_path_whole, ini_path_whole, path_index):
    omit_index_change = []
    omit_jaccard_result = []
    omit_leven_result = []
    jaccard_change = []
    leven_change = []
    index_change = [0 for i in range(len(traj["trace"]["p"]))]
    for i in range(len(traj["trace"]["p"])):
        # Create a new trajectory omitting the target sample
        url_omit = server_prefix + "traj/remove_point/?id=" + traj["id"] + "&pid=" + traj["trace"]["p"][i]["id"]
        omit_traj_info = urllib2.urlopen(url_omit)
        omit_traj = json.load(omit_traj_info)
        # Perform map matching and save the emission and transition probability tables in cache
        url_omit_hmm = server_prefix + "map-matching/perform/?city=" + str(city) + "&id=" + omit_traj["id"]
        omit_map_matching = urllib2.urlopen(url_omit_hmm)
        omit_map_matching_result = json.load(omit_map_matching)
        omit_hmm_path = omit_map_matching_result["path"]
        omit_path_index = omit_map_matching_result["path_index"]
        omit_path_index = [int(index) for index in omit_path_index]
        # Insert the index of omitted point back, assuming the omitted point is not changed
        omit_path_index.insert(i, path_index[i])
        index_diff = path_index_change(omit_path_index, path_index)
        omit_index_change.append(index_diff)
        omit_path_whole = []
        for fragment in omit_hmm_path["road"]:
            omit_path_whole.append(fragment["road"]["id"])
        omit_jaccard_result.append(jaccard_distance(omit_path_whole, true_path_whole))
        omit_leven_result.append(levenshtein_distance(omit_path_whole, true_path_whole))
        jaccard_change.append(jaccard_distance(omit_path_whole, ini_path_whole))
        leven_change.append(levenshtein_distance(omit_path_whole, ini_path_whole))
        # Calculate the index change of each point
        for i in range(len(traj["trace"]["p"])):
            if omit_path_index[i] != path_index[i]:
                index_change[i] += 1
        # Delete the new trajectory omitting the target sample
        url_remove = server_prefix + "traj/remove/?id=" + omit_traj["id"]
        remove_traj = urllib2.urlopen(url_remove)
    index_change = [float(change) / float(len(index_change)) for change in index_change]
    return omit_index_change, omit_jaccard_result, omit_leven_result, jaccard_change, leven_change, index_change


if __name__ == "__main__":
    if len(sys.argv) != 8:
        # Reperform strategy includes: scan, modify
        # Selection strategy includes: forward, backward, binary, random, entropy_dist, entropy_confidence, max_change, min_change
        print "Input format: python trajectory_generator.py <server type> <protocol> <ip address> <port> <ground_truth_src> <city id> <selection strategy>"
    else:
        if sys.argv[1] == "celery":
            server_prefix = sys.argv[2] + "://" + sys.argv[3] + ":" + sys.argv[4] + "/"
        elif sys.argv[1] == "nginx":
            server_prefix = sys.argv[2] + "://" + sys.argv[3] + "/avatar/"
        ground_truth_src = sys.argv[5]
        city = int(sys.argv[6])
        selection_strategy = sys.argv[7]
        # Build ground truth index
        ground_truth_file = open(ground_truth_src, "r")
        # result_file = open(output_prefix + "_" + sys.argv[7] + "_" + sys.argv[8] + ".json", "a")
        # time_file = open(output_prefix + "_" + sys.argv[7] + "_" + sys.argv[8] + "_time.json", "a")
        # All the results for evaluation
        selection_time = []
        ini_acc_table = []
        num_selection_table = []
        selection_acc = []
        num_first_hit = []
        line_count = 1
        # output = open("test_omit_table.json", "a")
        for line in ground_truth_file.readlines():
            ground_truth = json.loads(line)
            true_path = {}
            true_path_whole = []
            for fragment in ground_truth["ground_truth"][0]:
                true_path_whole.append(fragment[0])
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
            path_index = [int(index) for index in path_index]
            #match_result = compare_result_with_truth(hmm_path, true_path)
            match_result = compare_result_in_path(hmm_path, true_path_whole)
            ini_acc_table.append(match_result[0])
            hmm_path_whole = []
            for fragment in hmm_path["road"]:
                hmm_path_whole.append(fragment["road"]["id"])
            leven_result = levenshtein_distance(hmm_path_whole, true_path_whole)
            if leven_result == 0:
                print "The " + str(line_count) + "th trajectory is right in path!"
            # print "The trajectory contains " + str(len(trace["p"])) + " samples. After initial map matching, " + str(match_result[0]) + " has been matched to the right road!"
            # Prepare question selection list
            selection_start = time.time()
            if selection_strategy == "random":
                shuffle_index = []
                for i in range(len(trace["p"])):
                    shuffle_index.append(i)
                random.shuffle(shuffle_index)
                if DEBUG:
                    print shuffle_index
            if selection_strategy == "entropy_dist":
                for i in range(len(emission_prob)):
                    emission_prob[i] = filter(lambda x : x != 1e-300, emission_prob[i])
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
            if selection_strategy == "min_change":
                model_change = model_change_table(emission_prob, transition_prob, path_index)
                # print model_change
                sorted_index = sorted(range(len(model_change)), key=lambda k: model_change[k])
                if DEBUG:
                    print sorted_index
            if selection_strategy in ["omit_index", "omit_jaccard", "omit_leven", "omit_change"]:
                omit_index_change, omit_jaccard_result, omit_leven_result, jaccard_change, leven_change, index_change = omit_change_table(traj_result, true_path, true_path_whole, hmm_path_whole, path_index)
                # output_str = json.dumps([match_result[1].keys(), omit_index_change, jaccard_change, leven_change, omit_jaccard_result, omit_leven_result])
                # output.write(output_str + "\n")
                if selection_strategy == "omit_index":
                    sorted_index = sorted(range(len(omit_index_change)), key=lambda k: omit_index_change[k], reverse=True)
                elif selection_strategy == "omit_jaccard":
                    sorted_index = sorted(range(len(omit_jaccard_result)), key=lambda k: omit_jaccard_result[k], reverse=True)
                elif selection_strategy == "omit_leven":
                    sorted_index = sorted(range(len(omit_leven_result)), key=lambda k: omit_leven_result[k], reverse=True)
                elif selection_strategy == "omit_change":
                    influence = [index_change[i] for i in range(len(index_change))]
                    sorted_index = sorted(range(len(influence)), key=lambda k: influence[k], reverse=True)
                    sorted_influence = sorted(influence, reverse=True)
                    print sorted_influence
                print sorted_index
            # My new mix model strategy
            if selection_strategy in ["3mix", "entropy_mix", "global_mix", "2mix", "ensemble"]:
                if selection_strategy != "2mix":
                    brute_force_match = build_confidence_table(emission_prob, transition_prob)
                    confidence_table = []
                    for sample in brute_force_match:
                        sample_result = []
                        for result in sample:
                            if result[1] is not None:
                                sample_result.append(result[0])
                        confidence_table.append(sample_result)
                    centropy_index = sort_by_entropy(confidence_table)
                if selection_strategy != "entropy_mix":
                    path_index = [int(index) for index in path_index]
                    model_change = model_change_table(emission_prob, transition_prob, path_index)
                    # For those points who have only one candidate road, put them at the bottom of the list
                    for i in range(len(model_change)):
                        if model_change[i][0] is None:
                            model_change[i] = [float("inf"), Decimal("inf")]
                    model_index = sorted(range(len(model_change)), key=lambda k: model_change[k])
                if selection_strategy != "global_mix":
                    for i in range(len(emission_prob)):
                        emission_prob[i] = filter(lambda x : x != 1e-300, emission_prob[i])
                    dentropy_index = sort_by_entropy(emission_prob)
                # combine the ranking lists
                score_list = [0 for i in range(len(trace["p"]))]
                for i in range(len(trace["p"])):
                    if selection_strategy != "2mix":
                        score_list[centropy_index[i]] += i
                    if selection_strategy != "entropy_mix":
                        score_list[model_index[i]] += i
                    if selection_strategy != "global_mix":
                        score_list[dentropy_index[i]] += i
                sorted_index = sorted(range(len(score_list)), key=lambda k: score_list[k])
            selection_end = time.time()
            selection_time.append(selection_end - selection_start)
            first_hit = False # When does the first selected point is actually matched wrongly
            num_selection = 0  # How many points are selected before all wrong points are found
            num_wrong = len(trace["p"]) - match_result[0] # Total number of wrong points
            num_remain = num_wrong
            while num_remain > 0:
                # Sequentially choose samples along the trajectory to ask user
                if selection_strategy == "forward":
                    p_index = num_selection
                elif selection_strategy in ["random", "binary"]:
                    p_index = shuffle_index[num_selection]
                elif selection_strategy in ["entropy_dist", "entropy_confidence", "min_change", "2mix", "3mix", "entropy_mix", "global_mix", "omit_index", "omit_jaccard", "omit_leven", "omit_change"]:
                    p_index = sorted_index[num_selection]
                num_selection += 1
                if p_index in match_result[1]:
                    num_remain -= 1
                    if not first_hit:
                        num_first_hit.append(num_selection)
                        first_hit = True
            if num_selection > 0:
                num_selection_table.append(num_selection)
                selection_acc.append(float(num_wrong) / float(num_selection))
            print "Finished " + str(line_count) + " trajectories..."
            line_count += 1
        # Calculate the statstic result
        # output.close()
        avg_time = float(sum(selection_time)) / float(len(selection_time))
        avg_ini_acc = float(sum(ini_acc_table)) / float(len(ini_acc_table))
        avg_num_selection = float(sum(num_selection_table)) / float(len(num_selection_table))
        avg_selection_acc = float(sum(selection_acc)) / float(len(selection_acc))
        avg_first_hit = float(sum(num_first_hit)) / float(len(num_first_hit))
        print selection_strategy + " selection method takes " + str(avg_time) + " seconds in average to generate an ordered sequence!"
        print str(avg_ini_acc) + " points in average are matched to the right road after initial map matching!"
        print str(avg_first_hit) + " points in average are selected before the first wrong point is found!"
        print str(avg_num_selection) + " points in average are selected before all the wrong points are found!"
        print "The accuracy of finding wrong points using " + selection_strategy + " strategy is " + str(avg_selection_acc) + " in average!"