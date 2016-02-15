import json
import sys

DEBUG = False


def check_ini_acc(acc_table, acc_level):
    check = True
    for acc_vector in acc_table:
        acc = float(acc_vector[0]) / float(acc_vector[len(acc_vector) - 1])
        if acc_level == "high":
            if acc < 0.8 or acc >= 1.0:
                check = False
                print acc_vector
                print acc
        if acc_level == "mid":
            if acc < 0.6 or acc >= 0.8:
                check = False
                print acc_vector
                print acc
        if acc_level == "low":
            if acc >= 0.6:
                check = False
                print acc_vector
                print acc
    return check


def result_statistic_analysis(result_table):
    num_scan = []
    num_modify = []
    for result in result_table:
        num_scan.append(result[2])
        num_modify.append(result[4])
    avg_scan = float(sum(num_scan)) / float(len(num_scan))
    avg_modify = float(sum(num_modify)) / float(len(num_modify))
    return avg_scan, avg_modify


def acc_statistic_analysis(acc_table):
    avg_acc = [0.0 for i in range(len(acc_table[0]))]
    total_length = acc_table[0][len(acc_table[0]) - 1]
    if DEBUG:
        print total_length
    for acc_vector in acc_table:
        for i in range(len(acc_vector)):
            acc = float(acc_vector[i]) / float(total_length)
            avg_acc[i] += acc
    avg_acc = [sum_acc / float(len(acc_table)) for sum_acc in avg_acc]
    return avg_acc


if __name__ == "__main__":
    if len(sys.argv) != 8:
        # Reperform strategy includes: scan, modify
        # Selection strategy includes: forward, backward, binary, random, entropy_dist, entropy_confidence, max_change, min_change
        print "Input format: python active_hmm_result_analysis.py <result_repo> <num_traj> <num of point> <edge> <shake> <accuracy level> <output repo>"
    else:
        result_repo = sys.argv[1]
        num_traj = sys.argv[2]
        num_sample = sys.argv[3]
        edge = sys.argv[4]
        shake = sys.argv[5]
        acc_level = sys.argv[6]
        output_repo = sys.argv[7]
        traj_info = num_traj + "traj_" + num_sample + "p_" + str(int(edge) * int(num_sample)) + "e_" + shake + "shake_" + acc_level + "_acc"
        output = open(output_repo + "statistic_result_" + traj_info + ".json", "a")
        for order in ["forward", "binary", "random", "entropy_dist", "entropy_confidence", "max_change", "min_change"]:
            result_file = open(result_repo + "active_result_" + traj_info + "_scan_" + order + ".json", "r")
            acc_file = open(result_repo + "active_result_" + traj_info + "_scan_" + order + "_acc.json", "r")
            result_table = []
            for line in result_file.readlines():
                result_table.append(json.loads(line))
            acc_table = json.loads(acc_file.readline())
            avg_scan, avg_modify = result_statistic_analysis(result_table)
            required = check_ini_acc(acc_table, acc_level)
            avg_acc = acc_statistic_analysis(acc_table)
            if not required:
                print result_repo + "active_result_" + traj_info + "_scan_" + order + ".json"
            output.write(json.dumps([order, avg_scan, avg_modify, avg_acc]) + "\n")
        output.close()
