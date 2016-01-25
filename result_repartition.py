import sys
import json


if __name__ == "__main__":
    if len(sys.argv) != 10:
        # Reperform strategy includes: scan, modify
        # Selection strategy includes: forward, backward, binary, random, entropy_dist, entropy_confidence, max_change, min_change
        print "Input format: python active_hmm_result_analysis.py <traj file prefix> <result prefix> <new file prefix> <num_traj> <num of point> <edge> <shake> <reperform strategy> <selection strategy>"
    else:
        traj_repo = sys.argv[1]
        result_repo = sys.argv[2]
        new_repo = sys.argv[3]
        num_traj = sys.argv[4]
        num_sample = sys.argv[5]
        edge = sys.argv[6]
        shake = sys.argv[7]
        strategy = sys.argv[8]
        order = sys.argv[9]
        combined_traj = []
        combined_result = []
        combined_acc = []
        combined_time = []
        # Read all the trajectory and result records and combine together
        for acc_level in ["high", "mid", "low"]:
            traj_info = num_traj + "traj_" + num_sample + "p_" + str(int(edge) * int(num_sample)) + "e_" + shake + "shake_" + acc_level + "_acc"
            traj_file = open(traj_repo + "ground_truth_" + traj_info + ".json", "r")
            result_file = open(result_repo + "active_result_" + traj_info + "_" + strategy + "_" + order + ".json", "r")
            acc_file = open(result_repo + "active_result_" + traj_info + "_" + strategy + "_" + order + "_time.json", "r")
            time_file = open(result_repo + "active_result_" + traj_info + "_" + strategy + "_" + order + "_acc.json", "r")
            for line in traj_file.readlines():
                combined_traj.append(line)
            for line in result_file.readlines():
                combined_result.append(line)
            for line in time_file.readlines():
                combined_time.append(line)
            acc_table = json.loads(acc_file.readline())
            for acc in acc_table:
                combined_acc.append(acc)
            traj_file.close()
            result_file.close()
            acc_file.close()
        if len(combined_traj) != len(combined_result) or len(combined_traj) != len(combined_acc) or len(combined_traj) != len(combined_time):
            print "Number of trajectories doesn't match number of result records!"
            exit()
        # Re-partition trajectory and result records into proper files
        new_traj_high = open(new_repo + "ground_truth_" + traj_info + "high_acc.json", "w")
        new_traj_mid = open(new_repo + "ground_truth_" + traj_info + "mid_acc.json", "w")
        new_traj_low = open(new_repo + "ground_truth_" + traj_info + "low_acc.json", "w")
        new_result_high = open(new_repo + "active_result_" + traj_info + "high_acc_" + strategy + "_" + order + ".json", "a")
        new_result_mid = open(new_repo + "active_result_" + traj_info + "mid_acc_" + strategy + "_" + order + ".json", "a")
        new_result_low = open(new_repo + "active_result_" + traj_info + "low_acc_" + strategy + "_" + order + ".json", "a")
        new_time_high = open(new_repo + "active_result_" + traj_info + "high_acc_" + strategy + "_" + order + "_time.json", "a")
        new_time_mid = open(new_repo + "active_result_" + traj_info + "mid_acc_" + strategy + "_" + order + "_time.json", "a")
        new_time_low = open(new_repo + "active_result_" + traj_info + "low_acc_" + strategy + "_" + order + "_time.json", "a")
        acc_high = []
        acc_mid = []
        acc_low = []
        for i in range(len(combined_traj)):
            acc = float(combined_acc[i][0]) / float(combined_acc[i][len(combined_acc[i]) - 1])
            if acc < 0.6:
                new_traj_low.write(combined_traj[i] + "\n")
                new_result_low.write(combined_result[i] + "\n")
                new_time_low.write(combined_time[i] + "\n")
                acc_low.append(combined_acc[i])
            elif acc < 0.8:
                new_traj_mid.write(combined_traj[i] + "\n")
                new_result_mid.write(combined_result[i] + "\n")
                new_time_mid.write(combined_time[i] + "\n")
                acc_mid.append(combined_acc[i])
            elif acc < 1.0:
                new_traj_high.write(combined_traj[i] + "\n")
                new_result_high.write(combined_result[i] + "\n")
                new_time_high.write(combined_time[i] + "\n")
                acc_high.append(combined_acc[i])
        new_acc_high = open(new_repo + "active_result_" + traj_info + "high_acc_" + strategy + "_" + order + "_acc.json", "a")
        new_acc_str = json.dumps(acc_high)
        new_acc_high.write(new_acc_str)
        new_acc_mid = open(new_repo + "active_result_" + traj_info + "high_acc_" + strategy + "_" + order + "_acc.json", "a")
        new_acc_str = json.dumps(acc_high)
        new_acc_mid.write(new_acc_str)
        new_acc_low = open(new_repo + "active_result_" + traj_info + "high_acc_" + strategy + "_" + order + "_acc.json", "a")
        new_acc_str = json.dumps(acc_high)
        new_acc_low.write(new_acc_str)
        new_traj_high.close()
        new_traj_mid.close()
        new_traj_low.close()
        new_result_high.close()
        new_result_mid.close()
        new_result_low.close()
        new_time_high.close()
        new_time_mid.close()
        new_time_low.close()
        new_acc_high.close()
        new_acc_mid.close()
        new_acc_low.close()