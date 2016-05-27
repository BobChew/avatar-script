import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    acc_table = []
    result_prefix = "/home/bobchew/Workspace/AvatarProject/result/active_result_30p_30e_1shake_50acc_"
    for strategy in ["forward_fix,step", "random_step", "entropy_dist_step", "entropy_confidence_step", "min_change_step", "omit_entropy_step", "entropy_confidence_step,fix", "min_change_step,fix", "omit_entropy_step,fix"]:
        result_file = open(result_prefix + strategy + "_acc.json", "r")
        result = json.loads(result_file.readline())
        acc_vector = [[] for num_p in range(len(result[0]))]
        for num_traj in range(len(result)):
            for num_p in range(len(result[0])):
                acc_vector[num_p].append(result[num_traj][num_p])
        for i in range(len(acc_vector)):
            acc_vector[i].sort()
            avg_acc = float(sum(acc_vector[i])) / float(len(acc_vector[i]))
            med_acc = acc_vector[i][int(len(acc_vector[i]) * 0.5)]
            quad25_acc = acc_vector[i][int(len(acc_vector[i]) * 0.25)]
            quad75_acc = acc_vector[i][int(len(acc_vector[i]) * 0.75)]
            acc_vector[i] = quad75_acc
        acc_vector = [float(num_right) / float(acc_vector[len(acc_vector) - 1]) for num_right in acc_vector]
        acc_table.append(acc_vector)
        # print acc_vector
    num_scan = [x for x in range(len(acc_table[0]))]
    plt.xlabel("Number of scanned points")
    plt.ylabel("Accuracy of active map matching result")
    plt.plot(num_scan, acc_table[0], 'k', label="sequential")
    plt.plot(num_scan, acc_table[1], 'b', label="random")
    plt.plot(num_scan, acc_table[2], 'c', label="global_dentropy")
    plt.plot(num_scan, acc_table[3], 'g', label="global_centropy")
    plt.plot(num_scan, acc_table[4], 'm', label="global_change")
    plt.plot(num_scan, acc_table[5], 'r', label="global_sentropy")
    plt.plot(num_scan, acc_table[6], 'g--', label="global_centropy")
    plt.plot(num_scan, acc_table[7], 'm--', label="global_change")
    plt.plot(num_scan, acc_table[8], 'r--', label="global_sentropy")
    plt.legend(loc="lower right")
    plt.show()