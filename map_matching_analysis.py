import csv


if __name__ == "__main__":
    # ini_acc = []
    # scan_percent = []
    # accuracy = []
    # result_file = file("../result/active_result_forward_decay.csv", "r")
    # reader = csv.reader(result_file)
    # for line in reader:
    #     if float(line[0]) < 1.0:
    #         ini_acc.append(float(line[0]))
    #         scan_percent.append(float(line[1]))
    #         accuracy.append(float(line[2]))
    # avg_ini_acc = sum(ini_acc) / len(ini_acc)
    # avg_scan = sum(scan_percent) / len(scan_percent)
    # avg_acc = sum(accuracy) / len(accuracy)
    # print avg_ini_acc
    # print avg_scan
    # print avg_acc
    ini_acc = []
    scan_forward = []
    scan_centropy = []
    scan_change = []
    acc_forward = []
    acc_centropy = []
    acc_change = []
    forward_file = file("../result/active_result_forward.csv", "r")
    forward_reader = csv.reader(forward_file)
    for line in forward_reader:
        ini_acc.append(float(line[0]))
        scan_forward.append(float(line[1]))
        acc_forward.append(float(line[2]))
    centropy_file = file("../result/active_result_entropy_confidence.csv", "r")
    centropy_reader = csv.reader(centropy_file)
    for line in centropy_reader:
        scan_centropy.append(float(line[1]))
        acc_centropy.append(float(line[2]))
    change_file = file("../result/active_result_min_change.csv", "r")
    change_reader = csv.reader(change_file)
    for line in change_reader:
        scan_change.append(float(line[1]))
        acc_change.append(float(line[2]))
    num_traj = 0
    sum_ini_acc = 0.0
    sum_scan_forward = 0.0
    sum_scan_centropy = 0.0
    sum_scan_change = 0.0
    sum_acc_forward = 0.0
    sum_acc_centropy = 0.0
    sum_acc_change = 0.0
    for i in range(len(ini_acc)):
        if scan_centropy[i] < scan_forward[i] and scan_change[i] < scan_forward[i]:
            num_traj += 1
            sum_ini_acc += ini_acc[i]
            sum_scan_forward += scan_forward[i]
            sum_scan_centropy += scan_centropy[i]
            sum_scan_change += scan_change[i]
            sum_acc_forward += acc_forward[i]
            sum_acc_centropy += acc_centropy[i]
            sum_acc_change += acc_change[i]
    print num_traj
    print sum_ini_acc / num_traj
    print sum_scan_forward / num_traj
    print sum_scan_centropy / num_traj
    print sum_scan_change / num_traj
    print sum_acc_forward / num_traj
    print sum_acc_centropy / num_traj
    print sum_acc_change / num_traj