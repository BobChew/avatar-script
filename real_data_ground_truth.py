import sys
import json
import urllib2
import random
import math
import time
from decimal import Decimal


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print "Input format: python trajectory_generator.py <server type> <protocol> <ip address> <port> <traj_src> <output_file>"
    else:
        if sys.argv[1] == "celery":
            server_prefix = sys.argv[2] + "://" + sys.argv[3] + ":" + sys.argv[4] + "/"
        elif sys.argv[1] == "nginx":
            server_prefix = sys.argv[2] + "://" + sys.argv[3] + "/avatar/"
        traj_src = sys.argv[5]
        traj_file = open(traj_src, "r")
        ground_truth_src = sys.argv[6]
        output = open(ground_truth_src, "a")
        # Build ground truth index
        # traj_file = open(traj_src, "r")
        for traj_id in traj_file.readlines():
            # Get the ground truth paths
            url_get_traj = server_prefix + "traj/get/?id=" + traj_id
            # print "Get trajectory url is: " + url_get_traj
            traj_info = urllib2.urlopen(url_get_traj)
            traj_data = json.load(traj_info)
            traj_set = {}
            traj_set["traj_id"] = [traj_data["id"]]
            ground_truth = []
            length = 0
            for road in traj_data["path"]["road"]:
                if road["p"] is None:
                    ground_truth.append([road["road"]["id"], []])
                else:
                    pid_set = []
                    for pid in road["p"].split(","):
                        pid_set.append(int(pid))
                    ground_truth.append([road["road"]["id"], pid_set])
                length += road["road"]["length"]
            traj_set["ground_truth"] = [ground_truth]
            traj_set["path_length"] = [length]
            ground_truth_str = json.dumps(traj_set)
            output.write(ground_truth_str + "\n")
        print "Finished!"