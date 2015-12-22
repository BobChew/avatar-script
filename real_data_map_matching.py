import sys
import json
import urllib2
import random
import math
import time
from decimal import Decimal


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print "Input format: python trajectory_generator.py <protocol> <ip address> <port> <traj_src> <city>"
    else:
	server_prefix = sys.argv[1] + "://" + sys.argv[2] + ":" + sys.argv[3] + "/avatar/"
        traj_src = sys.argv[4]
        if sys.argv[5] in ["shenzhen", "Shenzhen"]:
            city = 1
        else:
            print "No map for this city!"
            exit()
	# Build ground truth index
	traj_file = open(traj_src, "r")
	for traj_id in traj_file.readlines():
	    # Perform HMM map matching
	    task_start = time.time()
	    url_hmm = server_prefix + "map-matching/perform/?city=" + str(city) + "&id=" + traj_id
            print "Map matching url is: " + url_hmm
            map_matching_info = urllib2.urlopen(url_hmm)
            task_end = time.time()
            print "Map matching task takes " + str(task_end - task_start) + " seconds!"
	    # print "The trajectory contains " + str(len(trace["p"])) + " samples. After initial map matching, " + str(match_result[0]) + " has been matched to the right road!";
	print "Finished!"
