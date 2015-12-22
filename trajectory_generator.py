import urllib2
import json
import sys

if __name__ == "__main__":
    if len(sys.argv) != 11:
        print "Input format: python trajectory_generator.py <protocol> <ip address> <port> <ground truth file> <city> <num of traj> <num of sample> <num of edges> <shake> <missing rate>"
    else:
        # server_prefix = sys.argv[1] + "://" + sys.argv[2] + ":" + sys.argv[3] + "/avatar/"
        server_prefix = sys.argv[1] + "://" + sys.argv[2] + "/avatar/"
        ground_truth_src = sys.argv[4]
        if sys.argv[5] in ["shenzhen", "Shenzhen"]:
            city = 3
        else:
            print "No map for this city!"
            exit()
        num_traj = sys.argv[6]
        num_sample = sys.argv[7]
        num_edge = sys.argv[8]
        shake = sys.argv[9]
        missing_rate = sys.argv[10]
        output = open(ground_truth_src, "a")
        num_complete_traj = 0
        while num_complete_traj < int(num_traj):
            url = server_prefix + "simulator/generate_syn_traj/?city=" + str(city) + "&traj=1&point=" + num_sample + "&edge=" + num_edge + "&miss=" + missing_rate
            # print "Trajectory generator url is: " + url
            try:
                traj_info = urllib2.urlopen(url)
                traj_set = json.load(traj_info)
                ground_truth_str = json.dumps(traj_set)
                output.write(ground_truth_str + "\n")
                num_complete_traj += 1
            # print "Created " + str(num_complete_traj) + " trajectories..."
            except urllib2.HTTPError, e:
                pass
                # print e.code
        output.close()
        print "Finished saving ground truth file!"
