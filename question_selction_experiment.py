import sys
import os


if __name__ == "__main__":
#    output_src = "output.csv"
#    traj_generator_command = "python trajectory_generator.py ground_truth.json shenzhen 1 50 100 0.0003 0.0006 0.0"
#    os.system(traj_generator_command)
    if len(sys.argv) != 8:
        # Reperform strategy includes: scan, modify
        # Selection strategy includes: forward, backward, binary, random, entropy_dist, entropy_confidence, max_change, min_change
        print "Input format: python question_selection_experiment.py <protocol> <ip address> <port> <num_traj> <num_sample> <num_edge> <shake>"
    else:
        # num_traj = [50]
        # num_sample = [10, 50]
        # edge = [1, 2, 3]
        # shake = [1, 2, 3]
        num_traj = sys.argv[4]
        num_sample = sys.argv[5]
        edge = sys.argv[6]
        shake = sys.argv[7]
        reperform_strategy = ["scan"]
        selection_strategy = ["forward", "random", "binary", "entropy_dist", "entropy_confidence", "max_change", "min_change"]
        traj_src = "syn_data/ground_truth_" + num_traj + "traj_" + num_sample + "p_" + str(int(edge) * int(num_sample)) + "e_" + shake + "shake.json"
        for active_type in reperform_strategy:
            for order in selection_strategy:
                active_src = "syn_result/active_result_" + num_traj + "traj_" + num_sample + "p_" + str(int(edge) * int(num_sample)) + "e_" + shake + "shake.json"
                active_hmm_command = "python active_map_matching.py " + sys.argv[1] + " " + sys.argv[2] + " " + sys.argv[3] + " " + traj_src + " shenzhen 1 " + active_type + " " + order + " " + active_src
                os.system(active_hmm_command)
                print traj_src + " " + active_type + " " + order + " is finished!"
    # for t in num_traj:
        # for s in num_sample:
            # for e in edge:
                # for sh in shake:
                    # traj_src = "syn_data/ground_truth_" + str(t) + "traj_" + str(s) + "p_" + str(int(e * s)) + "e_" + str(sh) + "shake.json"
    		    # for active_type in reperform_strategy:
			# for order in selection_strategy:
			    # active_src = "syn_result/active_result_" + str(t) + "traj_" + str(s) + "p_" + str(int(e * s)) + "e_" + str(sh) + "shake_" + active_type + "_" + order + ".json"
	    		    # active_hmm_command = "python active_map_matching.py http 0.0.0.0 9001 " + traj_src + " shenzhen 1 " + active_type + " " + order + " " + active_src
	    		    # os.system(active_hmm_command)
			    # print traj_src + " " + active_type + " " + order + " is finished!"
