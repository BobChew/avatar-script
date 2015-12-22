import os


if __name__ == "__main__":
#    output_src = "output.csv"
#    traj_generator_command = "python trajectory_generator.py ground_truth.json shenzhen 1 50 100 0.0003 0.0006 0.0"
#    os.system(traj_generator_command)
    reperform_strategy = ["scan"]
    selection_strategy = ["forward", "backward", "random", "binary", "entropy_dist", "entropy_confidence", "max_change", "min_change"]
    traj_src = "real_data/ground_truth.json"
    for active_type in reperform_strategy:
	for order in selection_strategy:
	    active_src = "real_result/active_result.json"
	    active_hmm_command = "python active_map_matching.py http nicpu3.cse.ust.hk 9001 " + traj_src + " shenzhen 1 " + active_type + " " + order + " " + active_src
	    os.system(active_hmm_command)
    print traj_src + " " + active_type + " " + order + " is finished!"
