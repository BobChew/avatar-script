import sys
import os

if __name__ == "__main__":
    # num_traj = [50]
    # num_sample = [10, 50]
    # edge = [1, 2, 3]
    # shake = [1, 2, 3]
    if len(sys.argv) != 8:
        # Reperform strategy includes: scan, modify
        # Selection strategy includes: forward, backward, binary, random, entropy_dist, entropy_confidence, max_change, min_change
        print "Input format: python preparation.py <protocol> <ip address> <port> <num_traj> <num_sample> <num_edge> <shake>"
    else:
        num_traj = sys.argv[4]
        num_sample = sys.argv[5]
        edge = sys.argv[6]
        shake = sys.argv[7]
        traj_src = "syn_data/ground_truth_" + num_traj + "traj_" + num_sample + "p_" + str(int(edge) * int(num_sample)) + "e_" + shake + "shake.json"
        generator_command = "python trajectory_generator.py " + sys.argv[1] + " " + sys.argv[2] + " " + sys.argv[3] + " " + traj_src + " shenzhen " + num_traj + " " + num_sample + " " + str(int(edge) * int(num_sample)) + " " + shake + " 0.0"
        os.system(generator_command)
        hmm_src = "syn_meta/hmm_result_" + num_traj + "traj_" + num_sample + "p_" + str(int(edge) * int(num_sample)) + "e_" + shake + "shake.csv"
        hmm_command = "python initial_map_matching.py " + sys.argv[1] + " " + sys.argv[2] + " " + sys.argv[3] + " " + traj_src + " shenzhen " + hmm_src
        os.system(hmm_command)
    # for t in num_traj:
        # for s in num_sample:
            # for e in edge:
                # for sh in shake:
                    # traj_src = "syn_data/ground_truth_" + str(t) + "traj_" + str(s) + "p_" + str(int(e * s)) + "e_" + str(sh) + "shake.json"
                    # generator_command = "python trajectory_generator.py http 127.0.0.1 9001 " + traj_src + " shenzhen " + str(t) + " " + str(s) + " " + str(int(e * s)) + " " + str(sh) + " 0.0"
                    # os.system(generator_command)
                    # hmm_src = "syn_meta/hmm_result_" + str(t) + "traj_" + str(s) + "p_" + str(int(e * s)) + "e_" + str(sh) + "shake.csv"
                    # hmm_command = "python initial_map_matching.py http 127.0.0.1 9001 " + traj_src + " shenzhen " + hmm_src
                    # os.system(hmm_command)
