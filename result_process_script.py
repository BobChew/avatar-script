import sys
import os

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print "Input format: python result_process_script.py <traj file prefix> <result prefix> <new file prefix> <analysis prefix>"
    else:
        traj_prefix = sys.argv[1]
        result_prefix = sys.argv[2]
        new_prefix = sys.argv[3]
        analysis_prefix = sys.argv[4]
        # Result repartition
        rep_prefix = "python result_repartition.py " + traj_prefix + " " + result_prefix + " " + new_prefix
        for strategy in ["forward", "random", "binary", "entropy_dist", "entropy_confidence", "max_change", "min_change"]:
            os.system(rep_prefix + " 100 10 2 1 scan " + strategy)
            os.system(rep_prefix + " 100 30 1 1 scan " + strategy)
            os.system(rep_prefix + " 100 30 2 1 scan " + strategy)
            os.system(rep_prefix + " 100 30 2 2 scan " + strategy)
            os.system(rep_prefix + " 100 30 2 3 scan " + strategy)
            os.system(rep_prefix + " 100 30 3 1 scan " + strategy)
            os.system(rep_prefix + " 100 50 2 1 scan " + strategy)
        # Result analysis
        ana_prefix = "python active_hmm_result_analysis.py " + new_prefix
        for acc in ["high", "mid", "low"]:
            os.system(ana_prefix + " 100 10 2 1 " + acc + " " + analysis_prefix)
            os.system(ana_prefix + " 100 30 1 1 " + acc + " " + analysis_prefix)
            os.system(ana_prefix + " 100 30 2 1 " + acc + " " + analysis_prefix)
            os.system(ana_prefix + " 100 30 2 2 " + acc + " " + analysis_prefix)
            os.system(ana_prefix + " 100 30 2 3 " + acc + " " + analysis_prefix)
            os.system(ana_prefix + " 100 30 3 1 " + acc + " " + analysis_prefix)
            os.system(ana_prefix + " 100 50 2 1 " + acc + " " + analysis_prefix)