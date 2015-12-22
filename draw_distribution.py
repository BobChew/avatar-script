import json
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

#def f(x, delta):
#    para = 1.0 / (math.sqrt(2 * math.pi) * delta)
#    exponent = -0.5 * (x / delta) * (x / delta)
#    return para * math.exp(exponent)


if __name__ == "__main__":
    all_dist = []
    all_dist_partition = {}
    dist_src = "traj_dist_distribution.json"
    dist_file = open(dist_src)
    for line in dist_file:
	dist_vector = json.loads(line)
	all_dist += dist_vector
    bias = 0.0
    avg_dist = float(sum(all_dist)) / float(len(all_dist))
    for dist in all_dist:
	bias += (dist - avg_dist) * (dist - avg_dist)
    bias /= float(len(all_dist))
    print "Normal distribution parameters: mu="  + str(avg_dist) + ", delta=" + str(math.sqrt(bias))
    bias_zero = 0.0
    for dist in all_dist:
        bias_zero += dist * dist
    bias_zero /= float(len(all_dist))
    print "Zero mean normal distribution parameters: delta=" + str(math.sqrt(bias_zero))

#    for dist in all_dist:
#	if int(dist) + 1 not in all_dist_partition:
#	    all_dist_partition[int(dist) + 1] = 1
#	else:
#	    all_dist_partition[int(dist/10) + 1] += 1
#    dist_distribution = all_dist_partition.items()
#    dist_distribution.sort()
#    total_num = sum(all_dist_partition.values())
#    dist_list = []
#    prob_list = []
#    for tuple in dist_distribution:
#	dist_list.append(tuple[0])
#	prob_list.append(float(tuple[1])/float(total_num))
#    result = curve_fit(f, dist_list, prob_list)
#    print result
#    plt.plot(dist_list[0:20], prob_list[0:20])
#    plt.show()
