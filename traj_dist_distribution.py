import sys
import json
import urllib2

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print "Input format: python traj_dist_distribution.py <protocol> <ip address> <port> <city> <output file>"
    else:
	server_prefix = sys.argv[1] + "://" + sys.argv[2] + ":" + sys.argv[3] + "/avatar/"
        city = sys.argv[4]
        if sys.argv[4] == "shenzhen":
	    if sys.argv[2] == "10.89.5.185":
        	city = 2
	    else:
		city = 1
        else:
            print "No map for this city!"
            city = -1
	# Get all trajectory ids from database
	url_all_traj = server_prefix + "traj/get_all/"
	all_traj = urllib2.urlopen(url_all_traj)
	all_traj_result = json.load(all_traj)
	traj_set = all_traj_result["ids"]
	print len(traj_set)
	# Save the traj id, no need if already exists
#	traj_set_src = "all_traj_id_" + sys.argv[4] + ".csv"
#	traj_file = open(traj_set_src, "w")
#	for traj_id in traj_set:
#	    traj_file.write(traj_id + "\n")
#	traj_file.close()
	dist_file = open(sys.argv[5], "a")
	# Perform HMM map matching
	error_num = 0
	for traj_id in traj_set:
	    url_hmm = server_prefix + "map-matching/perform/?city=" + str(city) + "&id=" + traj_id
	    print "Map matching url is: " + url_hmm
	    try:
	        map_matching_info = urllib2.urlopen(url_hmm)
	        map_matching_result = json.load(map_matching_info)
	        hmm_dist = map_matching_result["dist"]
		dist_str = json.dumps(hmm_dist)
#       	dist_src = "all_dist_" + sys.argv[4] + ".json"
        	dist_file.write(dist_str + "\n")
	    except urllib2.HTTPError, e:
		print e.code
		error_num += 1
	dist_file.close()
	print str(error_num) + "trajectories have problems during map matching and are discarded!"
	print "Finished saving distance file!"
