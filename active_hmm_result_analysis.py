import json
import sys




if __name__ == "__main__":
    result_stat = []
    result_table = []
    result_file = open("test_omit_table.json", "r")
    for line in result_file.readlines():
        result_table.append(json.loads(line))
    for result in result_table:
        wrong_index = result[0]
        index_change = result[1]
        jaccard_change = result[2]
        leven_change = result[3]
        jaccard_result = result[4]
        leven_result = result[5]
        print sorted(range(len(jaccard_change)), key=lambda k: jaccard_change[k], reverse=True)
        # Without the wrong point, other point will change match
        wrong_index_change = 0
        # Without the right point, other point will change match
        right_index_change = 0
        # Without the wrong point, the jaccard dist of the path will change
        wrong_jacc_change = 0
        # Without the right point, the jaccard dist of the path will change
        right_jacc_change = 0
        # Without the wrong point, the jaccard path will be right
        wrong_jacc_right = 0
        # Without the right point, the jaccard path will be wrong
        right_jacc_wrong = 0
        # Without the wrong point, the leven dist of the path will change
        wrong_leven_change = 0
        # Without the right point, the leven dist of the path will change
        right_leven_change = 0
        # Without the wrong point, the leven path will be right
        wrong_leven_right = 0
        # Without the right point, the leven path will be wrong
        right_leven_wrong = 0
        for i in range(len(index_change)):
            if i in wrong_index:
                if index_change[i] != 0:
                    wrong_index_change += 1
                if jaccard_change[i] != 0.0:
                    wrong_jacc_change += 1
                if leven_change[i] != 0:
                    wrong_leven_change += 1
                if jaccard_result[i] == 0.0:
                    wrong_jacc_right += 1
                if leven_result[i] == 0:
                    wrong_leven_right += 1
            else:
                if index_change[i] != 0:
                    right_index_change += 1
                if jaccard_change[i] != 0.0:
                    right_jacc_change += 1
                if leven_change[i] != 0:
                    right_leven_change += 1
                if jaccard_result[i] != 0.0:
                    right_jacc_wrong += 1
                if leven_result[i] != 0:
                    right_leven_wrong += 1
        result_stat.append([wrong_index_change, wrong_jacc_change, wrong_leven_change, right_index_change, right_jacc_change, right_leven_change, wrong_jacc_right, wrong_leven_right, right_jacc_wrong, right_leven_wrong])
    num_wrong_index_change = 0
    num_wrong_path_change = 0
    num_right_index_stay = 0
    num_right_path_stay = 0
    sum_right_index_change = 0.0
    sum_right_path_change = 0.0
    num_wrong_to_right = 0
    sum_right_to_wrong = 0.0
    for stat in result_stat:
        if stat[0] != 0:
            num_wrong_index_change += 1
        if stat[1] != 0:
            num_wrong_path_change += 1
        if stat[3] == 0:
            num_right_index_stay += 1
        if stat[4] == 0:
            num_right_path_stay += 1
        sum_right_index_change += float(stat[3]) / 30.0
        sum_right_path_change += float(stat[4]) / 30.0
        if stat[6] == 0:
            num_wrong_to_right += 1
        sum_right_to_wrong += float(stat[8]) / 30.0
    avg_right_index_change = sum_right_index_change / 30.0
    avg_right_path_change = sum_right_path_change / 30.0
    avg_right_to_wrong = sum_right_to_wrong / 30.0
    # print num_wrong_index_change
    # print num_wrong_path_change
    # print num_right_index_stay
    # print num_right_path_stay
    # print avg_right_index_change
    # print avg_right_path_change
    # print num_wrong_to_right
    # print avg_right_to_wrong

