import sys
import json
import time

def read_data(input_file_path):
    tag_list = set()
    word_list = set()
    start_dict = {}
    transition_dict = {}
    emission_dict = {}
    with open(input_file_path) as f:
        for line in f.readlines():
            line = line.strip().split()
            # start_word = line[0].rsplit('/',1)[0]
            start_tag = line[0].rsplit('/',1)[1]
            if start_tag not in start_dict:
                start_dict[start_tag] = 1
            else:
                start_dict[start_tag] = start_dict.get(start_tag) + 1
                
            for index in range(len(line)):
                word = line[index].rsplit('/',1)[0]
                tag = line[index].rsplit('/',1)[1]
                tag_list.add(tag)
                word_list.add(word)
                if index != len(line)-1:
                    # next_word = line[index+1].rsplit('/',1)[0]
                    next_tag = line[index+1].rsplit('/',1)[1]
                    if tag not in transition_dict:
                        transition_dict[tag] = {}
                        if next_tag not in transition_dict[tag]:
                            transition_dict[tag][next_tag] = 1
                    else:
                        if next_tag not in transition_dict[tag]:
                            transition_dict[tag][next_tag] = 1
                        else:
                            transition_dict[tag][next_tag] = transition_dict[tag][next_tag] + 1
                    
                    if tag not in emission_dict:
                        emission_dict[tag] = {}
                        if word not in emission_dict[tag]:
                            emission_dict[tag][word] = 1
                    else:
                        if word not in emission_dict[tag]:
                            emission_dict[tag][word] = 1
                        else:
                            emission_dict[tag][word] = emission_dict[tag][word] + 1
                if index == len(line)-1:
                    word = 'end'
                    if tag not in transition_dict:
                        transition_dict[tag] = {}
                        if word not in transition_dict[tag]:
                            transition_dict[tag][word] = 1
                        else:
                            transition_dict[tag][word] = transition_dict[tag][word] + 1
                    else:
                        if word not in transition_dict[tag]:
                            transition_dict[tag][word] = 1
                        else:
                            transition_dict[tag][word] = transition_dict[tag][word] + 1
            
    return start_dict, transition_dict, emission_dict, tag_list, word_list

def get_start_prob(start_dict):
    all_tag_list = list(transition_dict.keys())
    all_tag_list.append('end')
    all_tag_list.append('start')
    start_sum = 0
    start_prob_dict = {}
    for tag in all_tag_list:
        if tag not in start_dict:
            start_dict[tag] = 1
        else:
            start_dict[tag] = start_dict[tag] + 1
        start_sum += start_dict[tag]
    for key, value in start_dict.items():
        start_prob_dict[key] = value/start_sum
    return start_prob_dict

def get_transition_prob(transition_dict):
    all_tag_list = list(transition_dict.keys())
    all_tag_list.append('end')
    all_tag_list.append('start')
    transition_prob_dict = transition_dict
    for key, value in transition_dict.items():
        total_sum = 0
        for tag in all_tag_list:
            if tag not in value:
                value[tag] = 1
            else:
                value[tag] = value[tag] + 1
            total_sum += value[tag]
        for item in value:
            transition_prob_dict[key][item] = value[item]/total_sum
    return transition_prob_dict

# def get_emission_prob(emission_dict):
#     emission_prob_dict = emission_dict
#     for key, value in emission_dict.items():
#         total_sum = 0
#         for word in value:
#             total_sum += value[word]
#         for word in value:
#             emission_prob_dict[key][word] = value[word]/total_sum
#     return emission_prob_dict

def get_emission_prob(emission_dict):
    emission_prob_dict = emission_dict
    for key, value in emission_dict.items():
        total_sum = 0
        for word in value:
            total_sum += value[word]
        for word in value:
            emission_prob_dict[key][word] = value[word]/total_sum
        # for word in word_list:
        #     if word not in value:
        #         emission_prob_dict[key][word] = 0
    return emission_prob_dict

# def write_file():
#     all_data_dict = {}
#     all_data_dict['start_prob_dict'] = start_prob_dict
#     all_data_dict['transition_prob_dict'] = transition_prob_dict
#     all_data_dict['emission_prob_dict'] = emission_prob_dict
#     with open('hmmlearn.txt', 'w') as f:
#      f.write(json.dumps(all_data_dict))

def write_file(all_data_dict):
    with open('hmmmodel.txt', 'w') as f:
        f.write("all_data_dic=" + str(all_data_dict) + "\n")


if __name__ == '__main__':
    start_time = time.time()
    input_file_path = sys.argv[1]
    start_dict, transition_dict, emission_dict, tag_list, word_list = read_data(input_file_path)
    start_prob_dict = get_start_prob(start_dict)
    transition_prob_dict = get_transition_prob(transition_dict)
    emission_prob_dict = get_emission_prob(emission_dict)
    all_data_dict = {'start_prob_dict':start_prob_dict, 'transition_prob_dict':transition_prob_dict, 'emission_prob_dict':emission_prob_dict, 'word_list':word_list, 'tag_list':tag_list}
    write_file(all_data_dict)
    end_time = time.time()
    time = end_time - start_time
    print(f"Duration:", time)