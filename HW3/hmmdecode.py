import sys
import json
import time
import ast

def read_data(input_file):
    hmmlearn_data = {}
    with open(input_file) as f:
        line = f.readline()
        item = line.split('=', 1)
        hmmlearn_data[item[0]] = item[1].strip()
    all_data_dic = ast.literal_eval(hmmlearn_data.get("all_data_dic"))
    start_prob_dict = all_data_dic["start_prob_dict"]
    transition_prob_dict = all_data_dic["transition_prob_dict"]
    emission_prob_dict = all_data_dic["emission_prob_dict"]
    word_list = list(all_data_dic["word_list"])
    tag_list = list(all_data_dic["tag_list"])
    return start_prob_dict,transition_prob_dict,emission_prob_dict,word_list,tag_list

def viterbi(tag_list, word_list,start_prob_dict,emission_dict,transition_dict, line):
    words = line.strip().split()
    max_prob = {}
    best_path = {}
    max_prob[0] = {}
    best_path[0] = {}
    for tag in tag_list:
        start_word = words[0]
        if start_word not in word_list:
            max_prob[0][tag] = start_prob_dict[tag]
            best_path[0][tag] = 'start'                
        else:
            if start_word in emission_dict[tag]:
                if 0 == emission_dict[tag][start_word]:
                    max_prob[0][tag] = 0
                    best_path[0][tag] = 'start'
                else:
                    max_prob[0][tag] = start_prob_dict[tag]*emission_dict[tag][start_word]
                    best_path[0][tag] = 'start'

    for index in range(1, len(words)):
        max_prob[index] = {}
        best_path[index] = {}
        if index != len(words)-1:
            for tag in tag_list:
                word = words[index]
                if word not in word_list:
                    max_p, argmax_pre = argmax1(index, tag, max_prob, transition_dict)
                    if max_p != 0:
                        max_prob[index][tag], best_path[index][tag] = max_p, argmax_pre
                else:
                    max_p, argmax_pre = argmax2(index, tag, max_prob,transition_dict,emission_dict, word) 
                    if max_p != 0:
                        max_prob[index][tag], best_path[index][tag] = max_p, argmax_pre
        else:
            if index == len(words)-1:
                max_p, argmax_pre = argmax3(index, tag, max_prob,transition_dict, tag_list)
                if max_p != 0:
                    max_prob[index][tag], best_path[index][tag] = max_p, argmax_pre
                
    end = tag
    predict_line = []
    for i in range(len(words)-1,-1,-1):
        predict_line.append((words[i],end))
        end = best_path[i][end]
    return predict_line[::-1]

def argmax1(index, tag, max_prob, transition_dict):
    max_p = -1
    argmax_pre = ""
    for item in max_prob[index-1]:
        prob = max_prob[index-1][item]*transition_dict[item][tag]
        if prob > max_p:
            max_p = prob
            argmax_pre = item
    return max_p, argmax_pre

def argmax2(index, tag, max_prob,transition_dict,emission_dict, word):
    max_p = -1
    argmax_pre = ""
    for item in max_prob[index-1]:
        if word in emission_dict[tag] and 0 != emission_dict[tag][word]:
            prob = max_prob[index-1][item]*transition_dict[item][tag]*emission_dict[tag][word]
        else:
            prob = 0
        if prob > max_p:
            max_p = prob
            argmax_pre = item
    return max_p, argmax_pre

def argmax3(index, tag, max_prob,transition_dict, tag_list):
    max_p = -1
    argmax_pre = ""
    for item in max_prob[index-1]:
        prob = max_prob[index-1][item] * transition_prob_dict[item]['end']
        
        if prob > max_p:
            max_p = prob
            argmax_pre = item
    return max_p, argmax_pre

def read_file_data(tag_list, word_list,start_prob_dict,emission_dict,transition_dict,input_file_path):
    with open(input_file_path) as f:
        all_data = f.readlines()[:1]
    list1 = []
    for line in all_data:
        predict_line = viterbi(tag_list, word_list,start_prob_dict,emission_dict,transition_dict,line)
        list1.append(predict_line)
    return list1

def write_file():
    with open('hmmoutput.txt', 'w') as f:
        for line in list1:
            str2=""
            for index in range(len(line)):
                if index != len(line)-1:
                    str1 = str(line[index][0])+'/'+str(line[index][1]+' ')
                else:
                    str1 = str(line[index][0])+'/'+str(line[index][1])
                str2 += str1
            f.write(str2+'\n')    

if __name__ == '__main__':
    start_time = time.time()
    input_file = 'hmmmodel.txt'
    input_file_path = sys.argv[1]
    start_prob_dict,transition_prob_dict,emission_prob_dict,word_list,tag_list = read_data(input_file)
    print(start_prob_dict)
    list1 = read_file_data(tag_list, word_list,start_prob_dict,emission_prob_dict,transition_prob_dict,input_file_path)

    write_file()
    end_time = time.time()
    time = end_time - start_time
    print(f"Duration:", time)