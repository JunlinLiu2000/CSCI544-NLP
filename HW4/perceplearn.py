import sys
import re
import time

def read_data(input_data_path):
    True_Fake_list = []
    Pos_Neg_list = []
    dict_list = []

    word_set = set()
    f = open(input_data_path)
    lines = f.readlines()
    for line in lines:
        dic1 = {}
        line = re.sub(r'[^\w\s]', '', line)
        line = line.split()
        if line[1] == 'Fake':
            True_Fake_list.append(-1)
        else:
            True_Fake_list.append(1)

        if line[2] == 'Pos':
            Pos_Neg_list.append(1)
        else:
            Pos_Neg_list.append(-1)
        for word in line[3:]:
            word = word.lower()
            word_set.add(word)
            if word in dic1:
                dic1[word] = dic1[word] + 1
            else:
                dic1[word] = 1

        dict_list.append(dic1)
    return True_Fake_list, Pos_Neg_list, dict_list, word_set


def PerceptronTrain(word_set, dict_list,True_Fake_list, Pos_Neg_list):
    TF_weight_dic = {}
    PN_weight_dic = {}
    for word in word_set:
        TF_weight_dic[word] = 0
        PN_weight_dic[word] = 0

    b_TF = 0
    b_PN = 0
    for iter in range(10):
        for i in range(len(dict_list)):
            a_TF = 0
            a_PN = 0
            dict = dict_list[i]
            for key, value in dict.items():
                a_TF += TF_weight_dic[key]*value
                a_PN += PN_weight_dic[key]*value
            a_TF = a_TF + b_TF
            a_PN = a_PN + b_PN
            if a_TF * True_Fake_list[i] <= 0:
                for key, value in dict.items():
                    TF_weight_dic[key] = TF_weight_dic[key] + True_Fake_list[i]*value
                b_TF = b_TF + True_Fake_list[i]
            
            if a_PN * Pos_Neg_list[i] <= 0:
                for key, value in dict.items():
                    PN_weight_dic[key] = PN_weight_dic[key] + Pos_Neg_list[i]*value
                b_PN = b_PN + Pos_Neg_list[i]
    
    return TF_weight_dic, b_TF, PN_weight_dic, b_PN

def AveragedPerceptronTrain(word_set, dict_list,True_Fake_list, Pos_Neg_list):
    TF_weight_dic = {}
    PN_weight_dic = {}
    TF_cached_dic = {}
    PN_cached_dic = {}
    for word in word_set:
        TF_weight_dic[word] = 0
        PN_weight_dic[word] = 0
        TF_cached_dic[word] = 0
        PN_cached_dic[word] = 0
    b_TF = 0
    b_PN = 0
    beta_TF = 0
    beta_PN = 0
    c = 1
    for iter in range(10):
        for i in range(len(dict_list)):
            a_TF = 0
            a_PN = 0
            dict = dict_list[i]
            for key, value in dict.items():
                a_TF += TF_weight_dic[key]*value
                a_PN += PN_weight_dic[key]*value
            a_TF = a_TF + b_TF
            a_PN = a_PN + b_PN
            if a_TF * True_Fake_list[i] <= 0:
                for key, value in dict.items():
                    TF_weight_dic[key] = TF_weight_dic[key] + True_Fake_list[i] * value
                    TF_cached_dic[key] = TF_cached_dic[key] + True_Fake_list[i] * value * c
                b_TF = b_TF + True_Fake_list[i]
                beta_TF = beta_TF + True_Fake_list[i] * c
            
            if a_PN * Pos_Neg_list[i] <= 0:
                for key, value in dict.items():
                    PN_weight_dic[key] = PN_weight_dic[key] + Pos_Neg_list[i] * value
                    PN_cached_dic[key] = PN_cached_dic[key] + Pos_Neg_list[i] * value * c
                b_PN = b_PN + Pos_Neg_list[i]
                beta_PN = beta_PN + Pos_Neg_list[i] * c

            c = c + 1
    
    for key, value in TF_weight_dic.items():
        TF_weight_dic[key] = value - (1/c) * TF_cached_dic[key]
        PN_weight_dic[key] = PN_weight_dic[key] - (1/c) * PN_cached_dic[key]
    b_TF = b_TF - (1/c) * beta_TF
    b_PN = b_PN - (1/c) * beta_PN

    return TF_weight_dic, b_TF, PN_weight_dic, b_PN

def write_vanillamodel(TF_weight_dic, b_TF, PN_weight_dic, b_PN):
    f = open('vanillamodel.txt', 'w')
    f.write(str(TF_weight_dic)+"\n")
    f.write(str(b_TF)+"\n")
    f.write(str(PN_weight_dic)+"\n")
    f.write(str(b_PN)+"\n")

def write_averagedmodel(avg_TF_weight_dic, avg_b_TF, avg_PN_weight_dic, avg_b_PN):
    f = open('averagedmodel.txt', 'w')
    f.write(str(avg_TF_weight_dic)+"\n")
    f.write(str(avg_b_TF)+"\n")
    f.write(str(avg_PN_weight_dic)+"\n")
    f.write(str(avg_b_PN)+"\n")

if __name__ == '__main__':
    start_time = time.time()
    input_data_path = sys.argv[1]
    True_Fake_list, Pos_Neg_list, dict_list, word_set = read_data(input_data_path)
    
    TF_weight_dic, b_TF, PN_weight_dic, b_PN = PerceptronTrain(word_set, dict_list,True_Fake_list, Pos_Neg_list)

    avg_TF_weight_dic, avg_b_TF, avg_PN_weight_dic, avg_b_PN = AveragedPerceptronTrain(word_set, dict_list,True_Fake_list, Pos_Neg_list)

    write_vanillamodel(TF_weight_dic, b_TF, PN_weight_dic, b_PN)
    write_averagedmodel(avg_TF_weight_dic, avg_b_TF, avg_PN_weight_dic, avg_b_PN)

    end_time = time.time()
    time = end_time - start_time
    print(f"Duration:", time)