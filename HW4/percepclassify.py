import sys
import re
import time
def read_data(input_data_path):
    answer_list = []
    f = open(input_data_path)
    lines = f.readlines()
    for line in lines:
        TF_weight_sum = 0
        PN_weight_sum = 0
        line = re.sub(r'[^\w\s]', '', line)
        line = line.split()
        for word in line:
            word = word.lower()
            if word not in TF_weight_dic:
                continue
            else:
                TF_weight_sum = TF_weight_sum + TF_weight_dic[word] 
                PN_weight_sum = PN_weight_sum + PN_weight_dic[word]
        T_or_F = ''
        P_or_N = ''
        if TF_weight_sum + b_TF > 0:
            T_or_F = 'True'
        else:
            T_or_F = 'Fake'
        if PN_weight_sum + b_PN > 0:
            P_or_N = 'Pos'
        else:
            P_or_N = 'Neg'
        answer_list.append([line[0], T_or_F, P_or_N])
    return answer_list

def read_model_data(input_model_path):
    f = open(input_model_path)
    lines = f.readlines()
    TF_weight_dic = eval(lines[0])
    b_TF = eval(lines[1])
    PN_weight_dic = eval(lines[2])
    b_PN = eval(lines[3])
    return TF_weight_dic, b_TF, PN_weight_dic, b_PN

def write_file(answer_list):
    f = open('percepoutput.txt', 'w')
    for i in answer_list:
        f.write(' '.join(i))
        f.write('\n')


if __name__ == '__main__':
    start_time = time.time()
    input_model_path = sys.argv[1]
    input_data_path = sys.argv[2]
    read_model_data(input_model_path)
    TF_weight_dic, b_TF, PN_weight_dic, b_PN = read_model_data(input_model_path)

    answer_list = read_data(input_data_path)
    write_file(answer_list)
    end_time = time.time()
    time = end_time - start_time
    print(f"Duration:", time)
