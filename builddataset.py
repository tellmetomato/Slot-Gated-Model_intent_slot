import numpy as np
from collections import Counter
import os
import pickle

def build_dataset(data_path,all_data_path):
    t2i, s2i, in2i = dict(),dict(),dict()
    i_t,i_s,i_in = 0,0,0
    with open(all_data_path) as f:
        m = f.read().split('\n')
        for i in m:
            if i == '\n':
                continue
            intent_word = i.split(' ')[-1]
            if not in2i.__contains__(intent_word):
                in2i[intent_word] = i_in
                i_in += 1
            for j in i.split(' ')[0:-2]:
                token_word = j.split(';')[0]
                slot_word = j.split(';')[-1]
                if not t2i.__contains__(token_word):
                    t2i[token_word] = i_t
                    i_t += 1
                if not s2i.__contains__(slot_word):
                    s2i[slot_word] = i_s
                    i_s += 1

    query = []
    slots = []
    intent_1 = []
    with open(data_path) as f:
        m = f.read().split('\n')
        for i in m:
            if i == '\n':
                continue
            intent_word = i.split(' ')[-1]
            intent_1.append(in2i[intent_word])
            seq_token = []
            seq_slot = []
                
            for j in i.split(' ')[0:-2]:
                token_word = j.split(';')[0]
                slot_word = j.split(';')[-1]
                seq_token.append(t2i[token_word])

                seq_slot.append(s2i[slot_word])

            if seq_token:

                query.append(np.array(seq_token,float))
                slots.append(np.array(seq_slot,float))

    
    query = np.array(query)
    slots = np.array(slots)
    intent_1 = np.array(intent_1)


    return query,slots,intent_1,i_t,i_s,i_in
    




