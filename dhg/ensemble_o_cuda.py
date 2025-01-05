import argparse
import pickle
import os
import torch
import numpy as np
from tqdm import tqdm
import math
from itertools import product
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-dir',
                        default='dhg/result\hg_dynamic\dhg/14',
                        help='')
    arg = parser.parse_args()
    with open(os.path.join(arg.main_dir, 'label.pkl'), 'rb') as l:
        label = list(pickle.load(l))
    with open(os.path.join(arg.main_dir, 'b', 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1))
    with open(os.path.join(arg.main_dir, 'j/', 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2))
    with open(os.path.join(arg.main_dir, 'jm/', 'epoch1_test_score.pkl'), 'rb') as r3:
        r3 = list(pickle.load(r3))
    # with open(os.path.join('work_dir/hg_dynamic_8/dhg/28/j', 'epoch1_test_score.pkl'), 'rb') as r3:
    #     r3 = list(pickle.load(r3))
    with open(os.path.join('dhg/result\hg_dynamic_4\dhg/14\jm', 'epoch1_test_score.pkl'), 'rb') as r4:
        r4 = list(pickle.load(r4))
    

    right_num = total_num = right_num_5 = 0
    n = 4
    norm = lambda x: x / np.linalg.norm(x)
    values = [0.,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    combinations = list(product(values, repeat=n))
    alpha = torch.tensor(combinations, dtype=torch.float32).reshape(11,11,11,11,n)
    acc = torch.zeros((11,11,11,11))
    right_num = torch.zeros_like(acc)
    alpha = alpha.to('cuda')
    acc = acc.to('cuda')
    right_num = right_num.to('cuda')
    total_num = 0
    for i in tqdm(range(len(label))):
        l = label[i]
        r11 = np.array(r1[i])
        r22 = np.array(r2[i])
        r33 = np.array(r3[i])
        r44 = np.array(r4[i])
        # r55 = np.array(r5[i])
        # r66 = np.array(r6[i])
        r11 = torch.from_numpy(r11)
        r22 = torch.from_numpy(r22)
        r33 = torch.from_numpy(r33)
        r44 = torch.from_numpy(r44)
        # r55 = torch.from_numpy(r55)
        # r66 = torch.from_numpy(r66)
        r = torch.cat((r11,r22,r33,r44)).reshape(n,-1)
        r = r.to('cuda')
        out = alpha@r
        out = torch.argmax(out,dim=-1)
        l = torch.ones_like(out)*l
        right_num += (out==l).int()
        total_num += 1
    acc = right_num / total_num
    print(f"acc={acc.max()}")
    index = acc.argmax()
    alpha = alpha.reshape(-1,n)
    ap = alpha[index]
    print(ap)
