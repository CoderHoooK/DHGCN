import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm
import math

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
                        default='work_dir/exp_3/shrec/14',
                        help='')
    arg = parser.parse_args()
    with open(os.path.join(arg.main_dir, 'label.pkl'), 'rb') as l:
        label = list(pickle.load(l))
    with open(os.path.join(arg.main_dir, 'j/', 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1))
    with open(os.path.join(arg.main_dir, 'jm/', 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2))
    with open(os.path.join(arg.main_dir, 'jr/', 'epoch1_test_score.pkl'), 'rb') as r3:
        r3 = list(pickle.load(r3))



    right_num = total_num = right_num_5 = 0

    norm = lambda x: x / np.linalg.norm(x)

    for i in tqdm(range(len(label))):
        l = label[i]
        r11 = np.array(r1[i])
        r22 = np.array(r2[i])
        r33 = np.array(r3[i])
        # r44 = np.array(r4[i])
        # r55 = np.array(r5[i])
        # r66 = np.array(r6[i])
        r = 0. *r11 + 0.3*r22 + 1.*r33
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
