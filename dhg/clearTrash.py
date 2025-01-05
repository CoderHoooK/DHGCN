import argparse
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir-path',
        default='work_dir/exp_1/dhg/28/jr',
        help='dir_path of trash files')
    
    args = parser.parse_args()
    path = args.dir_path
    assert os.path.exists(path), "路径不存在"

    log_path = os.path.join(path,'log.txt')
    assert os.path.exists(log_path), "log.txt路径错误 或 不存在"

    with open(log_path,'r') as f :
        content = f.readlines()
        best_epoch = content[-8][-3:-1]
        f.close()

    best_pt = 'runs-'+best_epoch
    remainFiles = ['log.txt',best_pt,'epoch1_test_score.pkl','.py','config.yaml','label.pkl']

    files = os.listdir(path)
    for i in files:
        flag = True
        for j in remainFiles:
            if j in i:
                flag = False
                break
        if flag:
            p = os.path.join(path,i)
            if os.path.isdir(p):
                continue
            os.remove(p)
            print("removed {}".format(i))


