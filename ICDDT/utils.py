import os
import math
import yaml
import torch
import random
import numpy as np
from collections import defaultdict

from datasets import BehaviorSetSequentialRecDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

main_args_list = [
    'dataset',
    'config_files',
    'output_dir'
]

train_args_list = [
    'lr',
    'batch_size',
    'max_epochs',
    'log_freq',
    'eval_freq',
    'seed',
    'tensorboard_on',
    'run_dir',
    'loss_type',
    'behavior_types'
]

model_args_list = [
    'init_std',
    'hidden_size',
    'max_seq_length',
    'no',
    'dropout',

    ## model params
    'n_heads',
    'n_layers',
    'inner_size',
    'hidden_act',
    'layer_norm_eps',
    'initializer_range',
    'diffusion_steps',
    'noise_schedule',
    
    'gamma',
    'aux_alpha',
    'tar_alpha',
    'aux_beta',
    'tar_beta'
]

optimizer_args_list = [
    'weigth_decay',
    'adam_beta1',
    'adam_beta2'
]

scheduler_args_list = [
    'decay_factor',
    'min_lr',
    'patience'
]

def clear_dict(d):
    if d is None:
        return None
    elif isinstance(d, list):
        return list(filter(lambda x: x is not None, map(clear_dict, d)))
    elif not isinstance(d, dict):
        return d
    else:
        r = dict(
                filter(lambda x: x[1] is not None,
                    map(lambda x: (x[0], clear_dict(x[1])),
                        d.items())))
        if not bool(r):
            return None
        return r

def setup_global_seed(SEED):
    print(f'Global SEED is setup as {SEED}.')

    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

def load_config_files(file_path, args):
    file_path = file_path + f'{args.dataset}.yaml'
    # file_path = '/home/fyue/diffusion/myfw/config/UB.yaml'
    with open(file_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    yaml_config.update(clear_dict(vars(args)))
    return yaml_config

def show_args_info(args):
    pad = 26
    category_list = ['Main', 'Train', 'Model', 'Optimizer', 'Scheduler']
    whole_args_list = dict()
    whole_args_list['Main'] = main_args_list
    whole_args_list['Train'] = train_args_list
    whole_args_list['Model'] = model_args_list
    whole_args_list['Optimizer'] = optimizer_args_list
    whole_args_list['Scheduler'] = scheduler_args_list

    args_info = set_color("*" * pad + f" Configure Info: " + f"*" * (pad+1) + '\n', 'red')
    for category in category_list:
        args_info += set_color(category + ' Hyper Parameters:\n', 'pink')
        args_info += '\n'.join([(set_color("{:<32}", 'cyan') + ' : ' +
                                 set_color("{:>35}", 'yellow')).format(arg_name, arg_value)
                                for arg_name, arg_value in vars(args).items()
                                if arg_name in whole_args_list[category]])
        args_info += '\n'

    print(args_info)

def check_output_path(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"{output_dir} has been automatically created..." )

def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'


def data_partition(dataset_name):
    userset = set()
    itemset = set()
    behaset = set() # {1, 2, 3, 4} for UB

    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_extra_for_test = defaultdict(list)
    user_test = defaultdict(list)
    
    base_path = '../Dataset/' + dataset_name + os.sep

    if dataset_name == 'UB' or dataset_name == 'JD':
        train_data_path = base_path + 'train.txt'
        valid_data_path = base_path + 'valid.txt'
        test_data_path = base_path + 'test.txt'
        extra_for_test_data_path = base_path + 'extra_for_test.txt'

        path_list = [train_data_path, valid_data_path, extra_for_test_data_path, test_data_path]
        dict_list = [user_train, user_valid, user_extra_for_test, user_test]

        for path, dict in zip(path_list, dict_list):
            f = open(path, 'r')
            for line in f:
                u = line.rstrip().split(',')[0]
                i = line.rstrip().split(',')[1]
                u = int(u)
                i = int(i)

                beha = line.rstrip().split(',')[2]
                beha = int(beha)
                
                userset.add(u)
                itemset.add(i)
                behaset.add(beha)
                dict[u].append([i, beha])
                
            f.close()


    print_dataset_msg = '\n'
    print_dataset_msg += set_color('Basic Statistics of ' +
                                   dataset_name + ' Dataset:\n', 'white')
    print_dataset_msg += set_color('# User Num: ' + str(len(userset)) + '\n', 'white')
    print_dataset_msg += set_color('# Max UserId: ' + str(max(userset)) + '\n', 'white')
    print_dataset_msg += set_color('# Item Num: ' + str(len(itemset)) + '\n', 'white')
    print_dataset_msg += set_color('# Max ItemId: ' + str(max(itemset)) + '\n', 'white')
    print_dataset_msg += set_color('# Behavior Type Num: ' + str(max(behaset)) + '\n', 'white')

    total_seq_length = 0.0
    for u in user_train:
        total_seq_length += len(user_train[u])

    print_dataset_msg += set_color(f'# Avg Seq Length: {total_seq_length / len(user_train):.2f}' + '\n', 'white')
    print(print_dataset_msg)

    return (user_train, user_valid, user_extra_for_test, user_test, userset, itemset, behaset)

def prepare_data_aug(user_train, config):
    '''user_train: <dict>'''
    max_seq_length = config['max_seq_length']

    whole_uid_list, whole_inter_list = [], []
    for i, u in enumerate(user_train):
        trunc_inter_list = user_train[u][-max_seq_length:]
        whole_uid_list += [u] * len(trunc_inter_list)
        whole_inter_list += trunc_inter_list

    last_uid = None
    seq_start = 0
    uid_list, inter_slice_list, target_item, target_beha, inter_list_length = [], [], [], [], []
    for i, uid in enumerate(whole_uid_list):
        if last_uid != uid:
            last_uid = uid
            seq_start = i
        else:
            if i - seq_start > max_seq_length:
                seq_start += 1
            uid_list.append(uid)
            inter_slice_list.append(slice(seq_start, i))
            target_item.append(whole_inter_list[i][0])
            target_beha.append(whole_inter_list[i][1])
            inter_list_length.append(i - seq_start)
            
    train_uid_list = uid_list
    train_inter_list = whole_inter_list
    inter_slice_list = inter_slice_list
    target_item = target_item
    target_beha = target_beha
    inter_list_length = inter_list_length

    augs = {
        'uid_list': train_uid_list, # [u1, u1, u1, ...]
        'inter_list': train_inter_list, #[u1seq, u2seq, ...]
        'inter_slice_list': inter_slice_list, # [u1seqidx1, u1seqidx2, u1seqidx3, ..]
        'target_item': target_item, # [u1tar1, u1tar2, u1tar3, ...]
        'target_beha': target_beha, # [u1tar1, u1tar2, u1tar3, ...]
        'inter_list_length': inter_list_length # [u1len1, u1len2, u1len3, ...]
    }

    return augs

def prepare_eval_data(train, target, config):
    '''
    For validation: train, valid
    For test: train + extra_for_test, test
    '''
    max_seq_length = config['max_seq_length']

    uid_list, inter_list, inter_slice_list, \
    target_item, target_beha, inter_list_length, ground_truth = [], [], [], [], [], [], []
    seq_start = 0
    for _, u in enumerate(train):
        if len(target[u]) == 0:
            continue

        trunc_inter_list = train[u][-max_seq_length:]
        inter_list += trunc_inter_list
        l = len(trunc_inter_list)

        uid_list.append(u)
        inter_slice_list.append(slice(seq_start, seq_start + l))
        target_item.append(target[u][0][0])
        target_beha.append(target[u][0][1])
        inter_list_length.append(l)
        ground_truth.append([target[u][0][0]])

        seq_start += l
        
    eva = {
        'uid_list': uid_list,
        'inter_list': inter_list,
        'inter_slice_list': inter_slice_list,
        'target_item': target_item,
        'target_beha': target_beha,
        'inter_list_length': inter_list_length,
        'ground_truth': ground_truth
    }
    return eva

def _merge_dict(*dict):
    merge = defaultdict(list)
    for u in dict[0].keys():
        for i in range(len(dict)):
            merge[u] += dict[i][u]

    return merge

def data_preparation(config_dict, data):
    '''user_train, user_valid, user_extra_for_test, user_test,
        userset, itemset, behaset'''
    config_dict['user_num'] = max(data[-3])
    config_dict['item_num'] = max(data[-2])
    config_dict['beha_num'] = max(data[-1])
    # config_dict['behavior_types'] = data[-1]
    train, valid, extra_fot_test, test = data[0], data[1], data[2], data[3]
    
    # augs
    augs = prepare_data_aug(train, config_dict)  # user_train, <class Dict>
    _merge = _merge_dict(train, extra_fot_test)
    val = prepare_eval_data(train, valid, config_dict)
    test = prepare_eval_data(_merge, test, config_dict)

    train_dataset = BehaviorSetSequentialRecDataset(config_dict, data, augs=augs)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config_dict['batch_size'])

    valid_dataset = BehaviorSetSequentialRecDataset(config_dict, data, augs=val, data_type="valid")
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=config_dict['batch_size'])

    test_dataset = BehaviorSetSequentialRecDataset(config_dict, data, augs=test, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=config_dict['batch_size'])

    return train_dataloader, valid_dataloader, test_dataloader, config_dict

def ndcg_k(pred, ground_truth, topk):
    ndcg = 0.0
    count = float(len(ground_truth))
    for uid in range(len(ground_truth)):
        if ground_truth[uid] == [0]:
            count -= 1.0
            continue
        k = min(topk, len(ground_truth[uid]))
        idcg = idcg_k(k)
        dcg_k = sum([int(pred[uid][j] in set(ground_truth[uid]))
                     / math.log(j+2, 2) for j in range(topk)])
        ndcg += dcg_k / idcg
    return ndcg / count

def idcg_k(k):
    '''
    Calculates the Ideal Discounted Cumulative Gain at k
    '''
    idcg = sum([1.0 / math.log(i+2, 2) for i in range(k)])
    if not idcg:
        return 1.0
    else:
        return idcg

def hr_k(pred, ground_truth, topk):
    hr = 0.0
    count = float(len(ground_truth))
    for uid in range(len(ground_truth)):
        if ground_truth[uid] == [0]:
            count -= 1.0
            continue
        pred_set = set(pred[uid][:topk])
        ground_truth_set = set(ground_truth[uid])
        hr += len(pred_set & ground_truth_set) / \
                  float(len(ground_truth_set))
    return hr / count


class EarlyStopping:
    '''
    Early stops the training if the test metrics doesn't improve after a given patience
    '''
    def __init__(self, patience=30, verbose=True, delta=1e-5, save_path="./output", logger=None):
        '''
        Args:
        :param patience: How many times have to wait after last time Rec. metrics improved
                         Default: 30
        :param verbose: If True, prints a message for each improvement
                         Default: True
        :param delta: Minimum change for the performance metrics to qualify as improvement
                      Default: 1e-5
        :param save_path: the folder path to save the best performance model
        '''
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_scores = None
        self.best_valid_epoch = 0
        self.hr_10_max = 0
        self.early_stop = False
        self.delta = delta
        self.save_path = save_path
        self.logger = logger

    def __call__(self, scores, epoch, model):
        '''
        scores: [HR_5, NDCG_5, HR_10, NDCG_10, HR_20, NDCG_20]
        Here we focus on HR_10
        '''
        hr_10 = scores[2]
        if self.best_scores is None:
            self.best_scores = scores
            self.best_valid_epoch = epoch
            self.save_checkpoint(scores, model)
        elif (hr_10 > self.hr_10_max + self.delta):
            self.save_checkpoint(scores, model)
            self.best_scores = scores
            self.best_valid_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.logger:
                self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


    def save_checkpoint(self, scores, model):
        '''
        Saves model when both NDCG and HR increase
        '''
        hr_10 = scores[2]
        if self.verbose:
            print(f'HR@10 metrics has increased from ({self.hr_10_max:.4f} --> {hr_10:.4f}). Saving model ...')
            if self.logger:
                self.logger.info(f'HR@10 metrics has increased from ({self.hr_10_max:.4f} --> {hr_10:.4f}). Saving model ...')

        torch.save(model.state_dict(), self.save_path)
        self.hr_10_max = hr_10

