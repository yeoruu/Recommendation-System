import torch
import random
import numpy as np
from torch.utils.data import Dataset

def neg_sample(seq_item, itemset):
    seq_item_set = set(seq_item)
    neg_item_idx = random.randint(0, len(itemset)-1)
    neg_item = itemset[neg_item_idx]
    while neg_item in seq_item_set:
        neg_item_idx = random.randint(0, len(itemset)-1)
        neg_item = itemset[neg_item_idx]
    return neg_item

class BehaviorSetSequentialRecDataset(Dataset):
    def __init__(self, config, data, augs=None, data_type="train"):
        self.config = config
        userset, itemset, behaset = data[-3], data[-2], data[-1]

        self.data_type = data_type
        self.augs = augs
        self.uid_list = augs['uid_list']
        self.inter_list = augs['inter_list']
        self.inter_slice_list = augs['inter_slice_list']
        self.target_item = augs['target_item']
        self.target_beha = augs['target_beha']
        self.inter_list_length = augs['inter_list_length']
        if self.data_type != 'train':
            self.ground_truth = augs['ground_truth']

        self.itemset = list(itemset)
        self.userset = list(userset)
        self.behaset = list(behaset)
        self.max_seq_length = config['max_seq_length']

    def _pack_up_data_to_tensor(self, index):
        user_id = self.uid_list[index]
        input_item = [inter[0] for inter in self.inter_list[self.inter_slice_list[index]]]
        input_beha = [inter[1] for inter in self.inter_list[self.inter_slice_list[index]]]

        target_item_last = self.target_item[index]
        target_beha_last = self.target_beha[index]

        pad_len = self.max_seq_length - len(input_item)
        input_item = [0] * pad_len + list(input_item)
        input_beha = [0] * pad_len + list(input_beha)
        
        target_item = input_item[1:] + [target_item_last]
        target_beha = input_beha[1:] + [target_beha_last]
        input_item = input_item[-self.max_seq_length:]
        input_beha = input_beha[-self.max_seq_length:]

        # warning msg part
        assert len(input_item) == self.max_seq_length
        assert len(input_beha) == self.max_seq_length
        assert len(target_beha) == self.max_seq_length

        if self.data_type == 'train':
            target_neg = neg_sample(input_item + [target_item_last], self.itemset)
            cur_tensors = {
                'user_id': torch.tensor(user_id, dtype=torch.long),
                'input_item': torch.tensor(input_item, dtype=torch.long),
                'input_beha': torch.tensor(input_beha, dtype=torch.long),
                'target_item': torch.tensor(target_item, dtype=torch.long),
                'target_beha': torch.tensor(target_beha, dtype=torch.long),
                'target_neg': torch.tensor(target_neg, dtype=torch.long)
            }
        else:
            ground_truth = self.ground_truth[index]
            cur_tensors = {
                'user_id': torch.tensor(user_id, dtype=torch.long),
                'input_item': torch.tensor(input_item, dtype=torch.long),
                'input_beha': torch.tensor(input_beha, dtype=torch.long),
                'target_item': torch.tensor(target_item, dtype=torch.long),
                'target_beha': torch.tensor(target_beha, dtype=torch.long),
                'ground_truth': torch.tensor(ground_truth, dtype=torch.long)
            }

        return cur_tensors

    def __getitem__(self, index):
        assert self.data_type in {"train", "valid", "test"}
        one_idx_tensors = self._pack_up_data_to_tensor(index)
        return one_idx_tensors

    def __len__(self):
        return len(self.uid_list)