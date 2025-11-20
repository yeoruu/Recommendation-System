import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.sparse import csr_matrix
from torch.utils.tensorboard import SummaryWriter
from utils import *

class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, config, logger):
        self.user_num = config['user_num']
        self.item_num = config['item_num']
        self.beha_num = config['beha_num']
        self.hidden_size = config['hidden_size']
        self.loss_fct = config['loss_type']
        self.logger = logger 
        if self.loss_fct == 'CE':
            self.loss_ce = nn.CrossEntropyLoss()

        self.config = config
        self.cuda_condition = config['cuda_condition']
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()
        self.output_path = model.save_path

        self.max_epochs = config['max_epochs']
        self.log_freq = config['log_freq']
        self.eval_freq = config['eval_freq']
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        betas = (config['adam_beta1'], config['adam_beta2'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'],
                                          betas=betas, weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                                    factor=config['decay_factor'],
                                                                    verbose=True,
                                                                    min_lr=config['min_lr'],
                                                                    patience=config['patience'])

        self.tensorboard_on = config['tensorboard_on']
        self.writer = None
        if self.tensorboard_on:
            self.writer = SummaryWriter(os.path.join(config['run_dir'],
                                                     f"{self.model.model_name}_{self.model.no}"))
            self._create_model_training_folder(self.writer)
        self.print_out_epoch = 0

    def _create_model_training_folder(self, writer):
        model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder)

    def train(self):
        self.fit(self.train_dataloader)

    def valid(self):
        return self.fit(self.valid_dataloader, mode="eval")

    def test(self, record=True):
        return self.fit(self.test_dataloader, mode="test", record=record)

    def fit(self, dataloader, mode="train", record=True):
        raise NotImplementedError

    def load(self):
        self.model.load_state_dict(torch.load(self.output_path))

    def save(self):
        torch.save(self.model.cpu().state_dict(), self.output_path)
        self.model.to(self.device)

    def CELoss(self, seq_output, pos_ids, test_item_emb):
        ''' pos_ids: [B] '''
        # test_item_emb = self.model.item_embedding.weight # [I, D]
        D = test_item_emb.size(-1)
        output = seq_output # [B, D]
        logits = torch.matmul(output, test_item_emb.transpose(0, 1)) # [B, I]
        loss = self.loss_ce(logits, pos_ids)
        
        return loss 
    

    def calculate_all_item_prob(self, output):
        # (|I|+1, d)
        item_emb_weight = self.model.item_embeddings.weight
        # item side prob
        # (b, d) * (d, |I|+1) --> (b, |I|+1)
        item_prob = torch.matmul(output, item_emb_weight.transpose(0, 1))
        return item_prob

    def calculate_eval_metrics(self, mode, print_out_epoch, pred_item_list, ground_truth_list, record=True):
        NDCG_n_list, HR_n_list = [], []
        for k in [5, 10, 20]:
            NDCG_n_list.append(ndcg_k(pred_item_list, ground_truth_list, k))
            HR_n_list.append(hr_k(pred_item_list, ground_truth_list, k))

        eval_metrics_info = {
            "Epoch": print_out_epoch,
            "HR@5": "{:.4f}".format(HR_n_list[0]),
            "NDCG@5": "{:.4f}".format(NDCG_n_list[0]),
            "HR@10": "{:.4f}".format(HR_n_list[1]),
            "NDCG@10": "{:.4f}".format(NDCG_n_list[1]),
            "HR@20": "{:.4f}".format(HR_n_list[2]),
            "NDCG@20": "{:.4f}".format(NDCG_n_list[2]),
        }

        if self.writer is not None and record:
            if mode == 'eval':
                self.writer.add_scalars('NDCG@10', {'Valid': NDCG_n_list[1]}, print_out_epoch)
                self.writer.add_scalars('HR@10', {'Valid': HR_n_list[1]}, print_out_epoch)
            else:
                self.writer.add_scalars('NDCG@10', {'Test': NDCG_n_list[1]}, print_out_epoch)
                self.writer.add_scalars('HR@10', {'Test': HR_n_list[1]}, print_out_epoch)

        if mode == 'eval':
            print(set_color(str(eval_metrics_info), "cyan"))
            self.logger.info(str(eval_metrics_info))
        return [HR_n_list[0], NDCG_n_list[0], HR_n_list[1],
                NDCG_n_list[1], HR_n_list[2], NDCG_n_list[2]], str(eval_metrics_info)


class MyTrainer(Trainer):
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, config, logger):
        super(MyTrainer, self).__init__(
            model, train_dataloader, valid_dataloader, test_dataloader, config, logger
        )

    def fit(self, dataloader, mode="train", record=True):
        assert mode in {"train", "eval", "test"}
        print(set_color("Rec Model mode: " + mode, "green"))

        if mode == "train":
            self.model.train()

            early_stopping = EarlyStopping(save_path=self.output_path, logger=self.logger)
            print(set_color(f"Rec dataset Num of batch: {len(dataloader)}", "white"))

            for epoch in range(self.max_epochs):

                ce_total_loss = 0.0
                seq_ce_total_loss = 0.0
                joint_total_loss = 0.0

                iter_data = tqdm(enumerate(dataloader), total=len(dataloader))
                for i, id_tensors in iter_data:
                    uid = id_tensors['user_id'].to(self.device)
                    input_item = id_tensors['input_item'].to(self.device)
                    input_beha = id_tensors['input_beha'].to(self.device)
                    target_item = id_tensors['target_item'].to(self.device)
                    target_neg = id_tensors['target_neg'].to(self.device)
                    target_beha = id_tensors['target_beha'].to(self.device)
                    
                    pos_ids = target_item[:, -1]
                    model_scores, diffu_rep, weights, t, item_rep_dis, seq_rep_dis, seq_rep =self.model(input_item, input_beha, pos_ids, target_beha, train_flag=True)
                    
                    test_item_emb = self.model.item_embeddings.weight # [I, D]
                    
                    
                    ce_loss = self.CELoss(diffu_rep, pos_ids, test_item_emb) 
                    
                    
                    joint_loss = ce_loss
                    self.optimizer.zero_grad()
                    joint_loss.backward()
                    self.optimizer.step()

                    ce_total_loss += ce_loss.item()
                    seq_ce_total_loss =0.0
                    joint_total_loss += joint_loss.item()

                ce_avg_loss = ce_total_loss / len(iter_data)
                seq_ce_avg_loss = seq_ce_total_loss / len(iter_data)
                joint_avg_loss = joint_total_loss / len(iter_data)
                self.scheduler.step(joint_avg_loss)

                if self.writer is not None:
                    self.writer.add_scalar('CE Loss', ce_avg_loss, epoch)
                    self.writer.add_scalar('SEQ CE Loss', seq_ce_avg_loss, epoch)
                    self.writer.add_scalar('Joint Loss', joint_avg_loss, epoch)

                loss_info = {
                    "Epoch": epoch + 1,
                    "CE Loss": "{:.6f}".format(ce_avg_loss),
                    "SEQ CE Loss": "{:.6f}".format(seq_ce_avg_loss),
                    "Joint Loss": "{:.6f}".format(joint_avg_loss)
                }

                if (epoch+1) % self.log_freq == 0:
                    print(set_color(str(loss_info), "yellow"))
                    self.logger.info(str(loss_info))

                if (epoch+1) % self.eval_freq == 0:
                    self.print_out_epoch = epoch + 1
                    
                    
                    scores, _ = self.valid()
                    early_stopping(scores, epoch + 1, self.model)
                    if early_stopping.early_stop:
                        print("Early Stopping")
                        self.logger.info("Early Stopping")

                        best_scores_info = {
                            "HR@5": "{:.4f}".format(early_stopping.best_scores[0]),
                            "NDCG@5": "{:.4f}".format(early_stopping.best_scores[1]),
                            "HR@10": "{:.4f}".format(early_stopping.best_scores[2]),
                            "NDCG@10": "{:.4f}".format(early_stopping.best_scores[3]),
                            "HR@20": "{:.4f}".format(early_stopping.best_scores[4]),
                            "NDCG@20": "{:.4f}".format(early_stopping.best_scores[5]),
                        }

                        print(set_color(f'\nBest Valid (' +
                                        str(early_stopping.best_valid_epoch) +
                                        ') Scores: ' +
                                        str(best_scores_info) + '\n', 'cyan'))
                        self.logger.info('Best Valid: %s, Scores: %s' % (str(early_stopping.best_valid_epoch), str(best_scores_info)))
                        
                        break

                    self.model.train()

            if not early_stopping.early_stop:
                print("Reach the max number of epochs!")
                self.logger.info("Reach the max number of epochs!")
                best_scores_info = {
                    "HR@5": "{:.4f}".format(early_stopping.best_scores[0]),
                    "NDCG@5": "{:.4f}".format(early_stopping.best_scores[1]),
                    "HR@10": "{:.4f}".format(early_stopping.best_scores[2]),
                    "NDCG@10": "{:.4f}".format(early_stopping.best_scores[3]),
                    "HR@20": "{:.4f}".format(early_stopping.best_scores[4]),
                    "NDCG@20": "{:.4f}".format(early_stopping.best_scores[5]),
                }

                print(set_color(f'\nBest Valid (' +
                                str(early_stopping.best_valid_epoch) +
                                ') Scores: ' +
                                str(best_scores_info) + '\n', 'cyan'))
                self.logger.info('Best Valid: %s, Scores: %s' % (str(early_stopping.best_valid_epoch), str(best_scores_info)))
                        

            # test phase
            self.model.load_state_dict(torch.load(self.output_path))
            _, test_info = self.test(record=False)
            print(set_color(f'\nFinal Test Metrics: ' +
                            test_info + '\n', 'pink'))
            self.logger.info('Final Test Metrics: %s' % (test_info))

        else:
            self.model.eval()
            iter_data = tqdm(enumerate(dataloader), total=len(dataloader))
            pred_item_list = None
            ground_truth_list = None

            with torch.no_grad():
                for i, id_tensors in iter_data:
                    '''
                    user_id: (b,)
                    input_item: (b, L)
                    input_beha: (b, L)
                    target_item: (b, L)
                    target_beha: (b, L)
                    ground_truth: (b, 1)
                    '''
                    uid = id_tensors['user_id'].to(self.device)
                    input_item = id_tensors['input_item'].to(self.device)
                    target_item = id_tensors['target_item'].to(self.device)
                    input_beha = id_tensors['input_beha'].to(self.device)
                    target_beha = id_tensors['target_beha'].to(self.device)
                    ground_truth = id_tensors['ground_truth'].to(self.device)
                    
                    scores_rec, rep_diffu, _, _, _, _, _ = self.model(input_item, input_beha, None, target_beha, train_flag=False) # TODO tag部分应该设置为None?
                    
                    item_prob = self.calculate_all_item_prob(rep_diffu).cpu().data.numpy().copy()
                    
                    
                    # extract top-20 prob of item idx
                    top_idx = np.argpartition(item_prob, -20)[:, -20:]
                    topn_prob = item_prob[np.arange(len(top_idx))[:, None], top_idx] # (b, 20)
                    # from large to small prob
                    topn_idx = np.argsort(topn_prob)[:, ::-1] # (b, 20)
                    batch_pred_item_list = top_idx[np.arange(len(top_idx))[:, None], topn_idx] # (b, 20)

                    if i == 0:
                        pred_item_list = batch_pred_item_list
                        ground_truth_list = ground_truth.cpu().data.numpy()
                    else:
                        pred_item_list = np.append(pred_item_list, batch_pred_item_list, axis=0)
                        ground_truth_list = np.append(ground_truth_list, ground_truth.cpu().data.numpy(), axis=0)

            return self.calculate_eval_metrics(mode, self.print_out_epoch,
                                               pred_item_list, ground_truth_list, record=record)

        if self.writer is not None:
            self.writer.close()
