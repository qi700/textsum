import os
import sys

# 添加project目录至环境变量
base_dir = os.path.abspath(os.path.dirname(__file__))
print(base_dir)
sys.path.append(base_dir)

import time
import argparse

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

import config
from data import Vocab
import data
from utils import calc_running_avg_loss
from config import USE_CUDA, DEVICE
from batcher import Batcher
from batcher import get_input_from_batch
from batcher import get_output_from_batch
from adagrad_custom import AdagradCustom
from torch.distributions import Categorical
from rouge import Rouge


class Train(object):
    def __init__(self,opt):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        #time.sleep(15)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        #train_dir = os.path.join(config.log_root, 'train_{}'.format(stamp))
        train_dir = os.path.join(config.log_root, 'train_logs')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        self.opt = opt
        self.summary_writer = tf.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter_step):
        """保存模型"""
        state = {
            'iter': iter_step,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
        #model_save_path = os.path.join(self.model_dir, 'model_{}_{}'.format(iter_step, stamp))
        model_save_path = os.path.join(self.model_dir, 'model_{}.tar'.format(iter_step))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        """模型初始化或加载、初始化迭代次数、损失、优化器"""
        # 初始化模型
        self.model = Model(model_file_path)
        # 模型参数的列表
        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        # 定义优化器
        # self.optimizer = optim.Adam(params, lr=config.adam_lr)
        # 使用AdagradCustom做优化器
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = AdagradCustom(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)
        # 初始化迭代次数和损失
        start_iter, start_loss = 0, 0
        # 如果传入的已存在的模型路径，加载模型继续训练
        if model_file_path is not None:
            state = torch.load(model_file_path, map_location = lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if USE_CUDA:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(DEVICE)
        if self.opt.new_lr is not None:
            self.optimizer = AdagradCustom(params, lr=self.opt.new_lr, initial_accumulator_value=config.adagrad_init_acc)

        return start_iter, start_loss

    def train_one_batch(self, batch):
        """
        训练一个batch，返回该batch的loss。
        enc_batch:             torch.Size([16, 400]), 16篇文章的编码，不足400词的用pad的编码补足, oov词汇用0编码；
        enc_padding_mask:      torch.Size([16, 400]), 对应pad的位置为0，其余为1；
        enc_lens:              numpy.ndarray, 列表内每个元素表示每篇article的单词数；
        enc_batch_extend_vocab:torch.Size([16, 400]), 16篇文章的编码;oov词汇用超过词汇表的编码；
        extra_zeros:           torch.Size([16, 文章oov词汇数量]) zero tensor;
        c_t_1:                 torch.Size([16, 512]) zero tensor;
        coverage:              Variable(torch.zeros(batch_size, max_enc_seq_len)) if is_coverage==True else None;coverage模式时后续有值
        ----------------------------------------
        dec_batch:             torch.Size([16, 100]) 摘要编码含有开始符号编码以及PAD；
        dec_padding_mask:      torch.Size([16, 100]) 对应pad的位置为0，其余为1；
        max_dec_len:           标量，摘要词语数量，不包含pad
        dec_lens_var:          torch.Size([16] 摘要词汇数量         
        target_batch:          torch.Size([16, 100]) 目标摘要编码含有STOP符号编码以及PAD
        """
        self.optimizer.zero_grad()
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch)
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        if self.opt.train_mle == "yes":
            mle_loss = self.train_batch_MLE(encoder_outputs, encoder_hidden,enc_padding_mask, encoder_feature,enc_batch_extend_vocab, extra_zeros, c_t_1, batch, coverage)
        else:
            mle_loss = get_cuda(torch.FloatTensor([0]))
        # --------------RL training-----------------------------------------------------
        if self.opt.train_rl == "yes":                                                              #perform reinforcement learning training
            # multinomial sampling
            sample_sents, RL_log_probs = self.train_batch_RL(encoder_outputs, encoder_hidden, enc_padding_mask, encoder_feature, enc_batch_extend_vocab, extra_zeros, c_t_1, batch.art_oovs,coverage, greedy=False)
            with torch.autograd.no_grad():
                # greedy sampling
                greedy_sents, _ = self.train_batch_RL(encoder_outputs, encoder_hidden, enc_padding_mask, encoder_feature, enc_batch_extend_vocab, extra_zeros, c_t_1, batch.art_oovs, coverage, greedy=True)

            sample_reward = self.reward_function(sample_sents, batch.original_abstracts)
            baseline_reward = self.reward_function(greedy_sents, batch.original_abstracts)
            # if iter%200 == 0:
            #     self.write_to_file(sample_sents, greedy_sents, batch.original_abstracts, sample_reward, baseline_reward, iter)
            rl_loss = -(sample_reward - baseline_reward) * RL_log_probs                             #Self-critic policy gradient training (eq 15 in https://arxiv.org/pdf/1705.04304.pdf)
            rl_loss = torch.mean(rl_loss)

            batch_reward = torch.mean(sample_reward).item()
        else:
            rl_loss = get_cuda(torch.FloatTensor([0]))
            batch_reward = 0
        
        
        (self.opt.mle_weight * mle_loss + self.opt.rl_weight * rl_loss).backward()
        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return mle_loss.item(), batch_reward

    
    def train_batch_MLE(self,encoder_outputs, encoder_hidden,enc_padding_mask, encoder_feature,enc_batch_extend_vocab, extra_zeros, c_t_1, batch, coverage):
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch)
       
       
        # [B, max(seq_lens), 2*hid_dim], [B*max(seq_lens), 2*hid_dim], tuple([2, B, hid_dim], [2, B, hid_dim])
        s_t_1 = self.model.reduce_state(encoder_hidden)   # (h,c) = ([1, B, hid_dim], [1, B, hid_dim])
        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]      # 摘要的一个单词，batch里的每个句子的同一位置的单词编码
            # print("y_t_1:", y_t_1, y_t_1.size())
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage, di)
            target = target_batch[:, di]  # 摘要的下一个单词的编码
            # print("target-iter:", target, target.size())
            # print("final_dist:", final_dist, final_dist.size())
            # input("go on>>")
            # final_dist 是词汇表每个单词的概率，词汇表是扩展之后的词汇表，也就是大于预设的50_000
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()   # 取出目标单词的概率gold_probs
            step_loss = -torch.log(gold_probs + config.eps)  # 最大化gold_probs，也就是最小化step_loss（添加负号）
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
                
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        
        return loss
    
    def train_batch_RL(self,encoder_outputs, encoder_hidden, enc_padding_mask, encoder_feature, enc_batch_extend_vocab, extra_zeros, c_t_1, article_oovs, coverage, greedy):
        '''Generate sentences from decoder entirely using sampled tokens as input. These sentences are used for ROUGE evaluation
        Args
        :param enc_out: Outputs of the encoder for all time steps (batch_size, length_input_sequence, 2*hidden_size)
        :param enc_hidden: Tuple containing final hidden state & cell state of encoder. Shape of h & c: (batch_size, hidden_size)
        :param enc_padding_mask: Mask for encoder input; Tensor of size (batch_size, length_input_sequence) with values of 0 for pad tokens & 1 for others
        :param ct_e: encoder context vector for time_step=0 (eq 5 in https://arxiv.org/pdf/1705.04304.pdf)
        :param extra_zeros: Tensor used to extend vocab distribution for pointer mechanism
        :param enc_batch_extend_vocab: Input batch that stores OOV ids
        :param article_oovs: Batch containing list of OOVs in each example
        :param greedy: If true, performs greedy based sampling, else performs multinomial sampling
        Returns:
        :decoded_strs: List of decoded sentences
        :log_probs: Log probabilities of sampled words
        '''
        s_t_1 = self.model.reduce_state(encoder_hidden)                                                                        #Decoder hidden states
        y_t_1 = get_cuda(torch.LongTensor(len(encoder_outputs)).fill_(self.vocab.word2id(data.START_DECODING)))                                     #Input to the decoder                                                              #Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        inds = []                                                                                   #Stores sampled indices for each time step
        decoder_padding_mask = []                                                                   #Stores padding masks of generated samples
        log_probs = []                                                                              #Stores log probabilites of generated samples
        mask = get_cuda(torch.LongTensor(len(encoder_outputs)).fill_(1))                                        #Values that indicate whether [STOP] token has already been encountered; 1 => Not encountered, 0 otherwise

        for t in range(config.max_dec_steps):
            probs, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage, t)
            if greedy is False:
                multi_dist = Categorical(probs)
                y_t_1 = multi_dist.sample()                                                           #perform multinomial sampling
                log_prob = multi_dist.log_prob(y_t_1)
                log_probs.append(log_prob)
            else:
                _, y_t_1 = torch.max(probs, dim=1)                                                        #perform greedy sampling
            y_t_1 = y_t_1.detach()
            inds.append(y_t_1)
            mask_t = get_cuda(torch.zeros(len(encoder_outputs)))                                                #Padding mask of batch for current time step
            mask_t[mask == 1] = 1                                                                   #If [STOP] is not encountered till previous time step, mask_t = 1 else mask_t = 0
            mask[(mask == 1) + (y_t_1 == self.vocab.word2id(data.STOP_DECODING)) == 2] = 0                                       #If [STOP] is not encountered till previous time step and current word is [STOP], make mask = 0
            decoder_padding_mask.append(mask_t)
            is_oov = (y_t_1>=config.vocab_size).long()                                                #Mask indicating whether sampled word is OOV
            y_t_1 = (1-is_oov)*y_t_1 + (is_oov)*self.vocab.word2id(data.UNKNOWN_TOKEN)                                          #Replace OOVs with [UNK] token

        inds = torch.stack(inds, dim=1)
        decoder_padding_mask = torch.stack(decoder_padding_mask, dim=1)
        if greedy is False:                                                                         #If multinomial based sampling, compute log probabilites of sampled words
            log_probs = torch.stack(log_probs, dim=1)
            log_probs = log_probs * decoder_padding_mask                                            #Not considering sampled words with padding mask = 0
            lens = torch.sum(decoder_padding_mask, dim=1)                                               #Length of sampled sentence
            log_probs = torch.sum(log_probs, dim=1) / lens  # (bs,)                                     #compute normalizied log probability of a sentence
        decoded_strs = []
        for i in range(len(encoder_outputs)):
            id_list = inds[i].cpu().numpy()
            oovs = article_oovs[i]
            S = data.outputids2words(id_list, self.vocab, oovs)                                     #Generate sentence corresponding to sampled words
            try:
                end_idx = S.index(data.STOP_DECODING)
                S = S[:end_idx]
            except ValueError:
                S = S
            if len(S) < 2:                                                                           #If length of sentence is less than 2 words, replace it with "xxx"; Avoids setences like "." which throws error while calculating ROUGE
                S = ["xxx"]
            S = " ".join(S)
            decoded_strs.append(S)

        return decoded_strs, log_probs

    def reward_function(self, decoded_sents, original_sents):
        rouge = Rouge()
        try:
            scores = rouge.get_scores(decoded_sents, original_sents)
        except Exception:
            print("Rouge failed for multi sentence evaluation.. Finding exact pair")
            scores = []
            for i in range(len(decoded_sents)):
                try:
                    score = rouge.get_scores(decoded_sents[i], original_sents[i])
                except Exception:
                    print("Error occured at:")
                    print("decoded_sents:", decoded_sents[i])
                    print("original_sents:", original_sents[i])
                    score = [{"rouge-l":{"f":0.0}}]
                scores.append(score[0])
        rouge_l_f1 = [score["rouge-l"]["f"] for score in scores]
        rouge_l_f1 = get_cuda(torch.FloatTensor(rouge_l_f1))
        return rouge_l_f1


    def trainIters(self, n_iters):
        # 训练设置，包括
        if self.opt.load_model != None:
            model_file_path = config.save_model_path + "/" + self.opt.load_model
        else:
            model_file_path = None

        iter_step, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        count = mle_total = r_total = 0
        while iter_step < n_iters:
            # 获取下一个batch数据
            batch = self.batcher.next_batch()
            mle_loss, r = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(mle_loss, running_avg_loss, self.summary_writer, iter_step)
            iter_step += 1
            mle_total += mle_loss
            r_total += r
            count += 1

            if iter_step % 10 == 0:
                self.summary_writer.flush()
                
            
            # print_interval = 1000
            if iter_step % 10 == 0:
                # lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                print('steps %d, seconds for %d steps: %.2f, loss: %f' % (iter_step, 10,
                                                                          time.time() - start, mle_loss))
                start = time.time()
                mle_avg = mle_total / count
                r_avg = r_total / count
                print("iter:", iter, "mle_loss:", "%.3f" % mle_avg, "reward:", "%.4f" % r_avg)
                count = mle_total = r_total = 0

            # 5000次迭代就保存一下模型
            if iter_step % 50 == 0:
                self.save_model(running_avg_loss, iter_step)

def init_print():
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    print("时间:{}".format(stamp))
    print("***参数:***")
    for k, v in config.__dict__.items():
        if not k.startswith("__"):
            print(":".join([k, str(v)]))

def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_path", 
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()
    init_print()
    train_processor = Train()
    train_processor.trainIters(config.max_iterations, args.model_path)
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mle', type=str, default="yes")
    parser.add_argument('--train_rl', type=str, default="no")
    parser.add_argument('--mle_weight', type=float, default=1.0)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--new_lr', type=float, default=None)
    opt = parser.parse_args()
    opt.rl_weight = 1 - opt.mle_weight
    print("Training mle: %s, Training rl: %s, mle weight: %.2f, rl weight: %.2f"%(opt.train_mle, opt.train_rl, opt.mle_weight, opt.rl_weight))
    print("intra_encoder:", config.intra_encoder, "intra_decoder:", config.intra_decoder)
    train_processor = Train(opt)
    train_processor.trainIters(config.max_iterations)
