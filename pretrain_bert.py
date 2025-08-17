import random
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from transformers import BertConfig, BertForMaskedLM, BertTokenizer 

bert_tokenizer = BertTokenizer.from_pretrained(r'C:\Users\Jay\pythonProject\pretrain_bert-main\vocab.txt')
bert_config = BertConfig(
    vocab_size=len(bert_tokenizer),
    hidden_size=512,  # hidden_size//num_hidden_layers=0
    num_hidden_layers=4,
    num_attention_heads=4,
    max_position_embeddings=1024,
)
mlm_prob=0.15
weight_decay = 0.01
warmup_ratio = 0.1
eps = 1e-8
learning_rate = 5e-5
batch_size = 4
num_epochs = 50
max_length = 1024  # max_position_embeddings
logging_step = 400
save_steps = 800
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


def build_optimizer(model, train_steps):
    no_decay = ['bias', 'LayerNorm.weight']

    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=eps)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * warmup_ratio, t_total=train_steps)

    return optimizer, scheduler


# geohash7转35位二进制
def geohash7_to_binary(geohash7):
    if geohash7 == -1:
        return [random.randint(0, 1) for _ in range(35)]
    elif geohash7 in bert_tokenizer.vocab:
        return [int(b) for b in bin(bert_tokenizer.convert_tokens_to_ids(geohash7))[2:].zfill(35)]
    else:
        char_to_bin = {char: format(i, '05b') for i, char in enumerate('0123456789bcdefghjkmnpqrstuvwxyz')}
        binary_parts = []
        for char in geohash7:
            binary_parts.append(char_to_bin[char])
        return binary_parts


class MLMDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.special_tokens_nums = len(bert_tokenizer.all_special_ids)
        self.vocab_size = bert_tokenizer.vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item]
        text_ids = self.truncate(text, max_len=max_length)
        text_ids, output_ids = self.random_mask(text_ids)  # 添加mask预测

        # 拼接 cls xxx sep
        input_ids = [bert_tokenizer.cls_token_id] + text_ids + [bert_tokenizer.sep_token_id]

        token_type_ids = [0] * (len(text_ids) + 2)
        # -100 index = padding token
        labels = [-100] + output_ids + [-100]

        # padding
        # pad_token_id=0
        input_ids = self.padding(input_ids, bert_tokenizer.pad_token_id)
        token_type_ids = self.padding(token_type_ids, 0)
        labels = self.padding(labels, -100)
        # attention_mask = (input_ids != 0)
        attention_mask = [1 if i != 0 else 0 for i in input_ids]
        assert len(input_ids) == len(token_type_ids) == len(labels)
        return {'input_ids': torch.tensor(input_ids), 'token_type_ids': torch.tensor(token_type_ids),
                'attention_mask': torch.tensor(attention_mask),
                'labels': torch.tensor(labels)}

    def random_mask(self, text_ids):
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids))

        for r, i in zip(rands, text_ids):
            if r < mlm_prob * 0.8:   ## mask预测自己 0 ~ 0.12
                input_ids.append(bert_tokenizer.mask_token_id)
                output_ids.append(i)
            elif r < mlm_prob * 0.9:  ## 自己预测自己 0.12 ~ 0.135
                input_ids.append(i)
                output_ids.append(i)
            elif r < mlm_prob:    ## 随机的一个词预测自己，随机词不会从特殊符号中选取，有小概率抽到自己 0.135 ~ 0.15
                input_ids.append(-1)  # 随机生成一个词
                output_ids.append(i)
            else:
                input_ids.append(i)
                output_ids.append(-100)  # 保持原样不预测
        
        input_ids = [geohash7_to_binary(i) for i in input_ids]
        output_ids = [geohash7_to_binary(i) for i in output_ids]

        return input_ids, output_ids

    def truncate(self, token_ids, max_len=max_length):
        max_len -= 2  # 空留给cls sep
        assert max_len >= 0
        return token_ids[:max_len]

    def padding(self, token_ids, pad_id, max_len=max_length):
        token_ids = token_ids + [pad_id] * (max_len - len(token_ids))
        return token_ids



if __name__ == '__main__':

    corpus = [
        ['wtw3sjq', 'wtw3sjr', 'wtw3sjt', 'wtw3sjw', 'wtw3sjx', 'wtw3sk0', 'wtw3sk1'],
        ['wtw3sk5', 'wtw3sk6', 'wtw3sk7', 'wtw3sk8', 'wtw3sk9', 'wtw3skb', 'wtw3skc', 'wtw3skd', 'wtw3ske', 'wtw3skf'],
        ['wtw3skg', 'wtw3skh', 'wtw3skj', 'wtw3skk'],
        ['wtw3skt', 'wtw3sku', 'wtw3skv', 'wtw3skw', 'wtw3skx', 'wtw3sky'],
        ['wtw3sm3', 'wtw3sm4', 'wtw3sm6', 'wtw3sm7', 'wtw3sm8', 'wtw3sm9', 'wtw3smb', 'wtw3smc', 'wtw3smd'],
        ['wtw3smf', 'wtw3smg', 'wtw3smh'],
        ['wtw3sms', 'wtw3smu', 'wtw3smv', 'wtw3smw', 'wtw3smx', 'wtw3smz', 'wtw3sn0']
    ]
    # print(pd.Series(seq_lens).describe())
    bert_model = BertForMaskedLM(bert_config)

    train_dataset = MLMDataset(corpus)
    valid_dataset = MLMDataset(corpus)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

    total_steps = num_epochs * len(train_dataloader)
    optimizer, scheduler = build_optimizer(bert_model, total_steps)

    bert_model.to(device)
    bert_model.train()

    total_loss, cur_avg_loss, global_steps = 0., 0., 0

    # 训练：
    for epoch in range(20):
        for data in tqdm(train_dataloader):
            # print(data)
            input_ids = data['input_ids'].to(device).long()
            token_type_ids = data['token_type_ids'].to(device).long()
            attention_mask = data['attention_mask'].to(device).long()
            labels = data['labels'].to(device).long()

            outputs = bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 labels=labels)
            loss = outputs['loss']
            loss.backward()
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            cur_avg_loss += loss.item()

            if (global_steps + 1) % logging_step == 0:
                epoch_avg_loss = cur_avg_loss / logging_step
                global_avg_loss = total_loss / (global_steps + 1)

                print(f"\n>> epoch - {epoch},  global steps - {global_steps + 1}, "
                      f"epoch avg loss - {epoch_avg_loss:.4f}, global avg loss - {global_avg_loss:.4f}.")

                cur_avg_loss = 0.0
            global_steps += 1

            model_save_path = f'pretrained_models/checkpoint-{global_steps}'
            if (global_steps + 1) % save_steps == 0:
                bert_model.save_pretrained(model_save_path)
                bert_tokenizer.save_vocabulary(model_save_path)

                print(f'\n>> model saved in : {model_save_path} .')
