import os
import random

import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from config import set_args
from model import Model
from torch.utils.data import DataLoader
from utils import l2_normalize, compute_corrcoef, compute_pearsonr
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from data_helper import CustomDataset, collate_fn, pad_to_maxlen, load_data, load_test_data, my_load_data, my_collate_fn, my_load_test_data
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

def write_arg_log(s):
    logs_path = os.path.join(args.output_dir, 'arg_log.txt')
    with open(logs_path, 'a+') as f:
        f.write('\n')
        for k, v in s.items():
            # v += '\n'
            k = k + " :  "
            f.write(k)
            f.write(str(v))
            f.write('\n')


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def my_get_sent_id_tensor(s_list, max_len = 512):
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []

    for s in s_list:
        inputs = tokenizer.encode_plus(text=s, add_special_tokens=True, max_length = max_len, pad_to_max_length=True, return_token_type_ids=True)
        input_ids.append(inputs['input_ids'])
        attention_mask.append(inputs['attention_mask'])
        token_type_ids.append(inputs['token_type_ids'])
        assert (len(inputs['input_ids']) == max_len)

    all_input_ids = torch.tensor(input_ids)
    all_input_mask = torch.tensor(attention_mask)
    all_segment_ids = torch.tensor(token_type_ids)
    return all_input_ids, all_input_mask, all_segment_ids

def get_sent_id_tensor(s_list):
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    max_len = max([len(_)+2 for _ in s_list])
    for s in s_list:
        inputs = tokenizer.encode_plus(text=s, text_pair=None, add_special_tokens=True, return_token_type_ids=True)
        input_ids.append(pad_to_maxlen(inputs['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(inputs['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(inputs['token_type_ids'], max_len=max_len))
    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids


def evaluate():
    sent1, sent2, label = my_load_test_data(args.test_data)
    all_a_vecs = []
    all_b_vecs = []
    all_labels = []
    model.eval()
    for s1, s2, lab in tqdm(zip(sent1, sent2, label)):
        input_ids, input_mask, segment_ids = my_get_sent_id_tensor([s1, s2]) #[s1, s2] = [doc, code]
        lab = torch.tensor([lab], dtype=torch.float)
        if torch.cuda.is_available():
            input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
            lab = lab.cuda()

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=input_mask, encoder_type='cls')

        all_a_vecs.append(output[0].cpu().numpy())
        all_b_vecs.append(output[1].cpu().numpy())
        all_labels.extend(lab.cpu().numpy())

    all_a_vecs = np.array(all_a_vecs)
    all_b_vecs = np.array(all_b_vecs)
    all_labels = np.array(all_labels)

    a_vecs = l2_normalize(all_a_vecs)
    b_vecs = l2_normalize(all_b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(all_labels, sims)
    pearsonr = compute_pearsonr(all_labels, sims)
    return corrcoef, pearsonr


def calc_loss(y_true, y_pred):
    y_true = y_true[::2]

    norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5

    y_pred = y_pred / norms

    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20

    y_pred = y_pred[:, None] - y_pred[None, :]

    y_true = y_true[:, None] < y_true[None, :]
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)

    if torch.cuda.is_available():
        y_pred = torch.cat((torch.tensor([0]).float().cuda(), y_pred), dim=0)
    else:
        y_pred = torch.cat((torch.tensor([0]).float(), y_pred), dim=0)
        
    reslut = torch.logsumexp(y_pred, dim=0)

    del y_pred
    del y_true

    return reslut


if __name__ == '__main__':
    args = set_args()
    set_seed()

    #training log
    write_arg_log(vars(args))

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

    # load data
    train_sentence, train_label = my_load_data(args.train_data)

    train_dataset = CustomDataset(sentence=train_sentence, label=train_label, tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.train_batch_size, collate_fn=my_collate_fn)

    total_steps = len(train_dataloader) * args.num_train_epochs

    num_train_optimization_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    model = Model()

    if torch.cuda.is_available():
        model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)
    scaler = GradScaler()

    print("***** Running training *****")
    print("  Num examples = %d" % len(train_dataset))
    print("  Batch size = %d" % args.train_batch_size)
    print("  Num steps = %d" % num_train_optimization_steps)

    lossfuction_1 = nn.CrossEntropyLoss()

    progress_bar_out = tqdm(range(args.num_train_epochs * len(train_dataloader)))
    for epoch in range(args.num_train_epochs):
        model.train()
        train_label, train_predict = [], []
        epoch_loss = 0
        count = 0
        for step, batch in enumerate(train_dataloader):
            input_ids, input_mask, segment_ids, label_ids = batch
            if torch.cuda.is_available():
                input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
                label_ids = label_ids.cuda()

            with autocast():
                all_hidden_states = model(input_ids=input_ids, attention_mask=input_mask, encoder_type='all')

                loss = torch.tensor(0,dtype=float).cuda()

                index = 0
                index_layer = [0,1,2,9,10,11]  #layers
                for hidden_state in all_hidden_states:
                    if (index in index_layer):
                        hidden_state = hidden_state[:, 0, :]

                        layer_i_loss = calc_loss(label_ids, hidden_state)
                        loss = loss + layer_i_loss

                        ################### Reorganize Data BEGIN#########################################

                        temp_label_ids = label_ids
                        temp_label_ids = temp_label_ids[::2]

                        doc_batch_outputs = hidden_state[::2]
                        code_batch_outputs = hidden_state[1::2]

                        assert (len(doc_batch_outputs) + len(code_batch_outputs) == len(hidden_state))

                        idx = 0
                        for lable in temp_label_ids:

                            examples = []
                            neg_pos_labels = []

                            if (lable == 1):
                                pos_doc_embedding = doc_batch_outputs[idx]
                                pos_code_embedding = code_batch_outputs[idx]

                                examples.append(pos_doc_embedding)
                                examples.append(pos_code_embedding)
                                neg_pos_labels.append(1.)
                                neg_pos_labels.append(1.)

                                jump = 0
                                for code_em in code_batch_outputs:
                                    if (jump != idx):
                                        examples.append(pos_doc_embedding)
                                        examples.append(code_em)
                                        neg_pos_labels.append(0.)
                                        neg_pos_labels.append(0.)
                                    jump += 1

                                assert (len(examples) == len(neg_pos_labels))
                                assert (len(neg_pos_labels) == (len(temp_label_ids) * 2))

                                examples_cuda = torch.stack(
                                    examples)
                                lables_cuda = torch.tensor(neg_pos_labels,dtype=torch.float16).cuda()

                                batch_loss = calc_loss(lables_cuda, examples_cuda)
                                loss = loss + batch_loss

                            idx += 1

                        ################### Reorganize Data END#########################################



                    index += 1


            if (count % 20 == 0):
                print("epoch:{}, step:{}/{}, Loss:{:10f}".format(epoch, step, len(train_dataloader), loss))  # 在进度条前面定义一段文字
            count += 1

            scaler.scale(loss).backward()
            epoch_loss += loss.item()


            if (step + 1) % args.gradient_accumulation_steps == 0:
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar_out.update(1)

        # save model in each epoch
        out_model_path = os.path.join(args.output_dir, "Epoch-{}.pkl".format(epoch))
        torch.save(model, out_model_path)

