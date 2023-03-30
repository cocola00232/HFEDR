import torch
from torch.utils.data import Dataset



def my_load_data(path):
    sentence, label = [], []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()

        for line in lines:
            temp_line = line.split("<CODESPLIT>")
            if (len(temp_line)) == 5:
                sentence.append(temp_line[-2])  # doc
                sentence.append(temp_line[-1])  # code
                label.append(int(temp_line[0]))
                label.append(int(temp_line[0]))

    return sentence, label


def load_data(path):
    sentence, label = [], []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            try:
                sentence.extend([line[0], line[1]])
                lab = int(line[2])
                label.extend([lab, lab])
            except:
                continue
    return sentence, label

def my_load_test_data(path):
    sent1, sent2, label = [], [], []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            temp_line = line.split("<CODESPLIT>")
            if (len(temp_line)) == 5:
                sent1.append(temp_line[-2])  # doc
                sent2.append(temp_line[-1])  # code
                label.append(int(temp_line[0]))

    return sent1, sent2, label

def load_test_data(path):
    sent1, sent2, label = [], [], []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            sent1.append(line[0])
            sent2.append(line[1])
            label.append(int(line[2]))
    return sent1, sent2, label


class CustomDataset(Dataset):
    def __init__(self, sentence, label, tokenizer):
        self.sentence = sentence
        self.label = label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):

        inputs = self.tokenizer.encode_plus(
            text=self.sentence[index],
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': self.label[index]
        }


def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids

def my_collate_fn(batch):

    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    for item in batch:
        input_ids.append(item['input_ids'])
        attention_mask.append(item['attention_mask'])
        token_type_ids.append(item['token_type_ids'])
        labels.append(item['label'])

    all_input_ids = torch.tensor(input_ids)
    all_input_mask = torch.tensor(attention_mask)
    all_segment_ids = torch.tensor(token_type_ids)
    all_label_ids = torch.tensor(labels, dtype=torch.float)

    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids

def collate_fn(batch):
    max_len = max([len(d['input_ids']) for d in batch])


    input_ids, attention_mask, token_type_ids, labels = [], [], [], []

    for item in batch:
        input_ids.append(pad_to_maxlen(item['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(item['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(item['token_type_ids'], max_len=max_len))
        labels.append(item['label'])

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    all_label_ids = torch.tensor(labels, dtype=torch.float)
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids
