import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils import l2_normalize
from transformers import AutoTokenizer, AutoModel


class LineByLineTextDataset(Dataset):
    def __init__(self, file_path: str):
        assert os.path.isfile(file_path)
        print("read data file at:", file_path)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.text_lines = []
        self.code_lines = []
        self.labels = []

        # self.lines = self.lines[:3000]

        for line in self.lines:
            temp_line = line.split("<CODESPLIT>")
            if (len(temp_line)) == 5:

                self.text_lines.append(temp_line[-2])
                self.code_lines.append(temp_line[-1])
                self.labels.append(int(temp_line[0]))

        print("注释和代码总行数:", len(self.text_lines), len(self.code_lines))

    def __len__(self):
        return len(self.text_lines)

    def __getitem__(self, i):
        a = self.text_lines[i]
        b = self.code_lines[i]
        c = self.labels[i]
        return a, b, c


def inference_eval(batch_size, infer_file_path, tokenizer, model, batch_i):
    ########################## Data ############################
    infer_dataset = LineByLineTextDataset(file_path=infer_file_path)
    infer_dataLoader = DataLoader(infer_dataset, batch_size, shuffle=False)

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("train_device: ", device)

    ######################### Extract #########################

    model.eval()
    size = len(infer_dataLoader)
    test_progress_bar = tqdm(range(size))

    doc_lay_1, doc_lay_2, doc_lay_3, doc_lay_4, doc_lay_5, doc_lay_6 = [],[],[],[],[],[]
    all_doc_lays = [doc_lay_1, doc_lay_2, doc_lay_3, doc_lay_4, doc_lay_5, doc_lay_6]

    for text, code, labels in infer_dataLoader:

        text_list = list(text)
        code_list = list(code)

        text_batch_tokenized = tokenizer.batch_encode_plus(text_list, add_special_tokens=True, max_length=128,
                                                           pad_to_max_length=True)
        text_input_ids = torch.tensor(text_batch_tokenized['input_ids']).to(device)
        text_attention_mask = torch.tensor(text_batch_tokenized['attention_mask']).to(device)

        with torch.no_grad():
            text_output = model(input_ids=text_input_ids, attention_mask=text_attention_mask,
                                encoder_type='iter_n_layer')

        k = [0,1,2,9,10,11]

        temp = 0
        all_doc_lays_index = 0
        for text_layer in text_output:
            if(temp in k):

                text_layer = text_layer[:, 0, :]
                for i in range(batch_size):
                    all_doc_lays[all_doc_lays_index].append(text_layer[i].cpu().numpy())

                all_doc_lays_index += 1
            temp += 1

        test_progress_bar.update(1)


    print(len(all_doc_lays[0]))

    #save
    layer_file_index = 0
    for layer_file in all_doc_lays:
        torch.save(layer_file, "../embedding/" +"batch_"+str(batch_i)+"/"+ str(layer_file_index) + "_layer_doc.data")
        layer_file_index += 1


if __name__ == '__main__':
    # 配置
    batch_size = 200

    inference_model_name = "../outputs_cons_model/Epoch-4.pkl"   #arg1

    tokenizer = AutoTokenizer.from_pretrained("../codebert")
    model = torch.load(inference_model_name)
    model.cuda()

    for i in [0]: #arg2
        infer_file_path = "../classify/data/test/java/batch_" + str(i) + ".txt"
        print(infer_file_path)
        inference_eval(batch_size, infer_file_path, tokenizer, model, i)

