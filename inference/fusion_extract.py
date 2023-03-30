import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sim_refer import get_mrr
from numpy import *
from model import Model

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

                self.text_lines.append(temp_line[-2]) #
                self.code_lines.append(temp_line[-1]) #
                self.labels.append(int(temp_line[0]))

        print("注释和代码总行数:", len(self.text_lines), len(self.code_lines))

    def __len__(self):
        return len(self.text_lines)

    def __getitem__(self, i):
        a = self.text_lines[i]
        b = self.code_lines[i]
        c = self.labels[i]
        return a, b, c

def extract_code_embedding(batch_size, infer_file_path, tokenizer, model, batch_i):
    ########################## data #########################
    infer_dataset = LineByLineTextDataset(file_path=infer_file_path)
    infer_dataLoader = DataLoader(infer_dataset, batch_size, shuffle=False)
    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("train_device: ", device)
    ######################### Inference #########################
    model.eval()
    size = len(infer_dataLoader)
    test_progress_bar = tqdm(range(size))

    # The features of each layer are taken out separately and calculated. After this layer is calculated, the next layer is calculated.
    cod_lay_1, cod_lay_2, cod_lay_3, cod_lay_4, cod_lay_5, cod_lay_6 = [], [], [], [], [], []
    all_cod_lays = [cod_lay_1, cod_lay_2, cod_lay_3, cod_lay_4,cod_lay_5, cod_lay_6]  # 1

    for text, code, labels in infer_dataLoader:
        code_list = list(code)
        code_batch_tokenized = tokenizer.batch_encode_plus(code_list, add_special_tokens=True, max_length=128,
                                                           pad_to_max_length=True)
        code_input_ids = torch.tensor(code_batch_tokenized['input_ids']).to(device)
        code_attention_mask = torch.tensor(code_batch_tokenized['attention_mask']).to(device)

        with torch.no_grad():
            code_output = model(input_ids=code_input_ids, attention_mask=code_attention_mask,
                                encoder_type='iter_n_layer')

        k = [0,1,2,9,10,11]  #A1
        temp = 0
        all_doc_lays_index = 0  #
        for code_layer in code_output:
            if (temp in k):
                code_layer = code_layer[:, 0, :]
                for i in range(batch_size):
                    all_cod_lays[all_doc_lays_index].append(code_layer[i].cpu().numpy())  # 2

                all_doc_lays_index += 1
            temp += 1

        test_progress_bar.update(1)

    # 3
    layer_file_index = 0
    for layer_file in all_cod_lays:
        torch.save(layer_file,
                   "../embedding/" + "batch_i/" + str(layer_file_index) + "_layer_code.data")
        layer_file_index += 1


def extract_doc_embedding(batch_size, infer_file_path, tokenizer, model, batch_i):
    ########################## data ############################
    infer_dataset = LineByLineTextDataset(file_path=infer_file_path)
    infer_dataLoader = DataLoader(infer_dataset, batch_size, shuffle=False)

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("train_device: ", device)

    ######################### Extract #########################

    model.eval()
    size = len(infer_dataLoader)
    test_progress_bar = tqdm(range(size))

    doc_lay_1, doc_lay_2, doc_lay_3, doc_lay_4, doc_lay_5, doc_lay_6 = [], [], [], [], [], []
    all_doc_lays = [doc_lay_1, doc_lay_2, doc_lay_3, doc_lay_4, doc_lay_5, doc_lay_6]

    skip = 0
    for text, code, labels in infer_dataLoader:

        text_list = list(text)
        code_list = list(code)

        text_batch_tokenized = tokenizer.batch_encode_plus(text_list, add_special_tokens=True, max_length=128,
                                                           pad_to_max_length=True)
        text_input_ids = torch.tensor(text_batch_tokenized['input_ids']).to(device)
        text_attention_mask = torch.tensor(text_batch_tokenized['attention_mask']).to(device)

        if(skip % 1000 == 0): #0， 1000, 2000
            with torch.no_grad():
                text_output = model(input_ids=text_input_ids, attention_mask=text_attention_mask,
                                    encoder_type='iter_n_layer')  # shape: 12 * batch_size* maxlen * 768
        skip += batch_size

        k = [0,1,2,9,10,11]  #arg3 #A1

        temp = 0
        all_doc_lays_index = 0
        for text_layer in text_output:
            if (temp in k):
                # 截取cls
                text_layer = text_layer[:, 0, :]
                for i in range(batch_size):
                    all_doc_lays[all_doc_lays_index].append(text_layer[i].cpu().numpy())

                all_doc_lays_index += 1
            temp += 1

        test_progress_bar.update(1)

    print("len(all_doc_lays[0])", len(all_doc_lays[0]))

    # save files
    layer_file_index = 0
    for layer_file in all_doc_lays:  # arg4  last arg
        torch.save(layer_file,
                   "../embedding/" + "batch_i/" + str(layer_file_index) + "_layer_doc.data")
        layer_file_index += 1



if __name__ == '__main__':

    # setting begin #####
    batch_size = 200
    inference_model_name = "../outputs_model/Epoch-1.pkl" #arg1
    lang = 'ruby'
    # setting end #####

    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")  # tokenizer
    model = torch.load(inference_model_name)
    model.cuda()

    all_mrr = []
    all_rk_1 = []
    all_rk_5 = []
    all_rk_10 = []
    for i in [0]: #batch N  The number of the test files #arg2
        infer_file_path = "../data/test/"+ lang +"/batch_" + str(i) + ".txt" #arg3 #last arg
        print(infer_file_path)

        extract_doc_embedding(batch_size, infer_file_path, tokenizer, model, i)
        extract_code_embedding(batch_size, infer_file_path, tokenizer, model, i)

        # If the memory size is sufficient (>80G), mrr can be calculated directly,

        # batch_i_mean_mrr, rk_1, rk_5, rk_10 = get_mrr(i)
        # all_mrr.append(batch_i_mean_mrr)
        # all_rk_1.append(rk_1)
        # all_rk_5.append(rk_5)
        # all_rk_10.append(rk_10)
        #
        # print("all_mean_mrr now: ", mean(all_mrr))
        # print("all_mean_rk_1 now: ", mean(all_rk_1))
        # print("all_mean_rk_5 now: ", mean(all_rk_5))
        # print("all_mean_rk_10 now: ", mean(all_rk_10))



        #delete files for save storage
        # for file_id in [0, 1, 2, 3, 4, 5]:
        #     doc_embedding_file = "../embedding/batch_i/" + str(file_id) + "_layer_doc.data"
        #     code_embedding_file = "../embedding/batch_i/" + str(file_id) + "_layer_code.data"
        #
        #     if os.path.exists(doc_embedding_file):
        #         os.remove(doc_embedding_file)
        #     else:
        #         print("The file does not exist: ", str(doc_embedding_file))
        #
        #     if os.path.exists(code_embedding_file):
        #         os.remove(code_embedding_file)
        #     else:
        #         print("The file does not exist: ", str(code_embedding_file))

    # print("all_mean_rk_1: ", mean(all_rk_1))
    # print("all_mean_rk_5: ", mean(all_rk_5))
    # print("all_mean_rk_10: ", mean(all_rk_10))
    # print("all_mean_mrr: ", mean(all_mrr))

