import os

import numpy as np
import torch
import scipy
import numpy as np
import scipy.stats
from numpy import mean
from tqdm import tqdm


def l2_normalize(vecs):
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

def load_embedding(embedding_file):
    embedding = torch.load(embedding_file)
    return embedding

def ccgir_transform_and_normalize(vecs, kernel, bias):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

def ccgir_compute_kernel_bias(vecs, n_components):

    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s**0.5))
    W = np.linalg.inv(W.T)
    W = W[:, :n_components]
    return W, -mu

N_COMPONENTS = 512
def whitening(all_doc_vecs, all_code_vecs):

    all_doc_code = []
    all_doc_code.extend(all_doc_vecs)
    all_doc_code.extend(all_code_vecs)

    all_doc_code = np.array(all_doc_code)

    kernel, bias = ccgir_compute_kernel_bias([
        all_doc_code
    ], n_components=N_COMPONENTS)

    all_doc_code = ccgir_transform_and_normalize(all_doc_code, kernel, bias)  # [code_list_size, dim]

    return all_doc_code

def write_infer_result(s):
    path = os.path.join("/result", 'mrr_result.txt')

    with open(path, 'a+') as f:
        f.write('\n')
        f.write(str(s))

def similary(all_a_vecs, all_b_vecs, batch = 1000):
    assert (len(all_a_vecs) % batch == 0)

    cos_list = []

    # whitening
    all_doc_code = whitening(all_a_vecs, all_b_vecs)

    len_all = len(all_doc_code)
    half = int(len_all / 2)

    all_a_vecs = all_doc_code[:half]
    all_b_vecs = all_doc_code[half:]

    all_result = []

    all_a_vecs = np.array(all_a_vecs)
    all_b_vecs = np.array(all_b_vecs)

    j = 0
    for i in range(0, len(all_a_vecs), batch):
        batch_a_vecs = all_a_vecs[i: i + batch]
        batch_b_vecs = all_b_vecs[i: i + batch]

        assert (len(batch_a_vecs) == batch)
        a_vecs = l2_normalize(batch_a_vecs)
        b_vecs = l2_normalize(batch_b_vecs)

        sim_scores = (a_vecs * b_vecs).sum(axis=1)
        # print(sim_scores)
        cos_list.append(sim_scores)


    return cos_list



def get_mrr(batch_id):

    all_sim_list = []

    layer_num = 6 #A1
    select_layers = [0,1,2,3,4,5]

    progress_bar = tqdm(range(len(select_layers)))
    for i in range(layer_num):
        if (i in select_layers):
            doc_embedding_file = "../embedding/batch_i/" + str(i) + "_layer_doc.data"
            code_embedding_file = "../embedding/batch_i/" + str(i) + "_layer_code.data"

            doc_l12 = load_embedding(doc_embedding_file)
            code_l12 = load_embedding(code_embedding_file)

            temp_sim_list = similary(doc_l12, code_l12)
            all_sim_list.append(temp_sim_list)
            #shape: 2 * 1000 * 1000

            progress_bar.update(1)

    #Similarity Calculation
    all_sim_list = np.array(all_sim_list)
    stack_sim_list = sum(all_sim_list)

    all_result = []

    j = 0
    for i in range(1000):

        sim_scores = stack_sim_list[i]
        correct_score = sim_scores[j]
        j += 1

        rank = np.sum(sim_scores >= correct_score)
        all_result.append(rank)

    rk_1 = np.sum(np.array(all_result) == 1) / len(all_result)
    rk_5 = np.sum(np.array(all_result) <= 5) / len(all_result)
    rk_10 = np.sum(np.array(all_result) <= 10) / len(all_result)

    mean_mrr = np.mean(1.0 / np.array(all_result))

    print("mean_mrr: ", mean_mrr)
    print("all_rk_1: ", rk_1)
    print("all_rk_5: ", rk_5)
    print("all_rk_10: ", rk_10)

    return mean_mrr, rk_1, rk_5, rk_10

if __name__ == '__main__':
    all_mrr = []
    all_rk_1 = []
    all_rk_5 = []
    all_rk_10 = []

    for i in [0]:  # batch N  The number of the test files
        batch_i_mean_mrr, rk_1, rk_5, rk_10 = get_mrr(i)
        all_mrr.append(batch_i_mean_mrr)
        all_rk_1.append(rk_1)
        all_rk_5.append(rk_5)
        all_rk_10.append(rk_10)

        print("all_mean_mrr now: ", mean(all_mrr))
        print("all_mean_rk_1 now: ", mean(all_rk_1))
        print("all_mean_rk_5 now: ", mean(all_rk_5))
        print("all_mean_rk_10 now: ", mean(all_rk_10))

    # delete files for save storage
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