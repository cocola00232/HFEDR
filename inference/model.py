import torch
from torch import nn
from transformers import BertConfig, BertModel, AutoModel


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.config = BertConfig.from_pretrained('./roberta_pretrain/config.json')
        self.bert = AutoModel.from_pretrained("microsoft/graphcodebert-base")

    def forward(self, input_ids, attention_mask, encoder_type='cls'):

        output = self.bert(input_ids, attention_mask, output_hidden_states=True)

        if encoder_type == 'fist-last-avg':
            # 第一层和最后一层的隐层取出  然后经过平均池化
            first = output.hidden_states[1]   # hidden_states列表有13个hidden_state，第一个其实是embeddings，第二个元素才是第一层的hidden_state
            last = output.hidden_states[-1]
            seq_length = first.size(1)   # 序列长度

            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            final_encoding = torch.avg_pool1d(torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2), kernel_size=2).squeeze(-1)
            return final_encoding

        if encoder_type == 'last-avg':
            sequence_output = output.last_hidden_state  # (batch_size, max_len, hidden_size)
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding

        if encoder_type == "cls":
            sequence_output = output.last_hidden_state
            cls = sequence_output[:, 0]  # [b,d]
            return cls

        if encoder_type == "pooler":
            pooler_output = output.pooler_output  # [b,d]
            return pooler_output

        if encoder_type == "all": #train专用
            all_hidden_states = output.hidden_states
            # (all_hidden_states[-1][:,0,:] == output.last_hidden_state[:,0,:])
            # print(all_hidden_states[-1][:,0,:]) #
            # print(output.last_hidden_state[:,0,:])#

            assert (len(all_hidden_states) == 13) #13,前面有个embedding层，这个层不要
            all_hidden_states = all_hidden_states[1:]  # 截取12层

            return all_hidden_states #没有处理cls

        # 配套train encoder_type == "all"的inference
        if encoder_type == "all_stack": #12层的cls位置加起来，向做inference用，看会不会提高
            all_hidden_states = output.hidden_states
            all_hidden_states = all_hidden_states[1:]  # 截取12层

            stack_hidden_states = all_hidden_states[1] + all_hidden_states[2] + all_hidden_states[3]\
                                  + all_hidden_states[4] + all_hidden_states[5] + all_hidden_states[6]\
                                  + all_hidden_states[7] + all_hidden_states[8] + all_hidden_states[9]\
                                  + all_hidden_states[10] + all_hidden_states[11] + all_hidden_states[0]

            return stack_hidden_states[:,0,:] #截取cls的

        # 配套train encoder_type == "all"的inference
        if encoder_type == "all_cat": #inference专用，把所有层的hidden state, cat起来

            all_hidden_states = output.hidden_states
            all_hidden_states = all_hidden_states[2:]  # 截取11层,第0层在cat_hidden_states初始化时候用

            cat_hidden_states = all_hidden_states[0][:,0,:] #提前截取cls的
            for hidden_state in all_hidden_states:
                cat_hidden_states = torch.cat((cat_hidden_states, hidden_state[:,0,:]), 1) #截取cls的

            # print(cat_hidden_states.shape) # batch_size * (12*768=9216) #后续要关注如何降低维度的问题

            return cat_hidden_states

        if encoder_type == "n_layer": #inference专用，把指定层的hidden state, cat起来

            all_hidden_states = output.hidden_states
            all_hidden_states = all_hidden_states[2:]  # 截取11层,第0层在cat_hidden_states初始化时候用

            cat_hidden_states = all_hidden_states[0][:,0,:] #提前截取cls的

            # 选择用哪些层计算,不过这里all_hidden_states已经是11层了，
            # 所以需要变换，第0层默认在里面，所以去除第一个0，其余元素全部减去1
            index = 0
            # index_layers = [0,9,10] #实际是[1,2,11,12]层
            index_layers = [0,1, 8, 9, 10]  # 实际是[1,2,3,10,11,12]层 best

            for hidden_state in all_hidden_states:
                if (index in index_layers):
                    cat_hidden_states = torch.cat((cat_hidden_states, hidden_state[:,0,:]), 1) #截取cls的
                index += 1

            # print(cat_hidden_states.shape) # batch_size * (7*768=5376) #后续要关注如何降低维度的问题

            return cat_hidden_states

        if encoder_type == "iter_n_layer": #多个layer train的model，在inference时，使用这个，会逐一取出指定层的嵌入，返回的size时batch_size * 768
            all_hidden_states = output.hidden_states
            all_hidden_states = all_hidden_states[1:]  # 截取12层

            return all_hidden_states  # 没有截取cls的  #相当于把所有层都返回出去，然后在外面处理怎么选，注意内存爆炸
            #return shape: 12 * batch_size * maxlen * 768
            #一次全部return出去好一些，不然要进若干次这里来抽嵌入
