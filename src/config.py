import argparse


def set_args():
    parser = argparse.ArgumentParser('--HFEDR')
    parser.add_argument('--train_data', default='../data/train_valid/ruby/train.txt', type=str, help='train_data path')
    parser.add_argument('--test_data', default='../data/train_valid/ruby/valid.txt', type=str, help='valid_data path')
    parser.add_argument('--pretrained_model_path', default='microsoft/graphcodebert-base', type=str, help='pretrained_model_path') #./roberta_pretrain
    parser.add_argument('--output_dir', default='../outputs_model', type=str, help='output_dir')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='num_train_epochs')
    parser.add_argument('--train_batch_size', default=64, type=int, help='train_batch_size')
    parser.add_argument('--val_batch_size', default=1, type=int, help='val_batch_size')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='gradient_accumulation_steps')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='learning_rate')
    parser.add_argument('--seed', default=43, type=int, help='seed')
    return parser.parse_args()
