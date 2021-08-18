import argparse
from main import *

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--base_path', default="./dataset", type=str)
parser.add_argument('--dataset', default='NYT', type=str, help='specify the dataset from ["NYT","WebNLG","NYT_simple","WebNLG_simple"')
parser.add_argument('--bert_vocab_path', default="./pretrained/bert-base-cased/vocab.txt", type=str)
parser.add_argument('--bert_config_path', default="./pretrained/bert-base-cased/config.json", type=str)
parser.add_argument('--bert_model_path', default="./pretrained/bert-base-cased/pytorch_model.bin", type=str)
parser.add_argument('--stop_epoch', default=10, type=int)
parser.add_argument('--train', action="store_true")
parser.add_argument('--better_CE', default=True, type=bool, help='Use loss function based on confidence threshold')
parser.add_argument('--conf_value', default=0.1, type=float, help='confidence threshold')
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--max_len', default=100, type=int)
parser.add_argument('--warmup', default=0.0, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--max_grad_norm', default=1.0, type=float)
parser.add_argument('--num_train_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=18, type=int)
parser.add_argument('--min_num', default=1e-7, type=float)
args = parser.parse_args()


if args.train:
    train(args)
else:
    test(args)
