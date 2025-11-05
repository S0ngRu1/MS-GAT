import os
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from sklearn.metrics import accuracy_score , recall_score
from model.TextEncoder import TextEncoder
from data.dataloader import MMDataLoader , TextDataLoader
from utils.metrics import collect_metrics
from utils.functions import save_checkpoint, load_checkpoint, dict_to_str, count_parameters
import time
import random
import argparse
import warnings
import torch.multiprocessing



logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger('')


def xavier_init(m):
    # xavier_init 函数：这个函数用于对线性层的参数进行初始化，采用的是 Xavier 初始化方法。
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    '''定义了一个自定义的线性层 LinearLayer，并提供了一个函数 xavier_init 来初始化线性层的参数。'''
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 构建一个包含单个线性层的序列 (nn.Sequential)，该线性层将输入维度映射到输出维度。
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x



def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def get_optimizer(model, args):
    text_enc_param = list(model.module.text_encoder.named_parameters())
    text_clf_param = list(model.module.text_classfier.parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in text_enc_param if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay_tfm, 'lr': args.lr_text_tfm},
        {"params": [p for n, p in text_enc_param if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
            'lr': args.lr_text_tfm},
        {"params": text_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_text_cls},
        ]
    optimizer = optim.Adam(optimizer_grouped_parameters)
    return optimizer


def valid(args, model, data=None, best_valid=None, nBetter=None, step=None):
    '''在训练过程中定期验证模型的性能，并根据验证结果来调整模型或停止训练。'''
    model.eval()  # Set model to evaluation mode
    if best_valid is None:
        best_valid = 0.0
    with torch.no_grad():
        valid_loader = data
        y_pred = []
        y_true = []
        with tqdm(valid_loader, desc='Validation', unit='batch') as td:
            for batch in td:
                text_input_ids, text_token_type_ids, text_attention_mask, batch_label = batch
                text = (text_input_ids.to(args.device), text_token_type_ids.to(args.device), text_attention_mask.to(args.device))
                batch_label = batch_label.to(args.device)
                logit = model.module.infer(text, None)
                y_pred.append(logit.cpu())
                y_true.append(batch_label.cpu())
        logits = torch.cat(y_pred)
        te_true = torch.cat(y_true).data.cpu().numpy()
        te_prob = F.softmax(logits, dim=1).data.cpu().numpy()
        cur_valid = accuracy_score(te_true, te_prob.argmax(1))
        recall = recall_score(te_true, te_prob.argmax(1))
        # 判断当前模型是否比之前的最佳模型更好
        isBetter = cur_valid >= (best_valid + 1e-6)
        valid_results = {
        "accuracy": cur_valid,"recall": recall}
        valid_results.update(collect_metrics(args.dataset, te_true, te_prob))
        # 如果当前模型更好，则保存模型并更新 best_valid 和 nBetter
        if isBetter:
            if args.local_rank in [0, -1]:
                save_checkpoint(model, args.best_model_save_path)
            best_valid = cur_valid
            nBetter = 0
        else:
            nBetter += 1
    return valid_results, best_valid, nBetter


def train_valid(args, model, optimizer, scheduler=None, data=None):
    best_valid = 1e-5
    nBetter = 0
    total_step = 0
    gradient_accumulation_steps = 4

    for epoch in range(args.num_epoch):
        model.train()
        train_loader, valid_loader, test_loader = data
        y_pred = []
        y_true = []
        train_loss_m = 0
        if args.local_rank not in [-1]:
            train_loader.sampler.set_epoch(epoch)

        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epoch}', unit='batch') as td:
            for text_input_ids, text_token_type_ids, text_attention_mask, batch_label in td:
                text = (text_input_ids.to(args.device), text_token_type_ids.to(args.device), text_attention_mask.to(args.device))
                labels = batch_label.to(args.device).view(-1)
                loss, loss_m, logit_m = model(text, None, labels)
                loss = loss.sum()
                loss.backward()
                train_loss_m += loss_m.sum().item()
                y_pred.append(logit_m.cpu())
                y_true.append(batch_label.cpu())
                total_step += 1

                if total_step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    total_step = 0  # Reset total_step after gradient update
            
        # End of training for this epoch
        logits = torch.cat(y_pred)
        tr_true = torch.cat(y_true).data.cpu().numpy()
        tr_prob = F.softmax(logits, dim=1).data.cpu().numpy()
        train_accuracy = accuracy_score(tr_true, tr_prob.argmax(1))
        average_train_loss = train_loss_m / len(train_loader)
        logger.info(f'Epoch {epoch + 1}, Training Accuracy: {train_accuracy:.4f}, Loss: {average_train_loss:.4f}')

        # Validation after each epoch
        valid_results, best_valid, nBetter = valid(args, model, data=valid_loader, best_valid=best_valid, nBetter=nBetter)
        logger.info(f'Epoch {epoch + 1}, Validation Results: {valid_results}')
        if scheduler is not None:
            scheduler.step(train_accuracy)  
    return best_valid


def test_epoch(model, dataloader=None):
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        with tqdm(dataloader) as td:
            for text_input_ids, text_token_type_ids, text_attention_mask, batch_label in td:
                text = text_input_ids.cuda(), text_token_type_ids.cuda(), text_attention_mask.cuda()
                logit = model.module.infer(text, None)
                y_pred.append(logit.cpu())
                y_true.append(batch_label.cpu())
        logits = torch.cat(y_pred)
        true = torch.cat(y_true).data.cpu().numpy()
        prob = F.softmax(logits, dim=1).data.cpu().numpy()
    return prob, true

class Textmodel(nn.Module):
    '''定义了一个文本子网络分类器。'''
    def __init__(self, args):
        # 调用了父类 nn.Module 的构造函数 super().__init__() 来初始化父类的属性。然后，根据参数 args 中的设置，创建了不同的子网络和分类器。
        super(Textmodel, self).__init__()
        # text subnets
        self.args = args
        self.text_encoder = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder)
        self.text_classfier = Classifier(args.text_dropout, args.text_out, 2)

    def forward(self, text=None, data_list=None, label=None, infer=False):
        # 定义了交叉熵损失函数：使用了 PyTorch 中的 torch.nn.CrossEntropyLoss，
        # 设置 reduction='none'，这意味着损失函数会返回每个样本的损失而不是对它们进行平均或求和。
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        text = self.text_encoder(text=text)
        output_text = self.text_classfier(text[:, 0, :])
        if infer:
            return output_text

        Loss_m = torch.mean(criterion(output_text, label))
        Loss_sum = Loss_m
        return Loss_sum, Loss_m, output_text

    def infer(self, text=None, data_list=None):
        # 它调用了 forward 方法，并将 infer=True 作为参数传递给 forward 方法。
        # 在 forward 方法中，当 infer=True 时，只会返回多模态分类器的输出 output_mm。因此，infer 方法最终返回多模态分类器的输出 MMlogit。
        logit = self.forward(text, data_list, infer=True)
        return logit


class Classifier(nn.Module):
    def __init__(self, dropout, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.post_dropout = nn.Dropout(p=dropout)
        self.hidden_1 = LinearLayer(in_dim, 256)      
        self.hidden_2 = LinearLayer(256, 64)
        self.classify = LinearLayer(64, out_dim)

    def forward(self, input):
        # 通过第一个全连接层并应用 ReLU 激活函数，将输入特征进行线性变换和非线性变换，得到中间层的输出 input_p1。
        input_p1 = F.relu(self.hidden_1(input), inplace=False)
        # 对中间层的输出进行 dropout 操作，以防止过拟合。
        input_d = self.post_dropout(input_p1)
        # 通过第二个全连接层并应用 ReLU 激活函数，再次对中间层的输出进行线性变换和非线性变换，得到更新后的中间层的输出 input_p2。
        input_p2 = F.relu(self.hidden_2(input_d), inplace=False)
        # 通过最后一个全连接层，将中间层的输出映射到输出类别的维度，得到模型的最终输出 output。
        output = self.classify(input_p2)
        return output



def bert_MM17(args):
    train_loader, valid_loader, test_loader = TextDataLoader(args)
    data = train_loader, valid_loader, test_loader
    if args.local_rank in [-1]:
        
        model = DataParallel(Textmodel(args))
        model = model.to(args.device)
    
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    if not args.test_only:
        if args.local_rank in [-1, 0]:
            logger.info("Start training...")
        best_results = train_valid(args, model, optimizer, scheduler, data)

    load_checkpoint(model, args.best_model_save_path)

    te_prob, te_true = test_epoch(model, test_loader)
    best_results = collect_metrics(args.dataset, te_true, te_prob)
    if args.local_rank in [-1, 0]:
        logger.info("Test: " + dict_to_str(collect_metrics(args.dataset, te_true, te_prob)))

    return best_results 




torch.cuda.current_device()
torch.multiprocessing.set_sharing_strategy('file_system')
localtime = time.localtime(time.time())
str_time = f'{str(localtime.tm_mon)}-{str(localtime.tm_mday)}-{str(localtime.tm_hour)}-{str(localtime.tm_min)}'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run(args):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.fig_save_dir):
        os.makedirs(args.fig_save_dir)

    name_seed = args.name + '_' + str(args.seed)
    args.model_save_path = os.path.join(args.model_save_dir, f'{args.dataset}-{name_seed}-{str_time}.pth')
    args.best_model_save_path = os.path.join(args.model_save_dir, f'{args.dataset}-{name_seed}-best.pth')

    setup_seed(args.seed)
    if args.dataset in ['MM17']:
        results = bert_MM17(args)
        return results


def set_log(args):
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    log_file_path = os.path.join(args.logs_dir, f'{args.dataset}-{args.name}-{str_time}.log')
    # set logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # logger.setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger


if __name__ == '__main__':
    start = time.time()
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='OnlyText_bert_NoMMC', help='project name')
    parser.add_argument('--dataset', type=str, default='MM17', help='support MM17')
    parser.add_argument('--mmc', type=str, default='NoMMC', help='support UniSMMC/UnSupMMC/SupMMC/NoMMC')
    parser.add_argument('--mmc_tao', type=float, default=0.07, help='use supervised contrastive loss or not')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_epoch', type=int, default=32, help='num_epoch')
    parser.add_argument('--num_workers',type=int, default= 10, help= 'num_workers')
    parser.add_argument('--valid_step', type=int, default=50, help='valid_step')
    parser.add_argument('--text_encoder', type=str, default='bert-base-chinese', help='bert-base-chinese/roberta-base-chinese')
    parser.add_argument('--text_out', type=int, default=768, help='text_out')
    parser.add_argument('--weight_decay_tfm', type=float, default=1e-3, help='--weight_decay_tfm')
    parser.add_argument('--weight_decay_other', type=float, default=1e-2, help='--weight_decay_other')
    parser.add_argument('--lr_patience', type=float, default=2, help='--lr_patience')
    parser.add_argument('--lr_factor', type=float, default=0.2, help='--lr_factor')
    parser.add_argument('--lr_text_tfm', type=float, default=2e-5, help='--lr_text_tfm')
    parser.add_argument('--lr_text_cls', type=float, default=5e-5, help='--lr_text_cls')
    parser.add_argument('--text_dropout', type=float, default=0.0, help='--text_dropout')
    parser.add_argument('--data_dir', type=str, default='datasets', help='support ...')
    parser.add_argument('--test_only', type=bool, default=False, help='train+test or test only')
    parser.add_argument('--pretrained_dir', type=str, default='Pretrained', help='path to pretrained models from Hugging Face.')
    parser.add_argument('--model_save_dir', type=str, default='results/models', help='path to save model parameters.')
    parser.add_argument('--res_save_dir', type=str, default='results/results', help='path to save training results.')
    parser.add_argument('--fig_save_dir', type=str, default='results/imgs', help='path to save figures.')
    parser.add_argument('--logs_dir', type=str, default='results/logs', help='path to log results.')  # NO
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--seed', nargs='+', default=1, help='List of random seeds')
    args = parser.parse_args()
    logger = set_log(args)
    if args.local_rank == -1:
        device = torch.device("cuda")
    args.device = device

    if args.local_rank in [-1, 0]:
        logger.info("Pytorch version: " + torch.__version__)
        logger.info("CUDA version: " + torch.version.cuda)
        logger.info(f"CUDA device: + {torch.cuda.current_device()}")
        logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
        logger.info("GPU name: " + torch.cuda.get_device_name())
        logger.info("Current Hyper-Parameters:")
        logger.info(args)

    final_results = {}
    final_std_results = {}
    temp_results = {}

    # final_std_results
    temp_results = run(args)
    if len(final_results.keys()):
        for key in temp_results.keys():
            final_results[key] += temp_results[key]
            final_std_results[key].append(temp_results[key])
    else:
        final_results = temp_results
        final_std_results = {key: [] for key in temp_results.keys()}
        for key in temp_results.keys():
            final_std_results[key].append(temp_results[key])

    if args.local_rank in [-1, 0]:
        logger.info(f"Final test results:")
        for key in final_results.keys():
            print(key, ": ", final_std_results[key])
            final_std_results[key] = np.std(final_std_results[key])
            final_results[key] = final_results[key]
        logger.info(f"{args.dataset}-{args.name}")
        logger.info("Average: " + dict_to_str(final_results))
        logger.info("Standard deviation: " + dict_to_str(final_std_results))
    end = time.time()
    logger.info(f"Run {end - start} seconds in total！")
