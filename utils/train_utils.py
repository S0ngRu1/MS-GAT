
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from loguru import logger
from train import valid

def check_nan_in_inputs(inputs):
    """检查输入数据中是否存在 NaN/Inf，返回是否需要跳过该 batch"""
    for data in inputs:
        if isinstance(data, (list, tuple)):
            # 处理多元素输入（如 text 的三元组）
            for d in data:
                if torch.any(torch.isnan(d)) or torch.any(torch.isinf(d)):
                    return True
        else:
            if torch.any(torch.isnan(data)) or torch.any(torch.isinf(data)):
                return True
    return False


def compute_train_metrics(y_pred, y_true):
    """计算训练集指标（precision、recall、F1、accuracy）"""
    logits = torch.cat(y_pred)
    tr_true = torch.cat(y_true).cpu().numpy()
    tr_prob = F.softmax(logits, dim=1).cpu().numpy()
    tr_pred = tr_prob.argmax(1)
    accuracy = accuracy_score(tr_true, tr_pred)
    precision = precision_score(tr_true, tr_pred, average='macro')
    recall = recall_score(tr_true, tr_pred, average='macro')
    f1 = f1_score(tr_true, tr_pred, average='macro')
    return precision, recall, f1, logits.shape[0] 


def log_train_metrics(run, logger, epoch, batch_idx, precision, recall, f1, loss, is_epoch_end=False):
    """记录训练指标到 wandb 和 logger"""
    if is_epoch_end:
        prefix = f'Epoch {epoch + 1} 结束'
    else:
        prefix = f'Epoch {epoch + 1}, Batch {batch_idx + 1}'
    run.log({
        'train_loss': loss,
        'train_precision': precision,
        'train_recall': recall,
        'train_f1': f1
    })
    logger.info(f'{prefix}, Training precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Loss: {loss:.4f}')



def run_validation(args, run, model, valid_loader, best_valid, nBetter):
    """调用验证函数并更新最佳指标"""
    valid_results, new_best, new_nBetter = valid(args, run, model, data=valid_loader, best_valid=best_valid, nBetter=nBetter)
    logger.info(f'验证结果: {valid_results}')
    return valid_results, new_best, new_nBetter

