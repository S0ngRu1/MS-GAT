from tqdm import tqdm
import torch
from utils.train_utils import check_nan_in_inputs,compute_train_metrics,log_train_metrics
from loguru import logger as sys_logger

def train_fnd2(args, model, optimizer, data, run, logger):
    best_valid = 1e-5
    nBetter = 0
    train_loader, valid_loader, _ = data
    for epoch in range(args.num_epoch):
        model.train()
        y_pred, y_true = [], []
        train_loss_m = 0.0
        
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epoch}', unit='batch') as td:
            for batch_idx, (batch_image, fft_imge, sentiment_output, text_input_ids, text_token_type_ids, text_attention_mask, batch_label) in enumerate(td):
                # 检查输入 NaN
                inputs = [batch_image, fft_imge, text_input_ids, sentiment_output]
                if check_nan_in_inputs(inputs):
                    sys_logger.info("输入数据含 NaN，跳过该 batch...")
                    continue
                
                # 数据移至设备
                text = (text_input_ids.to(args.device), 
                        text_token_type_ids.to(args.device), 
                        text_attention_mask.to(args.device))
                sentiment_output = sentiment_output.to(args.device)
                image = batch_image.to(args.device)
                fft_imge = fft_imge.to(args.device)
                labels = batch_label.to(args.device).view(-1)
                
                # 模型前向传播
                loss, loss_m, logit_m = model(text, sentiment_output, image, fft_imge, labels)
                loss = loss.sum()
                
                # 检查损失 NaN
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    sys_logger.info("损失含 NaN，跳过该 batch...")
                    continue
                
                # 反向传播与参数更新
                loss.backward()
                train_loss_m += loss_m.sum().item()
                y_pred.append(logit_m.cpu())
                y_true.append(batch_label.cpu())
                optimizer.step()
                optimizer.zero_grad()
                
                # 每 2 个 batch 记录一次指标
                if (batch_idx + 1) % 2 == 0:
                    precision, recall, f1, _ = compute_train_metrics(y_pred, y_true)
                    avg_loss = train_loss_m / (batch_idx + 1)
                    log_train_metrics(run, logger, epoch, batch_idx, precision, recall, f1, avg_loss)
                
                # 每 300 个 batch 验证一次
                if (batch_idx + 1) % 300 == 0:
                    _, best_valid, nBetter = run_validation(args, run, model, valid_loader, best_valid, nBetter)
        
        # Epoch 结束时记录全局指标
        precision, recall, f1, _ = compute_train_metrics(y_pred, y_true)
        avg_loss = train_loss_m / len(train_loader)
        log_train_metrics(run, logger, epoch, batch_idx, precision, recall, f1, avg_loss, is_epoch_end=True)
        
        # Epoch 结束后验证
        _, best_valid, nBetter = run_validation(args, run, model, valid_loader, best_valid, nBetter)
    
    return best_valid


def train_fnd2_clip(args, model, optimizer, data, run, logger):
    best_valid = 1e-5
    nBetter = 0
    train_loader, valid_loader, _ = data
    for epoch in range(args.num_epoch):
        model.train()
        y_pred, y_true = [], []
        train_loss_m = 0.0
        
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epoch}', unit='batch') as td:
            for batch_idx, (batch_image, fft_imge, sentiment_output, batch_text, batch_label) in enumerate(td):
                # 检查输入 NaN
                inputs = [batch_image, fft_imge, batch_text, sentiment_output]
                if check_nan_in_inputs(inputs):
                    sys_logger.info("输入数据含 NaN，跳过该 batch...")
                    continue
                
                # 数据移至设备
                text = batch_text.to(args.device)
                sentiment_output = sentiment_output.to(args.device)
                image = batch_image.to(args.device)
                fft_imge = fft_imge.to(args.device)
                labels = batch_label.to(args.device).view(-1)
                
                # 模型前向传播
                loss, loss_m, logit_m = model(text, sentiment_output, image, fft_imge, labels)
                loss = loss.sum()
                
                # 检查损失 NaN
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    sys_logger.info("损失含 NaN，跳过该 batch...")
                    continue
                
                # 反向传播与参数更新
                loss.backward()
                train_loss_m += loss_m.sum().item()
                y_pred.append(logit_m.cpu())
                y_true.append(batch_label.cpu())
                optimizer.step()
                optimizer.zero_grad()
                
                # 每 2 个 batch 记录一次指标
                if (batch_idx + 1) % 2 == 0:
                    precision, recall, f1, _ = compute_train_metrics(y_pred, y_true)
                    avg_loss = train_loss_m / (batch_idx + 1)
                    log_train_metrics(run, logger, epoch, batch_idx, precision, recall, f1, avg_loss)
        
        # Epoch 结束时记录全局指标
        precision, recall, f1, _ = compute_train_metrics(y_pred, y_true)
        avg_loss = train_loss_m / len(train_loader)
        log_train_metrics(run, logger, epoch, batch_idx, precision, recall, f1, avg_loss, is_epoch_end=True)
        
        # Epoch 结束后验证
        _, best_valid, nBetter = run_validation(args, run, model, valid_loader, best_valid, nBetter)
    
    return best_valid


def train_fnd2_sgat(args, model, optimizer, data, run, logger):
    best_valid = 1e-5
    nBetter = 0
    train_loader, valid_loader, _ = data
    for epoch in range(args.num_epoch):
        model.train()
        y_pred, y_true = [], []
        train_loss_m = 0.0
        
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epoch}', unit='batch') as td:
            for batch_idx, (batch_image, fft_imge, sentiment_output, text_input_ids, text_token_type_ids, text_attention_mask, graph_data, batch_label) in enumerate(td):
                # 检查输入 NaN
                inputs = [batch_image, fft_imge, text_input_ids, sentiment_output]
                if check_nan_in_inputs(inputs):
                    sys_logger.info("输入数据含 NaN，跳过该 batch...")
                    continue
                
                # 数据移至设备
                text = (text_input_ids.to(args.device), 
                        text_token_type_ids.to(args.device), 
                        text_attention_mask.to(args.device))
                sentiment_output = sentiment_output.to(args.device)
                image = batch_image.to(args.device)
                fft_imge = fft_imge.to(args.device)
                labels = batch_label.to(args.device).view(-1)
                graph_data = graph_data.to(args.device)
                
                # 模型前向传播（含 graph_data）
                loss, loss_m, logit_m = model(text, sentiment_output, image, fft_imge, labels, graph_data=graph_data)
                loss = loss.sum()
                
                # 检查损失 NaN
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    sys_logger.info("损失含 NaN，跳过该 batch...")
                    continue
                
                # 反向传播与参数更新
                loss.backward()
                train_loss_m += loss_m.sum().item()
                y_pred.append(logit_m.cpu())
                y_true.append(batch_label.cpu())
                optimizer.step()
                optimizer.zero_grad()
                
                # 每 2 个 batch 记录一次指标
                if (batch_idx + 1) % 2 == 0:
                    precision, recall, f1, _ = compute_train_metrics(y_pred, y_true)
                    avg_loss = train_loss_m / (batch_idx + 1)
                    log_train_metrics(run, logger, epoch, batch_idx, precision, recall, f1, avg_loss)
        
        # Epoch 结束时记录全局指标
        precision, recall, f1, _ = compute_train_metrics(y_pred, y_true)
        avg_loss = train_loss_m / len(train_loader)
        log_train_metrics(run, logger, epoch, batch_idx, precision, recall, f1, avg_loss, is_epoch_end=True)
        
        # Epoch 结束后验证
        _, best_valid, nBetter = run_validation(args, run, model, valid_loader, best_valid, nBetter)
    
    return best_valid

