import os
import logging
import numpy as np
from tqdm import tqdm
import torch
import wandb
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score , recall_score, confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
from model.model import MyModel 
from data.graph_mutil_modal_dataloader import graph_data_loader
from data.image_dataloader import image_data_loader
from data.text_dataloader import text_dataloader
from data.mutil_modal_dataloader import mm_data_loader
from utils.metrics import collect_metrics
from utils.functions import save_checkpoint, load_checkpoint, dict_to_str
import matplotlib.pyplot as plt
from loguru import logger


def valid(args, run, model, data=None, best_valid=None, nBetter=None, step=None):
    if args.method in ['FND-2']:
        model.eval()
        if best_valid is None:
            best_valid = 0.0

        with torch.no_grad():
            valid_loader = data
            y_pred = []
            y_true = []
            with tqdm(valid_loader, desc='Validation', unit='batch') as td:
                for batch_image, fft_imge, sentiment_output, text_input_ids, text_token_type_ids, text_attention_mask, batch_label in td:
                    text = (text_input_ids.to(args.device), 
                            text_token_type_ids.to(args.device), 
                            text_attention_mask.to(args.device))
                    sentiment_output = sentiment_output.to(args.device)
                    image = (batch_image.to(args.device))
                    fft_imge = (fft_imge.to(args.device))
                    batch_label = batch_label.to(args.device)
                    logit = model.module.infer(text, sentiment_output, image, fft_imge)
                    y_pred.append(logit.cpu())
                    y_true.append(batch_label.cpu())
            logits = torch.cat(y_pred)
            tr_true = torch.cat(y_true).data.cpu().numpy()
            tr_prob = F.softmax(logits, dim=1).data.cpu().numpy()
            val_precision, val_recall, val_f1 = precision_score(tr_true, tr_prob.argmax(1), average='macro'), recall_score(tr_true, tr_prob.argmax(1), average='macro'), f1_score(tr_true, tr_prob.argmax(1), average='macro')
            run.log({
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1
            })
            logger.info(f'Validation Results: Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
            isBetter = val_f1 >= (best_valid + 1e-6)
            valid_results = {
                "precision": val_precision,
                "recall": val_recall,
                "f1": val_f1
            }
            valid_results.update(collect_metrics(args.dataset, tr_true, tr_prob))
            if isBetter:
                save_checkpoint(model, args.best_model_save_path)
                best_valid = val_f1
                nBetter = 0
            else:
                nBetter += 1

        return valid_results, best_valid, nBetter
    elif args.method in ['FND-2-CLIP']:
        model.eval()
        if best_valid is None:
            best_valid = 0.0

        with torch.no_grad():
            valid_loader = data
            y_pred = []
            y_true = []
            with tqdm(valid_loader, desc='Validation', unit='batch') as td:
                for batch_image, fft_imge, sentiment_output, batch_text, batch_label in td:
                    text = batch_text.to(args.device) 
                    sentiment_output = sentiment_output.to(args.device)
                    image = (batch_image.to(args.device))
                    fft_imge = (fft_imge.to(args.device))
                    batch_label = batch_label.to(args.device)
                    logit = model.module.infer(text, sentiment_output, image, fft_imge)
                    y_pred.append(logit.cpu())
                    y_true.append(batch_label.cpu())
            logits = torch.cat(y_pred)
            tr_true = torch.cat(y_true).data.cpu().numpy()
            tr_prob = F.softmax(logits, dim=1).data.cpu().numpy()
            val_precision, val_recall, val_f1 = precision_score(tr_true, tr_prob.argmax(1), average='macro'), recall_score(tr_true, tr_prob.argmax(1), average='macro'), f1_score(tr_true, tr_prob.argmax(1), average='macro')
            run.log({
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1
            })
            logger.info(f'Validation Results: Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
            isBetter = val_f1 >= (best_valid + 1e-6)
            valid_results = {
                "precision": val_precision,
                "recall": val_recall,
                "f1": val_f1
            }
            valid_results.update(collect_metrics(args.dataset, tr_true, tr_prob))
            if isBetter:
                save_checkpoint(model, args.best_model_save_path)
                best_valid = val_f1
                nBetter = 0
            else:
                nBetter += 1

        return valid_results, best_valid, nBetter
    
    elif args.method in ['FND-2-SGAT']:
        model.eval()
        if best_valid is None:
            best_valid = 0.0

        with torch.no_grad():
            valid_loader = data
            y_pred = []
            y_true = []
            with tqdm(valid_loader, desc='Validation', unit='batch') as td:
                for batch_image, fft_imge, sentiment_output, text_input_ids, text_token_type_ids, text_attention_mask, graph_data, batch_label in td:
                    text = (text_input_ids.to(args.device), 
                            text_token_type_ids.to(args.device), 
                            text_attention_mask.to(args.device))
                    sentiment_output = sentiment_output.to(args.device)
                    image = (batch_image.to(args.device))
                    fft_imge = (fft_imge.to(args.device))
                    batch_label = batch_label.to(args.device)
                    graph_data = graph_data.to(args.device)
                    logit = model.module.infer(text, sentiment_output, image, fft_imge, graph_data = graph_data)
                    y_pred.append(logit.cpu())
                    y_true.append(batch_label.cpu())
            logits = torch.cat(y_pred)
            tr_true = torch.cat(y_true).data.cpu().numpy()
            tr_prob = F.softmax(logits, dim=1).data.cpu().numpy()
            val_precision, val_recall, val_f1 = precision_score(tr_true, tr_prob.argmax(1), average='macro'), recall_score(tr_true, tr_prob.argmax(1), average='macro'), f1_score(tr_true, tr_prob.argmax(1), average='macro')
            run.log({
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1
            })
            logger.info(f'Validation Results: Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
            isBetter = val_f1 >= (best_valid + 1e-6)
            valid_results = {
                "precision": val_precision,
                "recall": val_recall,
                "f1": val_f1
            }
            valid_results.update(collect_metrics(args.dataset, tr_true, tr_prob))
            if isBetter:
                save_checkpoint(model, args.best_model_save_path)
                best_valid = val_f1
                nBetter = 0
            else:
                nBetter += 1

        return valid_results, best_valid, nBetter
    
    elif args.method in ['BERT']:
        model.eval()  
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
                    logit = model.module.infer(text, None,None)
                    y_pred.append(logit.cpu())
                    y_true.append(batch_label.cpu())
            logits = torch.cat(y_pred)
            te_true = torch.cat(y_true).data.cpu().numpy()
            te_prob = F.softmax(logits, dim=1).data.cpu().numpy()
            cur_valid = accuracy_score(te_true, te_prob.argmax(1))
            recall = recall_score(te_true, te_prob.argmax(1))
            isBetter = cur_valid >= (best_valid + 1e-6)
            valid_results = {
            "accuracy": cur_valid,"recall": recall}
            valid_results.update(collect_metrics(args.dataset, te_true, te_prob))
            if isBetter:
                save_checkpoint(model, args.best_model_save_path)
                best_valid = cur_valid
                nBetter = 0
            else:
                nBetter += 1
        return valid_results, best_valid, nBetter
    
    elif args.method in ['ViT']:
        model.eval()  
        if best_valid is None:
            best_valid = 0.0
        with torch.no_grad():
            valid_loader = data
            y_pred = []
            y_true = []
            with tqdm(valid_loader, desc='Validation', unit='batch') as td:
                for batch in td:
                    batch_image,  batch_label = batch
                    image = batch_image.to(args.device)
                    batch_label = batch_label.to(args.device)
                    logit = model.module.infer(None, image, None)
                    y_pred.append(logit.cpu())
                    y_true.append(batch_label.cpu())
            logits = torch.cat(y_pred)
            te_true = torch.cat(y_true).data.cpu().numpy()
            te_prob = F.softmax(logits, dim=1).data.cpu().numpy()
            cur_valid = accuracy_score(te_true, te_prob.argmax(1))
            recall = recall_score(te_true, te_prob.argmax(1))
            isBetter = cur_valid >= (best_valid + 1e-6)
            valid_results = {
            "accuracy": cur_valid,"recall": recall}
            valid_results.update(collect_metrics(args.dataset, te_true, te_prob))
            if isBetter:
                save_checkpoint(model, args.best_model_save_path)
                best_valid = cur_valid
                nBetter = 0
            else:
                nBetter += 1
        return valid_results, best_valid, nBetter


def train_valid(args, model, optimizer, scheduler=None, data=None):
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity= "AI-links",
        # Set the wandb project where this run will be logged.
        project = "FND",
        name = args.model_name,
        # Track hyperparameters and run metadata.
        config={
        "learning_rate": args.lr_mm,
        "dataset": args.dataset,
        "epochs": args.num_epoch,
        "batch_size": args.batch_size,
    },
    )
    
    if args.method in ['FND-2']:
        best_valid = 1e-5
        nBetter = 0
        for epoch in range(args.num_epoch):
            model.train()
            train_loader, valid_loader, _ = data
            y_pred = []
            y_true = []
            train_loss_m = 0
            with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epoch}', unit='batch') as td:
                for batch_idx, (batch_image, fft_imge, sentiment_output, text_input_ids, text_token_type_ids, text_attention_mask, batch_label) in enumerate(td):
                    
                    if torch.any(torch.isnan(batch_image)) or torch.any(torch.isnan(fft_imge)) or torch.any(torch.isnan(text_input_ids)) or torch.any(torch.isnan(sentiment_output)):
                        logging.info("NaN detected in input data, skipping batch...")
                        continue  # Skip this batch and continue training

                    text = (text_input_ids.to(args.device), 
                            text_token_type_ids.to(args.device), 
                            text_attention_mask.to(args.device))
                    sentiment_output = sentiment_output.to(args.device)
                    image = (batch_image.to(args.device))
                    fft_imge = (fft_imge.to(args.device))
                    labels = batch_label.to(args.device).view(-1)
                    loss, loss_m, logit_m = model(text, sentiment_output, image,fft_imge, labels)
                    loss = loss.sum()
                    # Check for NaN
                    if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                        logging.info(f"NaN detected in loss, skipping batch...")
                        continue  # Skip this batch and continue training
                    loss.backward()
                    train_loss_m += loss_m.sum().item()
                    y_pred.append(logit_m.cpu())
                    y_true.append(batch_label.cpu())
                    optimizer.step()
                    optimizer.zero_grad()
                    if (batch_idx + 1) % 2 == 0:
                        logits = torch.cat(y_pred)
                        tr_true = torch.cat(y_true).data.cpu().numpy()
                        tr_prob = F.softmax(logits, dim=1).data.cpu().numpy()
                        train_precision = precision_score(tr_true, tr_prob.argmax(1), average='macro')
                        train_recall = recall_score(tr_true, tr_prob.argmax(1), average='macro')
                        train_f1 = f1_score(tr_true, tr_prob.argmax(1), average='macro')
                        average_train_loss = train_loss_m / (batch_idx + 1)
                        run.log({
                            'train_loss': average_train_loss,
                            'train_precision': train_precision,
                            'train_recall': train_recall,
                            'train_f1': train_f1
                        })
                        logger.info(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Training precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Loss: {average_train_loss:.4f}')
                    if (batch_idx + 1) % 300 == 0:
                    # 验证集评估
                        valid_results, best_valid, nBetter = valid(args, run, model, data=valid_loader, best_valid=best_valid, nBetter=nBetter)
                        logger.info(f'Epoch {epoch + 1}, Validation Results: {valid_results}')
            # Epoch 结束时记录一次全局指标
            logits = torch.cat(y_pred)
            tr_true = torch.cat(y_true).data.cpu().numpy()
            tr_prob = F.softmax(logits, dim=1).data.cpu().numpy()
            train_precision, train_recall, train_f1 = precision_score(tr_true, tr_prob.argmax(1), average='macro'), recall_score(tr_true, tr_prob.argmax(1), average='macro'), f1_score(tr_true, tr_prob.argmax(1), average='macro')
            average_train_loss = train_loss_m / len(train_loader)
            run.log({
                'train_loss': average_train_loss,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1
            })
            logger.info(f'Epoch {epoch + 1}, Training precision: {train_precision:.4f}, Loss: {average_train_loss:.4f}')

            # 验证集评估
            valid_results, best_valid, nBetter = valid(args, run, model, data=valid_loader, best_valid=best_valid, nBetter=nBetter)
            logger.info(f'Epoch {epoch + 1}, Validation Results: {valid_results}')
            if scheduler is not None:
                scheduler.step(train_f1)

        return best_valid
    elif args.method in ['FND-2-CLIP']:
        best_valid = 1e-5
        nBetter = 0
        for epoch in range(args.num_epoch):
            model.train()
            train_loader, valid_loader, _ = data
            y_pred = []
            y_true = []
            train_loss_m = 0
            with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epoch}', unit='batch') as td:
                for batch_idx, (batch_image, fft_imge, sentiment_output, batch_text, batch_label) in enumerate(td):
                    
                    if torch.any(torch.isnan(batch_image)) or torch.any(torch.isnan(fft_imge)) or torch.any(torch.isnan(batch_text)) or torch.any(torch.isnan(sentiment_output)):
                        logging.info("NaN detected in input data, skipping batch...")
                        continue  # Skip this batch and continue training

                    text = batch_text.to(args.device)
                    sentiment_output = sentiment_output.to(args.device)
                    image = (batch_image.to(args.device))
                    fft_imge = (fft_imge.to(args.device))
                    labels = batch_label.to(args.device).view(-1)
                    loss, loss_m, logit_m = model(text, sentiment_output, image,fft_imge, labels)
                    loss = loss.sum()
                    # Check for NaN
                    if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                        logging.info(f"NaN detected in loss, skipping batch...")
                        continue  # Skip this batch and continue training
                    loss.backward()
                    train_loss_m += loss_m.sum().item()
                    y_pred.append(logit_m.cpu())
                    y_true.append(batch_label.cpu())
                    optimizer.step()
                    optimizer.zero_grad()
                    if (batch_idx + 1) % 2 == 0:
                        logits = torch.cat(y_pred)
                        tr_true = torch.cat(y_true).data.cpu().numpy()
                        tr_prob = F.softmax(logits, dim=1).data.cpu().numpy()
                        train_precision = precision_score(tr_true, tr_prob.argmax(1), average='macro')
                        train_recall = recall_score(tr_true, tr_prob.argmax(1), average='macro')
                        train_f1 = f1_score(tr_true, tr_prob.argmax(1), average='macro')
                        average_train_loss = train_loss_m / (batch_idx + 1)
                        run.log({
                            'train_loss': average_train_loss,
                            'train_precision': train_precision,
                            'train_recall': train_recall,
                            'train_f1': train_f1
                        })
                        logger.info(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Training precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Loss: {average_train_loss:.4f}')
                    # if (batch_idx + 1) % 300 == 0:
                    # # 验证集评估
                    #     valid_results, best_valid, nBetter = valid(args, run, model, data=valid_loader, best_valid=best_valid, nBetter=nBetter)
                    #     logger.info(f'Epoch {epoch + 1}, Validation Results: {valid_results}')
            # Epoch 结束时记录一次全局指标
            logits = torch.cat(y_pred)
            tr_true = torch.cat(y_true).data.cpu().numpy()
            tr_prob = F.softmax(logits, dim=1).data.cpu().numpy()
            train_precision, train_recall, train_f1 = precision_score(tr_true, tr_prob.argmax(1), average='macro'), recall_score(tr_true, tr_prob.argmax(1), average='macro'), f1_score(tr_true, tr_prob.argmax(1), average='macro')
            average_train_loss = train_loss_m / len(train_loader)
            run.log({
                'train_loss': average_train_loss,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1
            })
            logger.info(f'Epoch {epoch + 1}, Training precision: {train_precision:.4f}, Loss: {average_train_loss:.4f}')

            # 验证集评估
            valid_results, best_valid, nBetter = valid(args, run, model, data=valid_loader, best_valid=best_valid, nBetter=nBetter)
            logger.info(f'Epoch {epoch + 1}, Validation Results: {valid_results}')
            if scheduler is not None:
                scheduler.step(train_f1)

        return best_valid
    
    elif args.method in ['FND-2-SGAT']:
        best_valid = 1e-5
        nBetter = 0
        for epoch in range(args.num_epoch):
            model.train()
            train_loader, valid_loader, _ = data
            y_pred = []
            y_true = []
            train_loss_m = 0
            with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epoch}', unit='batch') as td:
                for batch_idx, (batch_image, fft_imge, sentiment_output, text_input_ids, text_token_type_ids, text_attention_mask, graph_data, batch_label) in enumerate(td):
                    
                    if torch.any(torch.isnan(batch_image)) or torch.any(torch.isnan(fft_imge)) or torch.any(torch.isnan(text_input_ids)) or torch.any(torch.isnan(sentiment_output)):
                        logging.info("NaN detected in input data, skipping batch...")
                        continue  # Skip this batch and continue training

                    text = (text_input_ids.to(args.device), 
                            text_token_type_ids.to(args.device), 
                            text_attention_mask.to(args.device))
                    sentiment_output = sentiment_output.to(args.device)
                    image = (batch_image.to(args.device))
                    fft_imge = (fft_imge.to(args.device))
                    labels = batch_label.to(args.device).view(-1)
                    graph_data = graph_data.to(args.device)
                    loss, loss_m, logit_m = model(text, sentiment_output, image,fft_imge, labels, graph_data = graph_data)
                    loss = loss.sum()
                    # Check for NaN
                    if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                        logging.info(f"NaN detected in loss, skipping batch...")
                        continue  # Skip this batch and continue training
                    loss.backward()
                    train_loss_m += loss_m.sum().item()
                    y_pred.append(logit_m.cpu())
                    y_true.append(batch_label.cpu())
                    optimizer.step()
                    optimizer.zero_grad()
                    if (batch_idx + 1) % 2 == 0:
                        logits = torch.cat(y_pred)
                        tr_true = torch.cat(y_true).data.cpu().numpy()
                        tr_prob = F.softmax(logits, dim=1).data.cpu().numpy()
                        train_precision = precision_score(tr_true, tr_prob.argmax(1), average='macro')
                        train_recall = recall_score(tr_true, tr_prob.argmax(1), average='macro')
                        train_f1 = f1_score(tr_true, tr_prob.argmax(1), average='macro')
                        average_train_loss = train_loss_m / (batch_idx + 1)
                        run.log({
                            'train_loss': average_train_loss,
                            'train_precision': train_precision,
                            'train_recall': train_recall,
                            'train_f1': train_f1
                        })
                        logger.info(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Training precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Loss: {average_train_loss:.4f}')
                    # if (batch_idx + 1) % 300 == 0:
                    # # 验证集评估
                    #     valid_results, best_valid, nBetter = valid(args, run, model, data=valid_loader, best_valid=best_valid, nBetter=nBetter)
                    #     logger.info(f'Epoch {epoch + 1}, Validation Results: {valid_results}')
            # Epoch 结束时记录一次全局指标
            logits = torch.cat(y_pred)
            tr_true = torch.cat(y_true).data.cpu().numpy()
            tr_prob = F.softmax(logits, dim=1).data.cpu().numpy()
            train_precision, train_recall, train_f1 = precision_score(tr_true, tr_prob.argmax(1), average='macro'), recall_score(tr_true, tr_prob.argmax(1), average='macro'), f1_score(tr_true, tr_prob.argmax(1), average='macro')
            average_train_loss = train_loss_m / len(train_loader)
            run.log({
                'train_loss': average_train_loss,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1
            })
            logger.info(f'Epoch {epoch + 1}, Training precision: {train_precision:.4f}, Loss: {average_train_loss:.4f}')

            # 验证集评估
            valid_results, best_valid, nBetter = valid(args, run, model, data=valid_loader, best_valid=best_valid, nBetter=nBetter)
            logger.info(f'Epoch {epoch + 1}, Validation Results: {valid_results}')
            if scheduler is not None:
                scheduler.step(train_f1)

        return best_valid
    
    elif args.method in ['BERT']:
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
            with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epoch}', unit='batch') as td:
                for batch_idx, (batch_image, fft_imge, sentiment_output, text_input_ids, text_token_type_ids, text_attention_mask, batch_label) in enumerate(td):
                    text = (text_input_ids.to(args.device), text_token_type_ids.to(args.device), text_attention_mask.to(args.device))
                    labels = batch_label.to(args.device).view(-1)
                    loss, loss_m, logit_m = model(text, None, None, labels)
                    loss = loss.sum()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    train_loss_m += loss_m.sum().item()
                    y_pred.append(logit_m.cpu())
                    y_true.append(batch_label.cpu())
                    total_step += 1

                    if (batch_idx + 1) % 30 == 0:
                        logits = torch.cat(y_pred)
                        tr_true = torch.cat(y_true).data.cpu().numpy()
                        tr_prob = F.softmax(logits, dim=1).data.cpu().numpy()
                        train_precision = precision_score(tr_true, tr_prob.argmax(1), average='macro')
                        train_recall = recall_score(tr_true, tr_prob.argmax(1), average='macro')
                        train_f1 = f1_score(tr_true, tr_prob.argmax(1), average='macro')
                        average_train_loss = train_loss_m / (batch_idx + 1)
                        run.log({
                            'train_loss': average_train_loss,
                            'train_precision': train_precision,
                            'train_recall': train_recall,
                            'train_f1': train_f1
                        })
                        logger.info(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Training precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Loss: {average_train_loss:.4f}')
                    
                    # 梯度累积
                    if total_step % gradient_accumulation_steps == 0:
                        
                        optimizer.step()
                        optimizer.zero_grad()
                        total_step = 0  
            # Epoch 结束时记录一次全局指标
            logits = torch.cat(y_pred)
            tr_true = torch.cat(y_true).data.cpu().numpy()
            tr_prob = F.softmax(logits, dim=1).data.cpu().numpy()
            train_precision, train_recall, train_f1 = precision_score(tr_true, tr_prob.argmax(1), average='macro'), recall_score(tr_true, tr_prob.argmax(1), average='macro'), f1_score(tr_true, tr_prob.argmax(1), average='macro')
            average_train_loss = train_loss_m / len(train_loader)
            run.log({
                'train_loss': average_train_loss,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1
            })
            logger.info(f'Epoch {epoch + 1}, Training precision: {train_precision:.4f}, Loss: {average_train_loss:.4f}')

            # 验证集评估
            valid_results, best_valid, nBetter = valid(args, run, model, data=valid_loader, best_valid=best_valid, nBetter=nBetter)
            logger.info(f'Epoch {epoch + 1}, Validation Results: {valid_results}')
            if scheduler is not None:
                scheduler.step(train_precision)

        return best_valid
    
    elif args.method in ['ViT']:
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
            with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epoch}', unit='batch') as td:
                for batch_image,  batch_label in td:
                    image = batch_image.to(args.device)
                    labels = batch_label.to(args.device).view(-1)
                    loss, loss_m, logit_m = model(None, image, None, labels)
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
            train_precision = accuracy_score(tr_true, tr_prob.argmax(1))
            average_train_loss = train_loss_m / len(train_loader)
            logger.info(f'Epoch {epoch + 1}, Training Accuracy: {train_precision:.4f}, Loss: {average_train_loss:.4f}')
            # Validation after each epoch
            valid_results, best_valid, nBetter = valid(args, model, data=valid_loader, best_valid=best_valid, nBetter=nBetter)
            logger.info(f'Epoch {epoch + 1}, Validation Results: {valid_results}')
            if scheduler is not None:
                scheduler.step(train_precision)  
        return best_valid


    
def test_epoch(args, model, dataloader=None):
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        if args.method in ['FND-2-CLIP']:
            with tqdm(dataloader) as td:
                for batch_image, fft_imge, sentiment_output, batch_text, batch_label in td:
                    if not fft_imge:
                        continue
                    # 数据转移到 GPU
                    text = batch_text.to(args.device)
                    sentiment_output = sentiment_output.to(args.device)
                    image = (batch_image.to(args.device))
                    fft_imge = (fft_imge.to(args.device))
                    batch_label = batch_label.to(args.device)
                    if args.method not in ['BERT', 'ViT']:
                        logit = model.module.infer(text, sentiment_output, image, fft_imge)
                    elif args.method == 'BERT':
                        logit = model.module.infer(text, None, None)
                    elif args.method == 'ViT':
                        logit = model.module.infer(None, image, None)

                    y_pred.append(logit.cpu())
                    y_true.append(batch_label.cpu())
        elif args.method in ['FND-2-SGAT']:
            with tqdm(dataloader) as td:
                for batch_image, fft_imge, sentiment_output, text_input_ids, text_token_type_ids, text_attention_mask, graph_data, batch_label in td:
                    # 数据转移到 GPU
                    text = (text_input_ids.to(args.device), 
                            text_token_type_ids.to(args.device), 
                            text_attention_mask.to(args.device))
                    sentiment_output = sentiment_output.to(args.device)
                    image = (batch_image.to(args.device))
                    fft_imge = (fft_imge.to(args.device))
                    batch_label = batch_label.to(args.device)
                    graph_data = graph_data.to(args.device)
                    logit = model.module.infer(text, sentiment_output, image, fft_imge, graph_data = graph_data)
                    y_pred.append(logit.cpu())
                    y_true.append(batch_label.cpu())
        logits = torch.cat(y_pred)
        true = torch.cat(y_true).data.cpu().numpy()
        prob = F.softmax(logits, dim=1).data.cpu().numpy()
    # 可视化分析
    try:
        visualize_results(args,true, prob, args.method, logits)
    except:
        logger.info("可视化出错")
        return prob, true

    return prob, true

def visualize_results(args, true_labels, predictions, method, logits):
    save_dir = os.path.join(args.res_save_dir, args.name)
    os.makedirs(save_dir, exist_ok=True)
        # T-SNE: 使用 logits
    tsne_logits = TSNE(n_components=2, perplexity=15, n_iter=1000, random_state=42)
    logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    features_logits_2d = tsne_logits.fit_transform(logits)

    plt.figure(figsize=(10, 7))
    colors = ['orange' if label == 0 else 'blue' for label in true_labels]
    scatter1 = plt.scatter(features_logits_2d[:, 0], features_logits_2d[:, 1], c=colors, s=5, alpha=0.7)
    plt.legend(*scatter1.legend_elements(), title="Classes", loc="best")
    plt.title(f'T-SNE (Logits) - {method}')
    plt.xlabel('TSNE-1')
    plt.ylabel('TSNE-2')
    plt.savefig(os.path.join(save_dir, 'tsne_logits.png'))
    plt.close()


def train(args):
    if args.method in ['BERT']:
        train_loader, valid_loader, test_loader = text_dataloader(args)
    elif args.method in ['ViT']:
        train_loader, valid_loader, test_loader = image_data_loader(args) 
    elif args.method in ['FND-2-SGAT']:
        train_loader, valid_loader, test_loader = graph_data_loader(args) 
    else:
        train_loader, valid_loader, test_loader = mm_data_loader(args)
    data = train_loader, valid_loader, test_loader
    model = MyModel(args).to(args.device)

    optimizer = optim.Adam(model.parameters(),  lr=args.lr, weight_decay=args.weight_decay)

    if args.train_test:
        logger.info("Start training...")
        train_valid(args, model, optimizer, data)
        load_checkpoint(model, args.best_model_save_path)
        te_prob, te_true = test_epoch(args,model, test_loader)
        logger.info("Test: " + dict_to_str(collect_metrics(args.dataset, te_true, te_prob)))

