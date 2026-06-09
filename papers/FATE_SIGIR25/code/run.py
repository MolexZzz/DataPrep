import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
import time

import pickle
from super_con import SupConLoss

import sys
sys.path.append("../Imputation_Method")

from MIWAE import MIWAE
from notmiwae import notMIWAE
from GAIN import gain_main
# from NOMI.nomi import nomi_main
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from torch import einsum
from einops import rearrange
import json
import torch.optim as optim
from datetime import datetime
import csv
from tabulate import tabulate

from dataloader import data_prep_concise, DataSetCatCon
from model import FATE

from utils import fair_classification_score
from utils import print_metrics
from utils import get_scheduler
from utils import EarlyStopper
from args import parse_arguments
from metrics import metric_evaluation
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

def embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model):
    device = x_cont.device
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    n1, n2 = x_cont.shape
    _, n3 = x_categ.shape

    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1,n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:,i,:] = model.simple_MLP[i](x_cont[:,i])
    else:
        raise Exception('This case should not work!')
    
    x_cont_enc = x_cont_enc.to(device)
    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)

    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    return x_categ_enc, x_cont_enc

def get_rep(x_categ, x_cont, cat_mask, con_mask, model, training = False):
    x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)
    reps = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, training = training)
    return reps


def fair_classification_score(model, dloader, attentiontype, device, prefix="test", task = 'binary', dataset = None, missing_rate = None, missing_mechanism = None):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    s_all = torch.empty(0).to(device)

    target_hat_list = []
    target_list = []
    sensitive_list = []
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, s, y_gts, cat_mask, con_mask, s_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5].to(device), data[6].to(device)
            cat_mask_mask = torch.ones_like(cat_mask)
            con_mask_mask = torch.ones_like(con_mask)
            x_cont = torch.cat([x_cont, cat_mask, con_mask], dim = 1)
            con_mask = torch.cat([con_mask, cat_mask_mask, con_mask_mask], dim = 1)

            x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)

            s_all = torch.cat([s_all,s.squeeze().float()],dim=0)

            reps = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask)

            
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)

            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)

            if task == 'binary' or task == 'multiclass':
                prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
            
            target_hat_list.append(torch.argmax(y_outs, dim=1).float().cpu().numpy())
            target_list.append(y_gts.squeeze().float().cpu().numpy())
            sensitive_list.append(s.squeeze().float().cpu().numpy())

    target_hat_list = np.concatenate(target_hat_list, axis=0)
    target_list = np.concatenate(target_list, axis=0)
    sensitive_list = np.concatenate(sensitive_list, axis=0)

    metric = metric_evaluation(y_gt=target_list, y_pre=target_hat_list, s=sensitive_list, prefix=f"{prefix}")

    return metric


def write_results(file_name, dataset, model, imputation_method, missing_rate,
                  missing_mechanism, running_time_mean, running_time_std, metrics, attentiontype, sensitive, target):
    
    # 获取当前时间
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 判断文件是否存在
    file_exists = os.path.isfile(file_name)

    # 如果文件不存在，创建文件并写入表头
    if not file_exists:
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Dataset", "Model", "Imputation Method", "Missing Rate", "Missing Mechanism", "Sensitive", "Target",
                             'AUC Mean', 'AUC Std', 'ACC Mean', 'ACC Std', 'F1 Mean', 'F1 Std', 
                             'DP Mean', 'DP Std', 'EOPP Mean', 'EOPP Std', 'ACCP Mean', 'ACCP Std', 
                             'EODD Mean', 'EODD Std', 'FNR Mean', 'FNR Std', 'FPR Mean', 'FPR Std',
                             'Running Time Mean', 'Running Time Std', "Attention Type", "Execution Time"])

    # 追加模式下写入数据
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([dataset, model, imputation_method, missing_rate, missing_mechanism, sensitive, target,
                         np.mean(metrics["auc"]), np.std(metrics["auc"]), np.mean(metrics["acc"]), np.std(metrics["acc"]),
                         np.mean(metrics["f1"]), np.std(metrics["f1"]), np.mean(metrics["dp"]), np.std(metrics["dp"]),
                         np.mean(metrics["eopp"]), np.std(metrics["eopp"]), np.mean(metrics["accp"]), np.std(metrics["accp"]),
                         np.mean(metrics["eodd"]), np.std(metrics["eodd"]), np.mean(metrics["fnr"]), np.std(metrics["fnr"]),
                         np.mean(metrics["fpr"]), np.std(metrics["fpr"]), running_time_mean, running_time_std, attentiontype, current_time])



def run_FATE(args):
    seeds = range(1, args.num_seeds)
    running_time_list = []
    all_res = {
            "auc": [],
            "acc": [],
            "f1": [],
            "dp": [],
            "eopp": [],
            "accp": [],
            "eodd": [],
            "fnr": [],
            "fpr": [],
    }
    for seed in seeds:
        time_start = time.time()
        print(f"============Current seed is {seed}===========")
        torch.manual_seed(seed)
        args.seed = seed

        # 二分类任务
        args.task = "binary"
        device = torch.device(f"cuda:{args.device}")

        cat_dims, cat_idxs, con_idxs, X_train, s_train, y_train, X_valid, s_valid, y_valid, X_test, s_test, y_test, train_mean, train_std = data_prep_concise(args)

        continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

        # Setting some hyperparams based on inputs and dataset
        _,nfeat = X_train['data'].shape
        if nfeat > 100:
            args.embedding_size = min(8, args.embedding_size)
            args.batch_size = min(64, args.batch_size)
        if args.attentiontype == 'col':
            args.transformer_depth = 1
            args.attention_heads = min(4, args.attention_heads)
            args.attention_dropout = 0.8
            args.embedding_size = min(32, args.embedding_size)
            args.ff_dropout = 0.8

        train_ds = DataSetCatCon(X_train, s_train, y_train, cat_idxs, con_idxs, args.dtask, continuous_mean_std)
        trainloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last = True)

        valid_ds = DataSetCatCon(X_valid, s_valid, y_valid, cat_idxs, con_idxs, args.dtask, continuous_mean_std)
        validloader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last = True)

        test_ds = DataSetCatCon(X_test, s_test, y_test, cat_idxs, con_idxs, args.dtask, continuous_mean_std)
        testloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last = True)

        # 分类任务
        y_dim= len(np.unique(y_train['data'][:, 0]))

        # cls token
        cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int) # Appending 1 for CLS token, this is later used to generate embeddings.

        model = FATE(
            categories = tuple(cat_dims),
            num_continuous = len(con_idxs) * 2 + len(cat_dims),
            dim = args.embedding_size,
            dim_out = 1,
            depth = args.transformer_depth,
            heads = args.attention_heads,
            attn_dropout = args.attention_dropout,
            ff_dropout = args.ff_dropout,
            mlp_hidden_mults = (4, 2),
            cont_embeddings = args.cont_embeddings,
            attentiontype = args.attentiontype,
            final_mlp_style = args.final_mlp_style,
            y_dim = y_dim,
            batch_size = args.batch_size,
            device = device
        )

        # loss function
        if y_dim == 2 and args.task == 'binary':
            criterion = nn.CrossEntropyLoss(reduction='none').to(device)
        else:
            raise 'case not written yet'
        
        fair_criterion = SupConLoss()
        model.to(device)

        #argsimizer
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = get_scheduler(args, optimizer)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        print('===============Training begins now.==================')
        train_loss_ls = []
        valid_loss_ls = []
        valid_result = {
            "auc": np.array([]),
            "acc": np.array([]),
            "f1": np.array([]),
            "dp": np.array([]),
            "eopp": np.array([]),
            "accp": np.array([]),
            "eodd": np.array([]),
            "fnr": np.array([]),
            "fpr": np.array([]),
        }
        test_result = {
            "auc": np.array([]),
            "acc": np.array([]),
            "f1": np.array([]),
            "dp": np.array([]),
            "eopp": np.array([]),
            "accp": np.array([]),
            "eodd": np.array([]),
            "fnr": np.array([]),
            "fpr": np.array([]),
        }

        # Early stopping的策略
        early_stopper = EarlyStopper(args.patience, args.min_delta)
        stop_epoch = args.num_epochs

        # Train Stage
        for epoch in range(args.num_epochs):
            model.train()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                optimizer.zero_grad()
                x_categ, x_cont, s, y_gts, cat_mask, con_mask, s_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5].to(device), data[6].to(device)

                #Change: 将mask矩阵当成连续变量输入到模型当中
                cat_mask_mask = torch.ones_like(cat_mask)
                con_mask_mask = torch.ones_like(con_mask)
                x_cont = torch.cat([x_cont, cat_mask, con_mask], dim = 1)
                con_mask = torch.cat([con_mask, cat_mask_mask, con_mask_mask], dim = 1)
                # 获得表征
                reps = get_rep(x_categ, x_cont, cat_mask, con_mask, model, training = True)

                # 直接计算ERM
                y_reps = reps[:, 0, :]
                y_outs = model.mlpfory(y_reps)
                y_gts = y_gts.squeeze()

                if type(y_outs) == tuple:
                    y_outs = torch.tensor(y_outs, dtype=torch.float32)
                
                # 分类任务
                y_gts = y_gts.to(torch.long)
                loss = torch.mean(criterion(y_outs, y_gts))
                # fair_loss = fair_criterion(y_outs, y_gts)

                loss.backward()
                optimizer.step()
                if args.optimizer == 'SGD':
                    scheduler.step()
                running_loss += loss.item()
            
            # ================validation=====================
            model.eval()
            valid_loss = 0.0
            logs = []
            headers = ["Step(Tr|Val|Te)"] + args.evaluation_metrics.split(",")
            for i, data in enumerate(validloader, 0):
                x_categ, x_cont, s, y_gts, cat_mask, con_mask, s_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5].to(device), data[6].to(device)

                cat_mask_mask = torch.ones_like(cat_mask)
                con_mask_mask = torch.ones_like(con_mask)
                x_cont = torch.cat([x_cont, cat_mask, con_mask], dim = 1)
                con_mask = torch.cat([con_mask, cat_mask_mask, con_mask_mask], dim = 1)

                # 获得表征
                reps = get_rep(x_categ, x_cont, cat_mask, con_mask, model)
                # reps, value = get_rep(x_categ, x_cont, cat_mask, con_mask, model)
                # 直接计算ERM
                y_reps = reps[:, 0, :]
                y_outs = model.mlpfory(y_reps)
                y_gts = y_gts.squeeze()

                if type(y_outs) == tuple:
                    y_outs = torch.tensor(y_outs, dtype=torch.float32)
                
                # 分类任务
                y_gts = y_gts.to(torch.long)
                loss = torch.mean(criterion(y_outs, y_gts))

                valid_loss += loss.item()
            
            train_loss_ls.append(running_loss)
            valid_loss_ls.append(valid_loss)
            
            if epoch % args.log_freq == 0 or epoch == 1 or epoch == args.num_epochs:
                model.eval()
                with torch.no_grad():
                    if args.task in ['binary']:
                        train_metrics = fair_classification_score(model, trainloader, attentiontype = args.attentiontype,device = device, prefix = "train", task = "binary", dataset = args.dataset, missing_rate = args.missing_rate, missing_mechanism = args.missing_mechanism)

                        test_metrics = fair_classification_score(model, testloader, attentiontype = args.attentiontype, device = device, prefix = "test", task = "binary",dataset = args.dataset, missing_rate = args.missing_rate, missing_mechanism = args.missing_mechanism)

                        valid_metrics = fair_classification_score(model, validloader, attentiontype = args.attentiontype, device = device, prefix = "val", task = "binary",dataset = args.dataset, missing_rate = args.missing_rate, missing_mechanism = args.missing_mechanism)

                        res_dict = {}
                        res_dict.update(train_metrics)
                        res_dict.update(valid_metrics)
                        res_dict.update(test_metrics)

                        # for printing
                        if epoch % (args.log_freq*10) == 0:
                            res = print_metrics(res_dict, args.evaluation_metrics, train=True)
                            logs.append([epoch, *res])
                            # if epoch > 3:
                            #     clear_lines(len(logs)*2 + 1)
                            table = tabulate(logs, headers=headers, tablefmt="grid", floatfmt="06.4f")
                            print(table)

                        valid_result['acc'] = np.append(valid_result['acc'], valid_metrics['val/acc'])
                        valid_result['auc'] = np.append(valid_result['auc'], valid_metrics['val/auc'])
                        valid_result['f1'] = np.append(valid_result['f1'], valid_metrics['val/f1'])
                        valid_result['dp'] = np.append(valid_result['dp'], valid_metrics['val/dp'])
                        valid_result['eopp'] = np.append(valid_result['eopp'], valid_metrics['val/eopp'])
                        valid_result['eodd'] = np.append(valid_result['eodd'], valid_metrics['val/eodd'])
                        valid_result['accp'] = np.append(valid_result['accp'], valid_metrics['val/accp'])
                        valid_result['fnr'] = np.append(valid_result['fnr'], valid_metrics['val/fnr'])
                        valid_result['fpr'] = np.append(valid_result['fpr'], valid_metrics['val/fpr'])

                        test_result['acc'] = np.append(test_result['acc'], test_metrics['test/acc'])
                        test_result['auc'] = np.append(test_result['auc'], test_metrics['test/auc'])
                        test_result['f1'] = np.append(test_result['f1'], test_metrics['test/f1'])
                        test_result['dp'] = np.append(test_result['dp'], test_metrics['test/dp'])
                        test_result['eopp'] = np.append(test_result['eopp'], test_metrics['test/eopp'])
                        test_result['eodd'] = np.append(test_result['eodd'], test_metrics['test/eodd'])
                        test_result['accp'] = np.append(test_result['accp'], test_metrics['test/accp'])
                        test_result['fnr'] = np.append(test_result['fnr'], test_metrics['test/fnr'])
                        test_result['fpr'] = np.append(test_result['fpr'], test_metrics['test/fpr'])

            # early_stopper的策略
            if early_stopper.early_stop(valid_loss):
                print(f"Early stopping triggered in epoch {epoch}! Training stopped.")
                stop_epoch = epoch + 1
                break
                
            model.train()

        # draw_plot(stop_epoch, train_loss_ls, valid_loss_ls, xlabel='epoch', ylabel="cross_entropy", xlim=[1, args.num_epochs], legend=['train', 'valid'], dataset = args.dataset, missing_mechanism = args.missing_mechanism, missing_rate = args.missing_rate, target = args.target, sensitive = args.sensitive, imputation_method = args.imputation_method)

        time_end = time.time()
        time_c = time_end - time_start
        running_time_list.append(time_c)
        # 调用valid上最好的模型进行测试
        best_model_idx = np.argmax(valid_result['acc'])
        best_test_accuracy = test_result['acc'][best_model_idx]
        best_test_auc = test_result['auc'][best_model_idx]
        best_test_f1 = test_result['f1'][best_model_idx]
        best_test_dp = test_result['dp'][best_model_idx]
        best_test_eopp = test_result['eopp'][best_model_idx]
        best_test_eodd = test_result['eodd'][best_model_idx]
        best_test_accp = test_result["accp"][best_model_idx]
        best_test_fnr = test_result["fnr"][best_model_idx]
        best_test_fpr = test_result["fpr"][best_model_idx]

        print("In the best valid model, the test accuracy is %.4f, the test auc is %.4f, the test f1 score is %.4f, the test dp is %.4f, the test eopp is %.4f, the test eodd is %.4f \n" % (best_test_accuracy, best_test_auc, best_test_f1, best_test_dp, best_test_eopp, best_test_eodd))

        all_res['acc'].append(best_test_accuracy)
        all_res['auc'].append(best_test_auc)
        all_res['f1'].append(best_test_f1)
        all_res['dp'].append(best_test_dp)
        all_res['eopp'].append(best_test_eopp)
        all_res['eodd'].append(best_test_eodd)
        all_res["accp"].append(best_test_accp)
        all_res["fnr"].append(best_test_fnr)
        all_res["fpr"].append(best_test_fpr)
    
    running_time_mean = np.mean(running_time_list)
    running_time_std = np.std(running_time_list)

    current_time = datetime.now().strftime('%m%d')
    write_results(file_name = f"../Result/{current_time}.csv", 
                    dataset = args.dataset, 
                    model = "FATE", 
                    imputation_method = args.imputation_method, 
                    missing_rate = args.missing_rate, 
                    missing_mechanism = args.missing_mechanism, 
                    running_time_mean = running_time_mean, 
                    running_time_std = running_time_std, 
                    metrics = all_res,
                    attentiontype = args.attentiontype,
                    sensitive = args.sensitive,
                    target = args.target)
    return 0


if __name__ == "__main__":
    args = parse_arguments()
    run_FATE(args)