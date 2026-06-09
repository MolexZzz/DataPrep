import numpy as np
import pandas as pd
import os
import yaml
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from datetime import datetime
import torch
import torch.nn as nn
import csv
import sys
from numpy import random
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from metrics import metric_evaluation


def Handling_Discrete_Variables(data):
    mapping_dict = {}

    categorical_indicator = data.dtypes == 'category'
    cat_cols = data.columns[categorical_indicator].tolist()


    nan_flag = data.isnull().values.any()

    if nan_flag == True:
        mask = data.isnull()
        data = data.fillna('missing_value')

    cat_dims = []
    for col in cat_cols:
        if col not in mapping_dict:
            unique_values = data[col].unique()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            mapping_dict[col] = mapping

        data[col] = data[col].map(mapping_dict[col])
  
        data[col] = data[col].astype('float32')
        cat_dims.append(len(mapping_dict[col]))

    if nan_flag == True:
        data[mask] = np.nan
    return data


def draw_plot(num_epochs, train_ls, valid_ls, xlabel='epoch', ylabel='cross_entropy', xlim = [], legend=['train', 'valid'], dataset=None, missing_mechanism = None, missing_rate = None, target = None, sensitive = None, imputation_method = None):
    plt.clf()
    if train_ls != None:
        train_ls_num = []
        for tmp in train_ls:
            train_ls_num.append(tmp)
        plt.plot(range(1, num_epochs + 1), train_ls_num, label='train')
    
    if valid_ls != None:
        valid_ls_num = []
        for tmp in valid_ls:
            valid_ls_num.append(tmp)
        plt.plot(range(1, num_epochs + 1), valid_ls_num, label='valid')


    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if valid_ls == None:
        plt.title('Training Loss')
    else:
        plt.title('Training and Validation Losses')

 
    plt.xlim(xlim)
    plt.legend(legend)
    if valid_ls != None:
        plt.savefig(fname=f"./fig/{dataset}_{missing_mechanism}_{missing_rate}_{target}_{sensitive}_{imputation_method}_train_valid.png")
    else:
        plt.savefig(fname=f"./fig/{dataset}_{missing_mechanism}_{missing_rate}_{target}_{sensitive}_{imputation_method}train.png")


def draw_classification_result(y, y_hat, s, file_path = "../Result/fig/bar/test"):
    plt.clf()

    group1_indices = [i for i in range(len(s)) if s[i] == 0]
    group2_indices = [i for i in range(len(s)) if s[i] == 1]


    TP_group1 = sum((y[i] == 1) and (y_hat[i] == 1) for i in group1_indices)
    TN_group1 = sum((y[i] == 0) and (y_hat[i] == 0) for i in group1_indices)
    FP_group1 = sum((y[i] == 0) and (y_hat[i] == 1) for i in group1_indices)
    FN_group1 = sum((y[i] == 1) and (y_hat[i] == 0) for i in group1_indices)

    TP_group2 = sum((y[i] == 1) and (y_hat[i] == 1) for i in group2_indices)
    TN_group2 = sum((y[i] == 0) and (y_hat[i] == 0) for i in group2_indices)
    FP_group2 = sum((y[i] == 0) and (y_hat[i] == 1) for i in group2_indices)
    FN_group2 = sum((y[i] == 1) and (y_hat[i] == 0) for i in group2_indices)

  
    labels = ['TP', 'TN', 'FP', 'FN']
    index = range(len(labels))
    bar_width = 0.35

    plt.bar([i - bar_width/2 for i in index], [TP_group1, TN_group1, FP_group1, FN_group1], bar_width, label='Sensitive Group 0')
    plt.bar([i + bar_width/2 for i in index], [TP_group2, TN_group2, FP_group2, FN_group2], bar_width, label='Sensitive Group 1')

    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Counts of TP, TN, FP, FN for Two Groups')
    plt.xticks(index, labels)
    plt.legend()

    for i in index:
        plt.text(i - bar_width/2, [TP_group1, TN_group1, FP_group1, FN_group1][i] + 0.02, str([TP_group1, TN_group1, FP_group1, FN_group1][i]), ha='center')
        plt.text(i + bar_width/2, [TP_group2, TN_group2, FP_group2, FN_group2][i] + 0.02, str([TP_group2, TN_group2, FP_group2, FN_group2][i]), ha='center')

    plt.savefig("{}.png".format(file_path))
    plt.show()


def write_to_csv(metrics, path, dataset, missing_mechanism, missing_rate, target, sensitive, imputation_method, experiment_describe):

    current_time = datetime.now()
    metrics["Dataset"] = dataset
    metrics["Time"] = current_time.strftime("%Y-%m-%d %H:%M:%S")
    metrics["Missing_mechanism"] = missing_mechanism
    metrics["Missing_rate"] = missing_rate
    metrics["Target"] = target
    metrics["Sensitive"] = sensitive
    metrics["Imputation_method"] = imputation_method
    metrics["Experiment_describe"] = experiment_describe
    df_new = pd.DataFrame(metrics, index=[0])
    if os.path.exists(path):

        folder_size = os.path.getsize(path)

        if folder_size != 0:
            df_existing = pd.read_csv(path)

 
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:

            df_combined = df_new
    else:
        df_combined = df_new
    
    df_combined.to_csv(path, index=False)
    print("Finish write to csv!")

def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                      milestones=[args.num_epochs // 2.667, args.num_epochs // 1.6, args.num_epochs // 1.142], gamma=0.1)
    return scheduler

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

    draw_classification_result(target_list, target_hat_list, sensitive_list, file_path = f"../Result/fig/bar/{dataset}_{missing_rate}_{missing_mechanism}_{prefix}")

    metric = metric_evaluation(y_gt=target_list, y_pre=target_hat_list, s=sensitive_list, prefix=f"{prefix}")

    return metric

def write_result_dict_to_csv(results_dict, file_name = 'evaluation_results.csv' , dataset = None, sensitive_attribute = None, target = None, missing_rate = None, missing_mechanism = None, method = "saint_fair", imputation = None, balance = None, reverse = None, epoch = 0):
    metric_names = set(key.split('/')[1] for key in results_dict.keys())
    if os.path.exists(file_name):
        with open(file_name, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            header_name = next(csv_reader, None)  
    else:
        header_name = []

    values = []

    if header_name == []:
        header_name.extend(["epoch", "dataset", "sensitive_attribute", "target", "missing_rate", "misisng_mechanism", "method", "imputation", "balance", "reverse"])
        values.extend([epoch, dataset, sensitive_attribute, target, missing_rate, missing_mechanism, method, imputation,balance, reverse])

        for name in metric_names:
            for stage in ["train", "val", "test"]:
                column_name = f'{stage}/{name}'
                if column_name not in header_name:
                    header_name.append(column_name)

                values.append(results_dict[column_name])
    else:
        for column_name in header_name:
            if column_name in ["epoch", "lam", "dataset", "sensitive_attribute", "target", "missing_rate", "missing_mechanism", "method", "imputation", "balance", "reverse"]:
                if column_name == "epoch":
                    values.append(epoch)
                elif column_name == "dataset":
                    values.append(dataset)
                elif column_name == "sensitive_attribute":
                    values.append(sensitive_attribute)
                elif column_name == "target":
                    values.append(target)
                elif column_name == "missing_rate":
                    values.append(missing_rate)
                elif column_name == "missing_mechanism":
                    values.append(missing_mechanism)
                elif column_name == "method":
                    values.append(method)
                elif column_name == "imputation":
                    values.append(imputation)
                elif column_name == "balance":
                    values.append(balance)
                elif column_name == "reverse":
                    values.append(reverse)
            elif column_name not in results_dict:
                values.append("")
            else:
                values.append(results_dict[column_name])

    if not os.path.exists(file_name):
        with open(file_name, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header_name)
            csv_writer.writerow(values)
    else:
        with open(file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(values)


def clear_lines(lines):
    for _ in range(lines):
        sys.stdout.write("\033[K")  # Clear to the end of line
        sys.stdout.write("\033[F")  # Move cursor up one line


def print_metrics(metrics_dict, metrics_print = "ap,dp", train=True):

    output = []
    for metric in metrics_print.split(","):
        if train == True:
            output.append( "{:0>6.4f}|{:0>6.4f}|{:0>6.4f}".format(metrics_dict["train/"+metric], metrics_dict["val/"+metric], metrics_dict["test/"+metric]) )
        else:
            output.append( "{:0>6.4f}|{:0>6.4f}".format(metrics_dict["val/"+metric], metrics_dict["test/"+metric]) )
    
    return tuple(output)

def use_svg_display():
    backend_inline.set_matplotlib_formats('svg')

#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)


class EarlyStopper:
    def __init__(self, patience = 1, min_delta = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
    
    # return True when validation loss is not decreased by the 'min_delta' for 'patience' epochs
    def early_stop(self, validation_loss):
        if ((validation_loss + self.min_delta) < self.min_validation_loss):
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif ((validation_loss + self.min_delta) > self.min_validation_loss):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def seed_everything(seed=2024):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class PandasDataSet(TensorDataset):
    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame("dummy")
        return torch.from_numpy(df.values).float()


def InfiniteDataLoader(dataset, batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True):
    while True:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        for data in data_loader:
            yield data