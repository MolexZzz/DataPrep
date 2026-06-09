import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Your description here")
    # 数据集的名称
    parser.add_argument("--dataset", type=str,default="adult", help="The name of the dataset")
    # 数据的缺失类型
    parser.add_argument("--missing_mechanism", type=str, choices=["MCAR", "MAR", "MNAR"], default="MCAR", help="Type of missingness: MCAR, MAR, MNAR")
    # 数据的缺失率
    parser.add_argument("--missing_rate", type=float, default=0.1, help="Missingness rate")
    # 预测目标
    parser.add_argument("--target", type=str, default="target", help="Name of the target column")
    # 敏感属性
    parser.add_argument("--sensitive", type=str, default="sex", help="Name of the sensitive column")
    
    # 随机种子
    parser.add_argument("--seed_list", type = list, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help="Seed for random number generation")
    # 是否对数据进行填补
    parser.add_argument("--imputation_method", type=str, default="zero", help = "The imputation method")
    # 敏感属性是否缺失
    parser.add_argument("--is_missing_sensitive", action="store_true", help="Flag indicating if missingness depends on sensitive attribute")
    # 在计算MAR或者MNAR，是值越大缺失率越高还是值越小缺失率越高
    parser.add_argument("--reverse", action="store_true", help="Flag indicating if to reverse the missingness probability")
    # 是否丢弃掉数据,仅使用完整的数据进行操作
    parser.add_argument("--drop_value", action="store_true", help="Whether to drop incomplete data and only use complete data for operation. Default is True.")


    ####以下参数和模型训练有关######
    parser.add_argument("--k", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for regularization")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--is_k_cross_validation", action="store_true", help="Whether to use k-fold cross-validation. Default is False.")

    parser.add_argument('--device', type=int, default=0, help='CUDA device index (default: 0)')


    ####以下参数和模型训练结果的保存有关####
    parser.add_argument("--experiment_describe", type=str, default="None", help="The describe of the experiment")

    ####以下参数和saint的训练相关####
    parser.add_argument("--task", type=str, default="binary", help="The type of task: regression or classification")
    parser.add_argument("--attentiontype", type=str, default="col", choices=["col", "mask", "normalize", "mask_normalize"], help="The type of the attention mechanism")
    parser.add_argument("--dtask", type=str, default="clf", help="The type of the task for the discriminator, reg")
    parser.add_argument('--cont_embeddings', default='MLP', type=str, choices=['MLP', 'Noemb', 'pos_singleMLP'])
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--c', default=32, type=int)
    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--attention_heads', default=8, type=int)
    parser.add_argument('--attention_dropout', default=0.1, type=float)
    parser.add_argument('--ff_dropout', default=0.1, type=float)
    parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common', 'sep'])
    parser.add_argument('--optimizer', default='AdamW', type=str, choices=['AdamW', 'Adam', 'SGD'])
    parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'linear'])
    parser.add_argument('--log_freq', default=1, type=float)
    parser.add_argument('--evaluation_metrics', default="acc,f1,auc,dp,eopp,eodd,accp,fnr,fpr", type=str)
    parser.add_argument('--balance', action="store_true", help="Whether the dataset is balance or not")
    parser.add_argument("--current_time", default=None, type=str)
    
    parser.add_argument('--patience', type = int, default=30)
    parser.add_argument('--min_delta', type = float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_seeds', type=int, default=11, help='Number of repetitions')

    args = parser.parse_args()
    return args


