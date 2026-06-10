import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm


# ==========================================
# 1. 神经网络组件 (Neural Networks)
# ==========================================

#生成器
class GainGenerator(nn.Module):
    def __init__(self, dim, h_dim):
        super(GainGenerator, self).__init__()
        self.fc1 = nn.Linear(dim * 2, h_dim)  # Input: Data + Mask
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, m):
        # x: 输入数据 (观测值是真的，缺失值用随机噪声代替)，m: 掩码矩阵
        # Concatenate Data and Mask
        inputs = torch.cat([x, m], dim=1)
        h1 = F.relu(self.fc1(inputs))
        h2 = F.relu(self.fc2(h1))
        # MinMax normalized output [0, 1]
        return torch.sigmoid(self.fc3(h2))

#判别器
class GainDiscriminator(nn.Module):
    def __init__(self, dim, h_dim):
        super(GainDiscriminator, self).__init__()
        self.fc1 = nn.Linear(dim * 2, h_dim)  # Input: Data + Hint
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, h):
        # x: 输入数据 (生成器补全的完整数据)，h: Hint 向量，掩码矩阵随机遮盖部分观测值
        # Concatenate Data and Hint
        inputs = torch.cat([x, h], dim=1)
        h1 = F.relu(self.fc1(inputs))
        h2 = F.relu(self.fc2(h1))
        return torch.sigmoid(self.fc3(h2))  # Probability output


# ==========================================
# 2. 工具函数 (Utils)
# ==========================================

def normalization(data):
    """Min-Max 归一化，处理 NaN"""
    _min = np.nanmin(data, axis=0)
    _max = np.nanmax(data, axis=0)
    _den = _max - _min
    _den[_den == 0] = 1e-6  # 防止除以0
    norm_data = (data - _min) / _den
    norm_parameters = {'min': _min, 'max': _max, 'den': _den}
    return norm_data, norm_parameters


def normalization_with_parameter(data, norm_parameters):
    """使用已有参数进行归一化"""
    return (data - norm_parameters['min']) / norm_parameters['den']


def renormalization(norm_data, norm_parameters):
    """反归一化"""
    return norm_data * norm_parameters['den'] + norm_parameters['min']


def sample_Z(batch_size, dim):
    """生成随机噪声 Z"""
    return np.random.uniform(0., 0.01, size=[batch_size, dim])


def sample_M(batch_size, dim, p):
    """生成 Hint 向量所需的随机掩码"""
    # 生成 Hint Mask B
    # 后续用于构造 Hint:
    # H = B * M + 0.5 * (1 - B)
    # B=1 表示暴露该位置的真实 Mask 信息
    # B=0 表示隐藏该位置的信息
    unif_random_matrix = np.random.uniform(0., 1., size=[batch_size, dim])
    binary_random_matrix = 1. * (unif_random_matrix > p)
    return binary_random_matrix


# ==========================================
# 3. 核心训练流程 (Core Algorithms)
# ==========================================

def train_gain_algorithm(generator, discriminator, data_x, mask, params, device):
    """
    执行 GAIN 的标准训练循环
    Args:
        data_x: 归一化后的数据 (已将NaN填为0)
        mask: 掩码矩阵 (1=Observed, 0=Missing)
    """
    no, dim = data_x.shape

    # 优化器
    opt_g = optim.Adam(generator.parameters())
    opt_d = optim.Adam(discriminator.parameters())

    print(f"Starting GAIN training on {device}...")

    pbar = tqdm(range(params['epoch']), desc="GAIN Training")

    for it in pbar:
        # 1. Mini-batch generation
        idx = np.random.permutation(no)
        batch_idx = idx[:params['batch_size']]

        #制造缺失数据
        X_mb = data_x[batch_idx, :]
        #对应的掩码矩阵
        M_mb = mask[batch_idx, :]

        # Sample random vectors
        Z_mb = sample_Z(params['batch_size'], dim)

        # Sample hint vectors
        H_mb_temp = sample_M(params['batch_size'], dim, 1 - params['hint_rate'])
        H_mb = M_mb * H_mb_temp

        # Combine random vectors with observed vectors
        # 观测值保持不变，缺失值用随机噪声代替
        X_mb_with_noise = M_mb * X_mb + (1 - M_mb) * Z_mb

        # Convert to Torch Tensors
        X_mb_torch = torch.tensor(X_mb_with_noise, dtype=torch.float32).to(device)
        M_mb_torch = torch.tensor(M_mb, dtype=torch.float32).to(device)
        H_mb_torch = torch.tensor(H_mb, dtype=torch.float32).to(device)
        X_original_torch = torch.tensor(X_mb, dtype=torch.float32).to(device)

        # -----------------------------------
        # Train Discriminator
        # -----------------------------------
        opt_d.zero_grad()

        G_sample = generator(X_mb_torch, M_mb_torch)
        # 生成器输出的补全数据与原始数据结合，形成完整输入给判别器
        Hat_X = X_mb_torch * M_mb_torch + G_sample * (1 - M_mb_torch)
        #D_prob 是判别器对每个特征的判断概率，表示该特征是观测值（真实数据）的概率
        #接近1 → 判别器认为是真实数据
        #接近0 → 判别器认为是Generator补出来的数据
        D_prob = discriminator(Hat_X.detach(), H_mb_torch)
        # 判别器的损失函数：正确分类观测值和生成值
        D_loss = -torch.mean(M_mb_torch * torch.log(D_prob + 1e-8) + \
                             (1 - M_mb_torch) * torch.log(1. - D_prob + 1e-8))

        D_loss.backward()
        opt_d.step()

        # -----------------------------------
        # Train Generator
        # -----------------------------------
        opt_g.zero_grad()

        G_sample = generator(X_mb_torch, M_mb_torch)
        # 生成器输出的补全数据与原始数据结合，形成预测结果输入给判别器
        Hat_X = X_mb_torch * M_mb_torch + G_sample * (1 - M_mb_torch)
        D_prob = discriminator(Hat_X, H_mb_torch)
    
        #欺骗判别器的损失：生成器希望判别器认为它生成的数据是真实的，因此希望 D_prob 接近 1
        G_loss_temp = -torch.mean((1 - M_mb_torch) * torch.log(D_prob + 1e-8))

        #MSE损失：衡量生成器补全数据与原始数据的差距，防止生成器忽略生成数据的质量
        #确保把缺失值补全的像真值的同时，也要尽量保持原始数据预测的准确性
        MSE_loss = torch.mean((M_mb_torch * X_original_torch - M_mb_torch * G_sample) ** 2) / torch.mean(M_mb_torch)

        #生成器的损失函数：欺骗判别器 + MSE损失
        G_loss = G_loss_temp + params['alpha'] * MSE_loss

        G_loss.backward()
        opt_g.step()

        # .item() 用于从 tensor 中取出数值，:.4f 表示保留4位小数
        pbar.set_postfix({
            'G_Loss': f"{G_loss.item():.4f}",
            'D_Loss': f"{D_loss.item():.4f}",
            'MSE': f"{MSE_loss.item():.4f}"
        })

    print("Training Complete.")