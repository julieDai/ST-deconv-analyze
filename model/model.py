import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
import numpy as np
from torchviz import make_dot
from ..utils.utils import get_unique_filename, get_newly_filename, get_newly_filename_index
import os
def mmd_loss(X, Y):
    """
    计算MK-MMD距离作为损失函数
    Args:
    - X: 第一个分布的样本数据
    - Y: 第二个分布的样本数据
    Returns:
    - mmd: MK-MMD距离
    """
    sigma = 1.0  # 高斯核函数的带宽参数
    n = X.size(0)
    m = Y.size(0)

    # 高斯核函数
    def kernel(x, y):
        return torch.exp(-torch.sum((x - y)**2) / (2 * sigma**2))

    # 计算样本之间的核函数值
    xx = torch.mean(torch.stack([kernel(x, x) for x in X]))
    yy = torch.mean(torch.stack([kernel(y, y) for y in Y]))
    xy = torch.mean(torch.stack([kernel(x, y) for x in X for y in Y]))

    # 计算MK-MMD距离
    mmd = xx - 2 * xy + yy
    
    return mmd

def dynamic_contrastive_loss(embeddings, positions, threshold_pos, threshold_neg, threshold_radius):
    """
    根据动态选择的正负样本对计算损失。
    """
    num_samples = embeddings.shape[0]
    loss = 0.0
    for i in range(num_samples):
        # 计算距离
        distances = torch.norm(embeddings - embeddings[i], dim=1)
        position_distances = torch.norm(positions - positions[i], dim=1)
        
        # 动态选择正负样本
        positive_mask = (position_distances < threshold_pos) & (position_distances > 0)
        negative_mask = (position_distances > threshold_neg) & (position_distances < threshold_radius)
        
        if positive_mask.any():
            positive_loss = -torch.log(
                F.softmax(-distances[positive_mask], dim=0)
            ).mean()
        else:
            positive_loss = 0
        
        if negative_mask.any():
            negative_loss = -torch.log(
                F.softmax(distances[negative_mask], dim=0)
            ).mean()
        else:
            negative_loss = 0
        
        loss += positive_loss + negative_loss
    
    return (loss / num_samples)*0.01
# 定义对比学习损失函数
def nt_xent_loss(encoded_closest, encoded_farthest, encoded_target_sample, temperature=0.1):
    """
    计算NT-Xent损失（Normalized Temperature-scaled Cross Entropy Loss）。

    参数:
    - encoded_closest: 编码后的最近邻样本特征。
    - encoded_farthest: 编码后的最远邻样本特征。
    - encoded_target_sample: 编码后的目标样本特征。
    - temperature: 温度参数，用于调整相似度计算的尺度。

    返回:
    - loss: 计算得到的损失值。
    """
    # 将特征向量连接为一个张量
    features = torch.cat([encoded_closest, encoded_farthest, encoded_target_sample], dim=0)

    # 计算样本对之间的相似度矩阵
    similarity_matrix = torch.matmul(features, features.t()) / temperature

    # 构建掩码，将相似度矩阵的对角线元素排除在外
    mask = torch.eye(similarity_matrix.size(0), dtype=bool).to(similarity_matrix.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

    # 计算正样本的对数概率
    pos_similarities = similarity_matrix[:len(encoded_closest), len(encoded_closest):2 * len(encoded_closest)]
    pos_log_prob = F.log_softmax(pos_similarities, dim=1).diagonal().mean()

    # 计算负样本的对数概率
    neg_similarities = similarity_matrix[:len(encoded_closest), 2 * len(encoded_closest):]
    neg_log_prob = F.log_softmax(neg_similarities, dim=1).mean()

    # 计算NT-Xent损失
    loss = -(pos_log_prob + neg_log_prob)

    return loss
def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    计算三元组损失
    """
    pos_dist = torch.norm(anchor - positive, p=2, dim=1)
    neg_dist = torch.norm(anchor - negative, p=2, dim=1)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    return loss.mean()

def dynamic_triplet_loss(embeddings, positions, threshold_pos, threshold_neg, threshold_radius):
    """
    根据动态选择的正负样本对计算三元组损失。
    """
    num_samples = embeddings.shape[0]
    losses = []
    margin = 1.0  # 三元组损失的边界值

    for i in range(num_samples):
        # 计算距离
        distances = torch.norm(embeddings - embeddings[i], dim=1)
        position_distances = torch.norm(positions - positions[i], dim=1)
        
        # 动态选择正负样本
        positive_mask = (position_distances < threshold_pos) & (position_distances > 0)
        negative_mask = (position_distances > threshold_neg) & (position_distances < threshold_radius)
        
        if positive_mask.any() and negative_mask.any():
            # 选择最近的正样本和最远的负样本
            positive_indices = torch.where(positive_mask)[0]
            negative_indices = torch.where(negative_mask)[0]
            
            # 可以随机选择，或选择距离最近的正样本和最远的负样本
            positive_idx = positive_indices[torch.argmin(distances[positive_indices])]
            negative_idx = negative_indices[torch.argmax(distances[negative_indices])]
            
            # 计算三元组损失
            anchor = embeddings[i].unsqueeze(0)
            positive = embeddings[positive_idx].unsqueeze(0)
            negative = embeddings[negative_idx].unsqueeze(0)
            loss = triplet_loss(anchor, positive, negative, margin)
            losses.append(loss)
    
    return torch.stack(losses).mean() if losses else torch.tensor(0.0)

# 定义多标签分类准确率计算函数
def calculate_multilabel_accuracy(predictions, targets, threshold=0.0303):
    """
    计算多标签分类的准确率。

    参数:
    - predictions: 模型的预测输出。
    - targets: 真实的标签。
    - threshold: 预测概率的阈值，用于将概率转换为二进制标签。

    返回:
    - accuracy: 计算得到的准确率。
    """
    # 将预测的概率值转换为二进制标签
    predicted_labels = (predictions > threshold).float()

    # 计算正确预测的数量
    correct_predictions = (predicted_labels == targets).all(dim=1).sum().item()

    # 计算准确率
    accuracy = correct_predictions / targets.size(0)

    return accuracy


# 定义自定义数据集类
class AEDataset(Dataset):
    """
    自定义的自编码器数据集类。

    参数:
    - adata: 数据集中的特征数据。
    - celltype_list: 细胞类型列表。
    - label: 数据集中的标签（可选）。
    """

    def __init__(self, data, celltype_list, label=None):
        self.data = data
        # 加上了position的维度2
        self.data_lenth = data.shape[1]+2
        self.label = label
        self.celltype_list = celltype_list
        self.num_label = torch.zeros(len(celltype_list))
        self.position_label = torch.zeros(2)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # 获取稀疏矩阵数据和标签数组
        input_data = self.data[index].toarray()
        input_data = torch.tensor(input_data, dtype=torch.float32).clone().detach()
        
        # 拼接 [0, 0] 到 input_data (初始设置)
        zero_position_label = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        input_data = torch.cat((input_data, zero_position_label), dim=1)
        
        if self.label is not None:
            if all(col in self.label.columns for col in self.celltype_list):
                self.num_label = self.label[self.celltype_list].iloc[index].values
                self.num_label = torch.tensor(self.num_label, dtype=torch.float32).clone().detach()
            
            if self.label[['x', 'y']] is not None:
                self.position_label = self.label[['x', 'y']].iloc[index].values
                self.position_label = torch.tensor(self.position_label, dtype=torch.float32).clone().detach()
                
                # 替换 input_data 最后的两个值为 position_label 的值
                input_data[:, -2:] = self.position_label.unsqueeze(0)

        return input_data, self.num_label, self.position_label
        


# 定义编码器和解码器的网络块
class EncoderBlock(nn.Module):
    """
    编码器网络块，包含线性层、LeakyReLU激活函数和Dropout。

    参数:
    - in_dim: 输入维度。
    - out_dim: 输出维度。
    - do_rates: Dropout比率。
    """

    def __init__(self, in_dim, out_dim, do_rates):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))

    def forward(self, x):
        out = self.layer(x)
        return out

class DecoderBlock(nn.Module):
    """
    解码器网络块，结构与编码器块相似。
    """

    def __init__(self, in_dim, out_dim, do_rates):
        super(DecoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))

    def forward(self, x):
        out = self.layer(x)
        return out

# 定义编码器和预测器模型
class Encoder(nn.Module):
    """
    编码器模型，用于将输入特征编码为低维表示。
    """

    def __init__(self, input_size):
        super(Encoder, self).__init__()
        self.encoder_da = nn.Sequential(EncoderBlock(input_size, 512, 0),
                                        EncoderBlock(512, 256, 0.3))

    def forward(self, x):
        encoded = self.encoder_da(x)
        return encoded


class Predictor(nn.Module):
    """
    预测器模型，用于基于编码后的特征预测标签。
    """

    def __init__(self, output_size):
        super(Predictor, self).__init__()
        self.predictor_da = nn.Sequential(EncoderBlock(256, 128, 0.2),
                                          nn.Linear(128, output_size),
                                          nn.Softmax(dim=1))

    def forward(self, x):
        predicted_ratios = self.predictor_da(x)
        return predicted_ratios

class DomainClassifier(nn.Module):
    def __init__(self, input_size):
        super(DomainClassifier, self).__init__()
        # 定义领域分类器的结构
        self.discriminator_da = nn.Sequential(EncoderBlock(256, 128, 0.2),
                                              nn.Linear(128, 1),
                                              nn.Sigmoid())
    def forward(self, x):
        # 前向传播计算领域分类结果
        discriminator_da = self.discriminator_da(x)
        return discriminator_da
    
# 定义自编码器(AE)的训练函数
def trainAE(ref_dataset, target_dataset, batch_size, shuffle, num_epoch,
            filepath, celltype_list, learning_rate, is_DAN, is_CL):
    """
    训练自编码器模型。

    参数:
    - ref_dataset: 参考数据集，用于训练编码器。
    - target_dataset: 目标数据集，未使用但为保持函数签名的一致性而保留。
    - batch_size: 批处理大小。
    - shuffle: 是否在每个epoch打乱数据。
    - num_epoch: 迭代次数。
    - filepath: 模型保存路径。
    - celltype_list: 细胞类型列表。
    - learning_rate: 学习率。

        """
    cuda_available = torch.cuda.is_available()

    print("CUDA Available:", cuda_available)

    # 设置设备
    device = torch.device("cuda:0")

    # 初始化数据加载器
    target_dataloder = DataLoader(target_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    ref_dataloder = DataLoader(ref_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


    # 初始化模型
    input_size = ref_dataset.data_lenth
    encoder = Encoder(input_size=input_size).to(device)
    predictor = Predictor(output_size=len(celltype_list)).to(device)
    domainclassifier = DomainClassifier(input_size=input_size).to(device)

    # 加载预训练的模型参数
    if os.path.exists(f'{filepath}DANN_model/Model_parameters/encoder_model.pth'):
        newly_encoder_model = get_newly_filename(f'{filepath}DANN_model/Model_parameters/', 'encoder_model.pth')
        newly_predictor_model = get_newly_filename(f'{filepath}AE_model/Model_parameters/', 'predictor_model.pth')
        newly_domain_model = get_newly_filename(f'{filepath}DANN_model/Model_parameters/','domainclassifier_model.pth')
        domainclassifier.load_state_dict(torch.load(f'{newly_domain_model}'))
        encoder.load_state_dict(torch.load(newly_encoder_model))
        predictor.load_state_dict(torch.load(newly_predictor_model))
    elif os.path.exists(f'{filepath}AE_model/Model_parameters/predictor_model.pth'):
        newly_encoder_model = get_newly_filename(f'{filepath}AE_model/Model_parameters/', 'encoder_model.pth')
        newly_predictor_model = get_newly_filename(f'{filepath}AE_model/Model_parameters/', 'predictor_model.pth')
        encoder.load_state_dict(torch.load(newly_encoder_model))
        predictor.load_state_dict(torch.load(newly_predictor_model))
    # 定义损失函数和优化器
    # bce_loss_fn = nn.BCELoss()
    l1_loss_fn = nn.L1Loss()
    optimizer_predictor = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=learning_rate)

    # 损失记录
    loss_list = []
    loss_list_l1 = []
    loss_list_nt = []
    # loss_list_bce = []

    # 训练循环
    for epoch in range(num_epoch):
        print(f"Epoch {epoch + 1}/{num_epoch}")
        encoder.train()
        predictor.train()

        total_loss_epoch = 0.0
        sum_accuracy = 0.0
        total_loss_l1_sum = 0.0
        total_loss_nt_sum = 0.0
        total_loss_BCE_sum = 0.0

        # 创建迭代器，方便使用next手动加载数据
        target_iter = iter(target_dataloder)
        

        for batch_idx, batch_ref in enumerate(ref_dataloder):

            batch_target = next(target_iter)

            # 数据准备
            input_tensor_batch_ref, input_label_batch_ref , input_position_label_batch_ref = batch_ref[0].to(device), batch_ref[1].to(device), batch_ref[2].to(device)
            input_tensor_batch_target= batch_target[0].to(device)

            # 执行前向传播
            encoded_ref = encoder(input_tensor_batch_ref.squeeze(dim=1))
            encoded_target = encoder(input_tensor_batch_target.squeeze(dim=1))

            # 使用domainclassifier，并计算BCEloss损失
            domainclassified_target = domainclassifier(encoded_target)
            domainclassified_ref = domainclassifier(encoded_ref)
            #BCEloss = bce_loss_fn(domainclassified_target, target_label) + bce_loss_fn(domainclassified_ref, ref_label)
            MMD_loss = mmd_loss(domainclassified_target, domainclassified_ref)
            CL_loss = dynamic_contrastive_loss(encoded_ref, input_position_label_batch_ref, threshold_pos=2, threshold_neg=3, threshold_radius=6)


            # 计算损失
            predicted_label = predictor(encoded_ref)
            l1_loss = l1_loss_fn(predicted_label, input_label_batch_ref)
            if is_DAN:
                if is_CL:
                    total_loss = l1_loss + MMD_loss + CL_loss
                else:
                    total_loss = l1_loss + MMD_loss
            else:
                if is_CL:
                    total_loss = l1_loss + CL_loss* 0.1 #+ BCEloss
                else:
                    total_loss = l1_loss
                

            # 反向传播和优化
            optimizer_predictor.zero_grad()
            total_loss.backward()
            optimizer_predictor.step()

            # 更新统计数据
            total_loss_epoch += total_loss.item()
            sum_accuracy += calculate_multilabel_accuracy(predicted_label, input_label_batch_ref)
            total_loss_l1_sum += l1_loss.item()
            # total_loss_BCE_sum += BCEloss.item()
            total_loss_nt_sum += CL_loss.item()

        # 输出统计信息
        avg_accuracy = sum_accuracy / len(ref_dataloder)
        avg_loss_epoch = total_loss_epoch / len(ref_dataloder)
        avg_loss_epoch_l1 = total_loss_l1_sum / len(ref_dataloder)
        # avg_loss_bce = total_loss_BCE_sum / len(ref_dataloder)
        CL_loss = total_loss_nt_sum / len(ref_dataloder)
        print(
            f"Epoch {epoch + 1}: Avg Loss: {avg_loss_epoch}, L1 Loss: {avg_loss_epoch_l1}, NT-Xent Loss: {CL_loss}, Accuracy: {avg_accuracy}")

        # 记录损失
        loss_list.append(avg_loss_epoch)
        loss_list_l1.append(avg_loss_epoch_l1)
        loss_list_nt.append(CL_loss)
        # loss_list_bce.append(avg_loss_bce)


    # 绘制和保存损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label='Total Loss')
    plt.plot(loss_list_l1, label='L1 Loss')
    plt.plot(loss_list_nt, label='CL Loss')
    # plt.plot(loss_list_bce, label = 'BCE loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    ae_pic_unique_filename = get_unique_filename(f'{filepath}AE_model/Model_result/','loss_curves.png')
    plt.savefig(ae_pic_unique_filename)
    plt.show()

    # 保存模型参数
    encoder_model_unique_filename = get_unique_filename(f'{filepath}AE_model/Model_parameters/','encoder_model.pth')
    predictor_model_unique_filename = get_unique_filename(f'{filepath}AE_model/Model_parameters/','predictor_model.pth')

    torch.save(encoder.state_dict(), encoder_model_unique_filename)
    torch.save(predictor.state_dict(), predictor_model_unique_filename)


def evalAE(target_dataset, batch_size, filepath, celltype_list, testdata_name):
    """
    评估自编码器(AE)模型在目标数据集上的性能。

    参数:
    - target_dataset: 用于评估的目标数据集。
    - batch_size: DataLoader的批处理大小。
    - filepath: 模型参数保存的路径。
    - celltype_list: 细胞类型列表，用于确定模型输出大小。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'GPU' if torch.cuda.is_available() else 'CPU'} is available for evaluation.")

    # 初始化目标数据集的DataLoader
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 模型初始化
    input_size = target_dataset.data_lenth
    encoder = Encoder(input_size).to(device)
    predictor = Predictor(len(celltype_list)).to(device)

    # 确定并加载最新的模型参数
    # 获取最新模型文件的路径
    dann_model_index = get_newly_filename_index(f'{filepath}/DANN_model/Model_parameters/', 'encoder_model.pth')
    ae_model_index = get_newly_filename_index(f'{filepath}/AE_model/Model_parameters/', 'encoder_model.pth')

    if dann_model_index >= ae_model_index:
        newly_encoder_model = get_newly_filename(f'{filepath}/DANN_model/Model_parameters/', 'encoder_model.pth')
    else:
        newly_encoder_model = get_newly_filename(f'{filepath}/AE_model/Model_parameters/', 'encoder_model.pth')

    newly_predictor_model = get_newly_filename(f'{filepath}/AE_model/Model_parameters/', 'predictor_model.pth')

    # 加载模型权重
    if os.path.exists(newly_encoder_model):
        encoder.load_state_dict(torch.load(newly_encoder_model, map_location=device))
        print(f'Encoder model{newly_encoder_model} loaded successfully.')
    else:
        print("Encoder model file does not exist.")

    if os.path.exists(newly_predictor_model):
        predictor.load_state_dict(torch.load(newly_predictor_model, map_location=device))
        print(f'Predictor model{newly_predictor_model} loaded successfully.')
    else:
        print("Predictor model file does not exist.")

    encoder.eval()
    predictor.eval()

    # 评估模型
    all_outputs = []
    with torch.no_grad():
        for batch_data in target_dataloader:
            input_data, _ , _ = batch_data
            input_data = input_data.squeeze(dim=1).to(device)
            encoded_output = encoder(input_data)
            predictor_output = predictor(encoded_output)
            all_outputs.append(predictor_output.cpu().numpy())

    # 合并所有批次的输出并保存到CSV文件
    predictor_matrix_ref = np.concatenate(all_outputs, axis=0)
    df = pd.DataFrame(predictor_matrix_ref)
    os.makedirs(f'{filepath}AE_model/Model_result/{testdata_name}/', exist_ok=True)
    csv_file_path = get_unique_filename(f'{filepath}AE_model/Model_result/{testdata_name}/', 'predictor_matrix_ref.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"Predictor matrix saved to {csv_file_path}.")
    return df