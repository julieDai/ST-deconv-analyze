import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch.nn.functional as F
import numpy as np
from torchviz import make_dot
from ..utils.utils import get_unique_filename, get_newly_filename
import subprocess



class DANDataset(Dataset):
    def __init__(self, data, celltype_list, label=None):
        self.data = data
        self.label = label
        # 加上了position的维度2
        self.data_lenth = data.shape[1]+2

        self.celltype_list = celltype_list
        self.num_label = torch.zeros(len(celltype_list))

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

        

        return input_data, self.num_label


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))
    def forward(self, x):
        out = self.layer(x)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(DecoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))
    def forward(self, x):
        out = self.layer(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()
       # self.encoder = nn.Linear(input_size, hidden_size)
        self.encoder_da = nn.Sequential(EncoderBlock(input_size, 512, 0),
                                        EncoderBlock(512, 256, 0.3))

    def forward(self, x):
        encoded = self.encoder_da(x)
        return encoded

class Predictor(nn.Module):
    def __init__(self, output_size):
        super(Predictor, self).__init__()
        # self.predictor = nn.Linear(input_size, output_size)
        self.predictor_da = nn.Sequential(EncoderBlock(256, 128, 0.2),
                                          nn.Linear(128, output_size),
                                          nn.Softmax(dim=1))
    def forward(self, x):
        predicted_ratios = self.predictor_da(x)
        return predicted_ratios
# 定义领域分类器（Domain Classifier）
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
def trainDAN(ref_dataset, target_dataset, batch_size, shuffle, num_epoch, filepath, learning_rate):
    cuda_available = torch.cuda.is_available()

    print("CUDA Available:", cuda_available)

    # 设置设备
    device = torch.device("cuda:0")
    # 将自定义数据集转换为 DataLoader 对象
    ref_dataloder = torch.utils.data.DataLoader(ref_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    target_dataloder = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    target_label  = torch.zeros(batch_size, 1).to(device) # 定义target domain label为0
    ref_label= torch.ones(batch_size, 1).to(device)  # 定义ref domain label为1

    input_size = ref_dataset.data_lenth
    
    # 创建模型实例并移动到 device 上
    encoder = Encoder(input_size=input_size).to(device)
    domainclassifier = DomainClassifier(input_size=input_size).to(device)

    os.makedirs(f'{filepath}DANN_model/Model_parameters/', exist_ok=True)
    os.makedirs(f'{filepath}DANN_model/Model_result/', exist_ok=True)

    #加载之前模型参数
    newly_encoder_model = get_newly_filename(f'{filepath}AE_model/Model_parameters/','encoder_model.pth')
    newly_domain_model = get_newly_filename(f'{filepath}DANN_model/Model_parameters/','domainclassifier_model.pth')
    encoder.load_state_dict(torch.load(f'{newly_encoder_model}'))
    print('load',newly_encoder_model)
    if os.path.exists(newly_domain_model):

        domainclassifier.load_state_dict(torch.load(f'{newly_domain_model}'))

    # 定义损失函数和优化器
    bce_loss_fn = nn.BCELoss()
    optimizer_domainclassifier = torch.optim.Adam(domainclassifier.parameters(), lr=learning_rate)
    # 进行模型训练
    num_epochs = num_epoch
    loss_list = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        encoder.train()
        domainclassifier.train()

        total_loss_epoch = 0.0  # 用于记录每个 epoch 的总损失
        sum_accuracy = 0.0
        total_loss_BCE_sum = 0.0

        target_iter = iter(target_dataloder)


        for batch_idx, batch_ref in enumerate(ref_dataloder):
            batch_target = next(target_iter)
            # 获取批次数据并移动到 device 上
            input_tensor_batch_ref, input_label_batch_ref = batch_ref
            input_tensor_batch_ref = input_tensor_batch_ref.to(device)
            input_label_batch_ref = input_label_batch_ref.to(device)

            input_tensor_batch1_target, input_label_batch1_target = batch_target
            input_tensor_batch1_target = input_tensor_batch1_target.to(device)

            input_tensor_batch_ref = input_tensor_batch_ref.squeeze(dim=1)
            input_tensor_batch1_target = input_tensor_batch1_target.squeeze(dim=1)

            # 编码输入矩阵
            encoded_target = encoder(input_tensor_batch1_target)
            encoded_ref = encoder(input_tensor_batch_ref)


            # 使用domainclassifier，并计算BCEloss损失
            domainclassified_target = domainclassifier(encoded_target)
            domainclassified_ref = domainclassifier(encoded_ref)
            BCEloss = bce_loss_fn(domainclassified_target, target_label) + bce_loss_fn(domainclassified_ref, ref_label)


            # 综合损失
            total_loss = BCEloss
            total_loss_epoch += total_loss.item() # 累加每个 batch 的损失
            total_loss_BCE = BCEloss

            total_loss_BCE_sum += total_loss_BCE.item()

            # 清空优化器的梯度，进行反向传播和优化
            optimizer_domainclassifier.zero_grad()
            total_loss.backward()

            # 分别进行优化
            optimizer_domainclassifier.step()

        # 输出每个 epoch 的平均损失
        avg_accuracy = sum_accuracy / len(ref_dataloder)
        avg_loss_epoch = total_loss_epoch / len(ref_dataloder)
        avg_loss_BCE = total_loss_BCE_sum / len(ref_dataloder)
        print(f"Average Loss for Epoch {epoch + 1}: {avg_loss_epoch}")
        print(f"Average accuracy for Epoch {epoch + 1}: {avg_accuracy}")
        print(f"Average loss_BCE for Epoch {epoch + 1}: {avg_loss_BCE}")
        loss_list.append(avg_loss_epoch)

    x = list(range(2, len(loss_list) + 1))
    loss_list = loss_list[1:]

    # 绘制 loss 曲线
    plt.plot(x, loss_list)

    folder_path = f'{filepath}DANN_model/Model_result/'
    L1_filename = "BCE_loss.png"
    L1_unique_filename =get_unique_filename(folder_path,L1_filename)
    plt.savefig(L1_unique_filename)

    # 显示图表
    plt.show()
    # 训练完成后，保存模型参数
    encoder_model_unique = get_unique_filename(f'{filepath}DANN_model/Model_parameters/','encoder_model.pth')
    domainclassifier_model_unique = get_unique_filename(f'{filepath}DANN_model/Model_parameters/','domainclassifier_model.pth')
    torch.save(encoder.state_dict(), f'{encoder_model_unique}')
    torch.save(domainclassifier.state_dict(), f'{domainclassifier_model_unique}')



