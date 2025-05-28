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
        # Added the position dimension, which has a size of 2
        self.data_lenth = data.shape[1]+2

        self.celltype_list = celltype_list
        self.num_label = torch.zeros(len(celltype_list))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # Load sparse matrix data and corresponding label array
        input_data = self.data[index].toarray()
        input_data = torch.tensor(input_data, dtype=torch.float32).clone().detach()
        # # Append [0, 0] to input_data (initial setup)
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
# Definition of the domain classifier
class DomainClassifier(nn.Module):
    def __init__(self, input_size):
        super(DomainClassifier, self).__init__()
        # Define the architecture of the domain classifier
        self.discriminator_da = nn.Sequential(EncoderBlock(256, 128, 0.2),
                                              nn.Linear(128, 1),
                                              nn.Sigmoid())
    def forward(self, x):
        # Perform forward propagation to compute domain classification results
        discriminator_da = self.discriminator_da(x)
        return discriminator_da
def trainDAN(ref_dataset, target_dataset, batch_size, shuffle, num_epoch, filepath, learning_rate):
    cuda_available = torch.cuda.is_available()

    print("CUDA Available:", cuda_available)

    # Set computing device (GPU)
    device = torch.device("cuda:0")
    # Wrap the custom dataset in a DataLoader
    ref_dataloder = torch.utils.data.DataLoader(ref_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    target_dataloder = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    target_label  = torch.zeros(batch_size, 1).to(device) # 定义target domain label为0
    ref_label= torch.ones(batch_size, 1).to(device)  # 定义ref domain label为1

    input_size = ref_dataset.data_lenth
    
    # Instantiate the model and transfer it to the designated device
    encoder = Encoder(input_size=input_size).to(device)
    domainclassifier = DomainClassifier(input_size=input_size).to(device)

    os.makedirs(f'{filepath}DANN_model/Model_parameters/', exist_ok=True)
    os.makedirs(f'{filepath}DANN_model/Model_result/', exist_ok=True)

    # Load previously saved model parameters
    newly_encoder_model = get_newly_filename(f'{filepath}AE_model/Model_parameters/','encoder_model.pth')
    newly_domain_model = get_newly_filename(f'{filepath}DANN_model/Model_parameters/','domainclassifier_model.pth')
    encoder.load_state_dict(torch.load(f'{newly_encoder_model}'))
    print('load',newly_encoder_model)
    if os.path.exists(newly_domain_model):

        domainclassifier.load_state_dict(torch.load(f'{newly_domain_model}'))

    # Define the loss function and the optimizer
    bce_loss_fn = nn.BCELoss()
    optimizer_domainclassifier = torch.optim.Adam(domainclassifier.parameters(), lr=learning_rate)
    # Perform model training
    num_epochs = num_epoch
    loss_list = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        encoder.train()
        domainclassifier.train()

        total_loss_epoch = 0.0  # Record the total loss for each epoch
        sum_accuracy = 0.0
        total_loss_BCE_sum = 0.0

        target_iter = iter(target_dataloder)


        for batch_idx, batch_ref in enumerate(ref_dataloder):
            batch_target = next(target_iter)
            # Retrieve batch data and move it to the device
            input_tensor_batch_ref, input_label_batch_ref = batch_ref
            input_tensor_batch_ref = input_tensor_batch_ref.to(device)
            input_label_batch_ref = input_label_batch_ref.to(device)

            input_tensor_batch1_target, input_label_batch1_target = batch_target
            input_tensor_batch1_target = input_tensor_batch1_target.to(device)

            input_tensor_batch_ref = input_tensor_batch_ref.squeeze(dim=1)
            input_tensor_batch1_target = input_tensor_batch1_target.squeeze(dim=1)

            # Encode the input matrix
            encoded_target = encoder(input_tensor_batch1_target)
            encoded_ref = encoder(input_tensor_batch_ref)


            # Apply the domain classifier and calculate the BCE loss
            domainclassified_target = domainclassifier(encoded_target)
            domainclassified_ref = domainclassifier(encoded_ref)
            BCEloss = bce_loss_fn(domainclassified_target, target_label) + bce_loss_fn(domainclassified_ref, ref_label)


            # Compute the combined loss
            total_loss = BCEloss
            total_loss_epoch += total_loss.item() # Add up the loss from each batch
            total_loss_BCE = BCEloss

            total_loss_BCE_sum += total_loss_BCE.item()

            # Zero optimizer gradients, backpropagate, and optimize
            optimizer_domainclassifier.zero_grad()
            total_loss.backward()

            # Perform optimization for the domain classifier
            optimizer_domainclassifier.step()

        # Log the average loss for each epoch
        avg_accuracy = sum_accuracy / len(ref_dataloder)
        avg_loss_epoch = total_loss_epoch / len(ref_dataloder)
        avg_loss_BCE = total_loss_BCE_sum / len(ref_dataloder)
        print(f"Average Loss for Epoch {epoch + 1}: {avg_loss_epoch}")
        print(f"Average accuracy for Epoch {epoch + 1}: {avg_accuracy}")
        print(f"Average loss_BCE for Epoch {epoch + 1}: {avg_loss_BCE}")
        loss_list.append(avg_loss_epoch)

    x = list(range(2, len(loss_list) + 1))
    loss_list = loss_list[1:]

    # Plot the loss curve
    plt.plot(x, loss_list)

    folder_path = f'{filepath}DANN_model/Model_result/'
    L1_filename = "BCE_loss.png"
    L1_unique_filename =get_unique_filename(folder_path,L1_filename)
    plt.savefig(L1_unique_filename)

    # Display the plot
    plt.show()
    # Save model parameters after training
    encoder_model_unique = get_unique_filename(f'{filepath}DANN_model/Model_parameters/','encoder_model.pth')
    domainclassifier_model_unique = get_unique_filename(f'{filepath}DANN_model/Model_parameters/','domainclassifier_model.pth')
    torch.save(encoder.state_dict(), f'{encoder_model_unique}')
    torch.save(domainclassifier.state_dict(), f'{domainclassifier_model_unique}')



