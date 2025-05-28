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
    Compute MK-MMD distance as a loss function.
    Args:
    - X: Sample data from the first distribution.
    - Y: Sample data from the second distribution.
    Returns:
    - mmd: MK-MMD distance.
    """
    sigma = 1.0  # Bandwidth parameter for the Gaussian kernel
    n = X.size(0)
    m = Y.size(0)

    # Gaussian kernel function
    def kernel(x, y):
        return torch.exp(-torch.sum((x - y)**2) / (2 * sigma**2))

    # Compute kernel values between samples
    xx = torch.mean(torch.stack([kernel(x, x) for x in X]))
    yy = torch.mean(torch.stack([kernel(y, y) for y in Y]))
    xy = torch.mean(torch.stack([kernel(x, y) for x in X for y in Y]))

    # Compute MK-MMD distance
    mmd = xx - 2 * xy + yy
    
    return mmd


# Define contrastive learning loss function
def nt_xent_loss(encoded_closest, encoded_farthest, encoded_target_sample, temperature=0.1):
    """
    Compute NT-Xent loss (Normalized Temperature-scaled Cross Entropy Loss).

    Args:
    - encoded_closest: Encoded features of the nearest neighbor sample.
    - encoded_farthest: Encoded features of the farthest neighbor sample.
    - encoded_target_sample: Encoded features of the target sample.
    - temperature: Temperature parameter to scale similarity scores.

    Returns:
    - loss: Computed loss value.
    """
    # Concatenate feature vectors into a single tensor
    features = torch.cat([encoded_closest, encoded_farthest, encoded_target_sample], dim=0)

    # Compute similarity matrix between sample pairs
    similarity_matrix = torch.matmul(features, features.t()) / temperature

    # Mask to exclude diagonal elements of similarity matrix
    mask = torch.eye(similarity_matrix.size(0), dtype=bool).to(similarity_matrix.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

    # Compute log-probability for positive samples
    pos_similarities = similarity_matrix[:len(encoded_closest), len(encoded_closest):2 * len(encoded_closest)]
    pos_log_prob = F.log_softmax(pos_similarities, dim=1).diagonal().mean()

    # Compute log-probability for negative samples
    neg_similarities = similarity_matrix[:len(encoded_closest), 2 * len(encoded_closest):]
    neg_log_prob = F.log_softmax(neg_similarities, dim=1).mean()

    # Compute NT-Xent loss
    loss = -(pos_log_prob + neg_log_prob)

    return loss


# Define multi-label classification accuracy function
def calculate_multilabel_accuracy(predictions, targets, threshold=0.0303):
    """
    Compute accuracy for multi-label classification.

    Args:
    - predictions: Model output predictions.
    - targets: Ground truth labels.
    - threshold: Threshold to convert probabilities to binary labels.

    Returns:
    - accuracy: Computed accuracy.
    """
    # Convert probabilities to binary labels
    predicted_labels = (predictions > threshold).float()

    # Count correct predictions
    correct_predictions = (predicted_labels == targets).all(dim=1).sum().item()

    # Compute accuracy
    accuracy = correct_predictions / targets.size(0)

    return accuracy


# Define custom dataset class for autoencoder
class AEDataset(Dataset):
    """
    Custom autoencoder dataset class.

    Args:
    - adata: Feature data.
    - celltype_list: List of cell types.
    - label: Label data (optional).
    """
    def __init__(self, data, celltype_list, label=None):
        self.data = data
        self.label = label
        self.celltype_list = celltype_list
        self.num_label = torch.zeros(len(celltype_list))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # Get sparse matrix data and labels
        input_data = self.data[index].toarray()
        input_data = torch.tensor(input_data, dtype=torch.float32).clone().detach()

        if self.label is not None:
            self.num_label = self.label[self.celltype_list].iloc[index].values
            self.num_label = torch.tensor(self.num_label, dtype=torch.float32).clone().detach()
        return input_data, self.num_label


# Define encoder and decoder network blocks
class EncoderBlock(nn.Module):
    """
    Encoder block with Linear layer, LeakyReLU activation, and Dropout.

    Args:
    - in_dim: Input dimension.
    - out_dim: Output dimension.
    - do_rates: Dropout rate.
    """
    def __init__(self, in_dim, out_dim, do_rates):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))

    def forward(self, x):
        return self.layer(x)


class DecoderBlock(nn.Module):
    """
    Decoder block, similar in structure to the encoder block.
    """
    def __init__(self, in_dim, out_dim, do_rates):
        super(DecoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))

    def forward(self, x):
        return self.layer(x)


# Define encoder and predictor models
class Encoder(nn.Module):
    """
    Encoder model to encode input features into low-dimensional representations.
    """
    def __init__(self, input_size):
        super(Encoder, self).__init__()
        self.encoder_da = nn.Sequential(EncoderBlock(input_size, 512, 0),
                                        EncoderBlock(512, 256, 0.3))

    def forward(self, x):
        return self.encoder_da(x)


class Predictor(nn.Module):
    """
    Predictor model to predict labels from encoded features.
    """
    def __init__(self, output_size):
        super(Predictor, self).__init__()
        self.predictor_da = nn.Sequential(EncoderBlock(256, 128, 0.2),
                                          nn.Linear(128, output_size),
                                          nn.Softmax(dim=1))

    def forward(self, x):
        return self.predictor_da(x)


class DomainClassifier(nn.Module):
    """
    Domain classifier for domain adaptation.
    """
    def __init__(self, input_size):
        super(DomainClassifier, self).__init__()
        # Define domain classifier structure
        self.discriminator_da = nn.Sequential(EncoderBlock(256, 128, 0.2),
                                              nn.Linear(128, 1),
                                              nn.Sigmoid())

    def forward(self, x):
        # Forward pass for domain classification
        return self.discriminator_da(x)
    
# Define training function for the Autoencoder (AE)
def trainAE(ref_dataset, target_dataset, data_closest, data_farthest, data_ref_sample, batch_size, shuffle, num_epoch,
            filepath, celltype_list, learning_rate, is_DAN):
    """
    Train the autoencoder model.

    Args:
    - ref_dataset: Reference dataset used for training the encoder.
    - target_dataset: Target dataset (not used but kept for consistent function signature).
    - data_closest: Nearest neighbor dataset.
    - data_farthest: Farthest neighbor dataset.
    - data_ref_sample: Reference sample dataset.
    - batch_size: Batch size.
    - shuffle: Whether to shuffle the data at each epoch.
    - num_epoch: Number of training epochs.
    - filepath: Path to save model files.
    - celltype_list: List of cell types.
    - learning_rate: Learning rate.
    """
    cuda_available = torch.cuda.is_available()
    print("CUDA Available:", cuda_available)

    # Set device
    device = torch.device("cuda:0")

    # Initialize data loaders
    target_dataloder = DataLoader(target_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    ref_dataloder = DataLoader(ref_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    closest_dataloder = DataLoader(data_closest, batch_size=batch_size * 3, shuffle=shuffle, drop_last=True)
    farthest_dataloder = DataLoader(data_farthest, batch_size=batch_size * 3, shuffle=shuffle, drop_last=True)
    target_sample_dataloder = DataLoader(data_ref_sample, batch_size=batch_size * 3, shuffle=shuffle, drop_last=True)

    target_label = torch.ones(batch_size, 1).to(device)  # Define target domain label as 1
    ref_label = torch.zeros(batch_size, 1).to(device)    # Define reference domain label as 0

    # Initialize models
    input_size = ref_dataset.data.shape[1]
    encoder = Encoder(input_size=input_size).to(device)
    predictor = Predictor(output_size=len(celltype_list)).to(device)
    domainclassifier = DomainClassifier(input_size=input_size).to(device)

    # Load pre-trained model parameters
    if os.path.exists(f'{filepath}DANN_model/Model_parameters/encoder_model.pth'):
        newly_encoder_model = get_newly_filename(f'{filepath}DANN_model/Model_parameters/', 'encoder_model.pth')
        newly_predictor_model = get_newly_filename(f'{filepath}AE_model/Model_parameters/', 'predictor_model.pth')
        newly_domain_model = get_newly_filename(f'{filepath}DANN_model/Model_parameters/', 'domainclassifier_model.pth')
        domainclassifier.load_state_dict(torch.load(f'{newly_domain_model}'))
        encoder.load_state_dict(torch.load(newly_encoder_model))
        predictor.load_state_dict(torch.load(newly_predictor_model))
    elif os.path.exists(f'{filepath}AE_model/Model_parameters/predictor_model.pth'):
        newly_encoder_model = get_newly_filename(f'{filepath}AE_model/Model_parameters/', 'encoder_model.pth')
        newly_predictor_model = get_newly_filename(f'{filepath}AE_model/Model_parameters/', 'predictor_model.pth')
        encoder.load_state_dict(torch.load(newly_encoder_model))
        predictor.load_state_dict(torch.load(newly_predictor_model))

    # Define loss functions and optimizer
    bce_loss_fn = nn.BCELoss()
    l1_loss_fn = nn.L1Loss()
    optimizer_predictor = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=learning_rate)

    # Loss tracking
    loss_list = []
    loss_list_l1 = []
    loss_list_nt = []
    loss_list_bce = []

    # Training loop
    for epoch in range(num_epoch):
        print(f"Epoch {epoch + 1}/{num_epoch}")
        encoder.train()
        predictor.train()

        total_loss_epoch = 0.0
        sum_accuracy = 0.0
        total_loss_l1_sum = 0.0
        total_loss_nt_sum = 0.0
        total_loss_BCE_sum = 0.0

        # Create iterators for manual batch loading
        closest_iter = iter(closest_dataloder)
        farthest_iter = iter(farthest_dataloder)
        target_sample_iter = iter(target_sample_dataloder)
        target_iter = iter(target_dataloder)

        for batch_idx, batch_ref in enumerate(ref_dataloder):
            batch_target = next(target_iter)

            # Load batch data
            batch_closet = next(closest_iter)
            batch_farthest = next(farthest_iter)
            batch_target_sample = next(target_sample_iter)

            # Prepare data
            input_tensor_batch_ref, input_label_batch_ref = batch_ref[0].to(device), batch_ref[1].to(device)
            input_tensor_batch_target, input_label_batch_target = batch_target[0].to(device), batch_target[1].to(device)
            input_tensor_batch_closest_target = batch_closet[0].to(device)
            input_tensor_batch_farthest_target = batch_farthest[0].to(device)
            input_tensor_batch_target_sample_target = batch_target_sample[0].to(device)

            # Forward pass
            encoded_ref = encoder(input_tensor_batch_ref.squeeze(dim=1))
            encoded_target = encoder(input_tensor_batch_target.squeeze(dim=1))
            encoded_closest = encoder(input_tensor_batch_closest_target.squeeze(dim=1))
            encoded_farthest = encoder(input_tensor_batch_farthest_target.squeeze(dim=1))
            encoded_target_sample = encoder(input_tensor_batch_target_sample_target.squeeze(dim=1))

            # Compute domain classification and BCE loss
            domainclassified_target = domainclassifier(encoded_target)
            domainclassified_ref = domainclassifier(encoded_ref)
            BCEloss = bce_loss_fn(domainclassified_target, target_label) + bce_loss_fn(domainclassified_ref, ref_label)

            # Compute total loss
            predicted_label = predictor(encoded_ref)
            l1_loss = l1_loss_fn(predicted_label, input_label_batch_ref)
            nt_xent_loss_num = nt_xent_loss(encoded_closest, encoded_farthest, encoded_target_sample)
            if is_DAN:
                total_loss = 1.5 * l1_loss + BCEloss  # + nt_xent_loss_num * 0.02
            else:
                total_loss = 1.5 * l1_loss  # + BCEloss + nt_xent_loss_num * 0.02

            # Backpropagation and optimization
            optimizer_predictor.zero_grad()
            total_loss.backward()
            optimizer_predictor.step()

            # Update stats
            total_loss_epoch += total_loss.item()
            sum_accuracy += calculate_multilabel_accuracy(predicted_label, input_label_batch_ref)
            total_loss_l1_sum += l1_loss.item()
            total_loss_BCE_sum += BCEloss.item()
            total_loss_nt_sum += nt_xent_loss_num.item()

        # Output statistics
        avg_accuracy = sum_accuracy / len(ref_dataloder)
        avg_loss_epoch = total_loss_epoch / len(ref_dataloder)
        avg_loss_epoch_l1 = total_loss_l1_sum / len(ref_dataloder)
        avg_loss_bce = total_loss_BCE_sum / len(ref_dataloder)
        avg_loss_nt = total_loss_nt_sum / len(ref_dataloder)
        print(f"Epoch {epoch + 1}: Avg Loss: {avg_loss_epoch}, L1 Loss: {avg_loss_epoch_l1}, BCE Loss: {avg_loss_bce}, NT-Xent Loss: {avg_loss_nt}, Accuracy: {avg_accuracy}")

        # Log losses
        loss_list.append(avg_loss_epoch)
        loss_list_l1.append(avg_loss_epoch_l1)
        loss_list_nt.append(avg_loss_nt)
        loss_list_bce.append(avg_loss_bce)

    # Plot and save loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label='Total Loss')
    plt.plot(loss_list_l1, label='L1 Loss')
    # plt.plot(loss_list_nt, label='NT-Xent Loss')
    plt.plot(loss_list_bce, label='BCE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    ae_pic_unique_filename = get_unique_filename(f'{filepath}AE_model/Model_result/', 'loss_curves.png')
    plt.savefig(ae_pic_unique_filename)
    plt.show()

    # Save model parameters
    encoder_model_unique_filename = get_unique_filename(f'{filepath}AE_model/Model_parameters/', 'encoder_model.pth')
    predictor_model_unique_filename = get_unique_filename(f'{filepath}AE_model/Model_parameters/', 'predictor_model.pth')

    torch.save(encoder.state_dict(), encoder_model_unique_filename)
    torch.save(predictor.state_dict(), predictor_model_unique_filename)

def evalAE(target_dataset, batch_size, filepath, celltype_list, testdata_name):
    """
    Evaluate the Autoencoder (AE) model on the target dataset.

    Args:
    - target_dataset: Target dataset for evaluation.
    - batch_size: Batch size for the DataLoader.
    - filepath: Path to the saved model parameters.
    - celltype_list: List of cell types used to determine the model output size.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'GPU' if torch.cuda.is_available() else 'CPU'} is available for evaluation.")

    # Initialize DataLoader for the target dataset
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Initialize models
    input_size = target_dataset.data.shape[1]
    encoder = Encoder(input_size).to(device)
    predictor = Predictor(len(celltype_list)).to(device)

    # Determine and load the most recent model parameters
    # Get the most recent model file path
    dann_model_index = get_newly_filename_index(f'{filepath}/DANN_model/Model_parameters/', 'encoder_model.pth')
    ae_model_index = get_newly_filename_index(f'{filepath}/AE_model/Model_parameters/', 'encoder_model.pth')

    if dann_model_index >= ae_model_index:
        newly_encoder_model = get_newly_filename(f'{filepath}/DANN_model/Model_parameters/', 'encoder_model.pth')
    else:
        newly_encoder_model = get_newly_filename(f'{filepath}/AE_model/Model_parameters/', 'encoder_model.pth')

    newly_predictor_model = get_newly_filename(f'{filepath}/AE_model/Model_parameters/', 'predictor_model.pth')

    # Load model weights
    if os.path.exists(newly_encoder_model):
        encoder.load_state_dict(torch.load(newly_encoder_model, map_location=device))
        print("Encoder model loaded successfully.")
    else:
        print("Encoder model file does not exist.")

    if os.path.exists(newly_predictor_model):
        predictor.load_state_dict(torch.load(newly_predictor_model, map_location=device))
        print("Predictor model loaded successfully.")
    else:
        print("Predictor model file does not exist.")

    encoder.eval()
    predictor.eval()

    # Evaluate the model
    all_outputs = []
    with torch.no_grad():
        for batch_data in target_dataloader:
            input_data, _ = batch_data
            input_data = input_data.squeeze(dim=1).to(device)
            encoded_output = encoder(input_data)
            predictor_output = predictor(encoded_output)
            all_outputs.append(predictor_output.cpu().numpy())

    # Concatenate all batch outputs and save to a CSV file
    predictor_matrix_ref = np.concatenate(all_outputs, axis=0)
    df = pd.DataFrame(predictor_matrix_ref)
    os.makedirs(f'{filepath}AE_model/Model_result/{testdata_name}/', exist_ok=True)
    csv_file_path = get_unique_filename(f'{filepath}AE_model/Model_result/{testdata_name}/', 'predictor_matrix_ref.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"Predictor matrix saved to {csv_file_path}.")
    return df
