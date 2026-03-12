import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import os
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 20  # Reduced for faster search, adjust if needed
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Data Loading
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        return self.deconv_layers(x)

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def hungarian_match(true_labels, cluster_assignments, n_clusters=10):
    cost_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost_matrix[i, j] = np.sum((cluster_assignments == i) & (true_labels == j))
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    mapping = dict(zip(row_ind, col_ind))
    return mapping

def evaluate_metrics(model, test_loader, latent_dim):
    model.eval()
    all_latent = []
    all_labels = []
    total_mse = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            z = model.encoder(images)
            recon = model.decoder(z)
            total_mse += nn.MSELoss()(recon, images).item() * images.size(0)
            all_latent.append(z.cpu().numpy())
            all_labels.append(labels.numpy())
    
    latent_vectors = np.concatenate(all_latent, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)
    avg_mse = total_mse / len(test_loader.dataset)
    
    # K-Means
    kmeans = KMeans(n_clusters=10, random_state=SEED, n_init=10)
    cluster_assignments = kmeans.fit_predict(latent_vectors)
    cluster_centers = kmeans.cluster_centers_
    
    # PMS
    mapping = hungarian_match(true_labels, cluster_assignments)
    predicted_labels = np.array([mapping[c] for c in cluster_assignments])
    pms = (np.sum(predicted_labels != true_labels) / len(true_labels)) * 100
    
    # AD
    distances_to_center = np.sum((latent_vectors - cluster_centers[cluster_assignments])**2, axis=1)
    ad = np.mean(distances_to_center)
    
    # AVC
    avc_per_cluster = []
    for k in range(10):
        mask = cluster_assignments == k
        if np.sum(mask) > 0:
            avc_per_cluster.append(np.mean(np.sum((latent_vectors[mask] - cluster_centers[k])**2, axis=1)))
    avc = np.mean(avc_per_cluster)
    
    # TD
    td = 0.0
    for i in range(10):
        for j in range(i + 1, 10):
            td += np.sum((cluster_centers[i] - cluster_centers[j])**2)
            
    return avg_mse, pms, ad, avc, td

def train_and_eval(latent_dim):
    print(f"\n--- Training for LATENT_DIM = {latent_dim} ---")
    model = ConvAutoencoder(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            recon = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS}, Loss: {train_loss/len(train_dataset):.6f}")
            
    metrics = evaluate_metrics(model, test_loader, latent_dim)
    return metrics

if __name__ == "__main__":
    latent_dims = [1, 2, 4, 8, 10, 12, 16, 24, 32, 48, 64]
    results = []
    
    for ld in latent_dims:
        start_time = time.time()
        mse, pms, ad, avc, td = train_and_eval(ld)
        duration = time.time() - start_time
        results.append({
            'LATENT_DIM': ld,
            'MSE': mse,
            'PMS': pms,
            'AD': ad,
            'AVC': avc,
            'TD': td,
            'Duration': duration
        })
        print(f"Results for LD={ld}: PMS={pms:.2f}%, MSE={mse:.6f}, TD={td:.2f}")
        
    df = pd.DataFrame(results)
    df.to_csv('latent_dim_search_results.csv', index=False)
    print("\nSearch complete. Results saved to latent_dim_search_results.csv")
