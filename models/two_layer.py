import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.core.lightning import LightningModule
import numpy as np
from torch.optim import Adam

DATA_SPLIT = {
    "train": 0.8,
    "test": 0.1,
    "val": 0.1
}

class CustomDataset(Dataset):
    def __init__(self, data):
        self.input = data[0]
        self.targets = torch.LongTensor(data[1])
        
    def __getitem__(self, index):
        x = self.input[index]
        y = self.targets[index]
        
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def load_datasets(X, Y):
    
    size = (np.array(X).shape[0])
    permute = np.random.permutation(size)

    X_p = X[permute]
    Y_p = Y[permute]
    train_size = int(size*DATA_SPLIT["train"])
    val_size = int(size*DATA_SPLIT["val"])
    test_size = int(size*DATA_SPLIT["test"])
    train = X_p[:train_size], Y_p[:train_size]
    val =  X_p[train_size: train_size + val_size], Y_p[train_size: train_size + val_size]
    test = X_p[train_size + val_size: ], Y_p[train_size + val_size: ]
    return train, val, test

class Model(LightningModule):
    def __init__(self, X, Y, loss_fn):
        super().__init__()
        
        self.loss = nn.CrossEntropyLoss()
        
        self.layer_1 = nn.Linear(384, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 384)
        
        self.train_set, self.val_set, self.test_set = load_datasets()

        
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return DataLoader(CustomDataset(self.train_set), batch_size=64)

    def val_dataloader(self):
        
        return DataLoader(CustomDataset(self.val_set), batch_size=64)

    def test_dataloader(self):
        return DataLoader(CustomDataset(self.test_set), batch_size=64)

    def test_step(self, batch, batch_idx):
        x, y = batch
        predicted = self(x)
        loss = self.loss(predicted, y)
        self.log("test_loss", loss)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        predicted = self(x)
        
        loss = self.loss(predicted, y)
        
        self.log("val_loss", loss)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        predicted = self(x)
        loss = self.loss(predicted, y)
        return loss
    

    def forward(self, x):
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        return x