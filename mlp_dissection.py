from turtle import forward
import torch
import pandas as pd
import numpy as np
import sys

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from torch.nn import functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.core.lightning import LightningModule
import numpy as np
from torch.optim import Adam
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
import torchmetrics.functional as FM

# Take in alpha beta and gamma as inputs
alpha = float(sys.argv[1])
beta = float(sys.argv[2])
gamma = float(sys.argv[3])

# Calculate human confidence
confidence = alpha - gamma / (1 + beta)

rng = np.random.default_rng()

# Create make_moons dataset and split training and test data setts
def get_Xy(dtype="moons"):
    if dtype=="moons":
        return make_moons(n_samples=10000, noise=0.3, random_state=1)

X, y = get_Xy()
df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
target = df.label
df.drop(["label"], axis=1, inplace=True)
y = np.array(target)
X = np.array(df, dtype=float)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=1)

DATA_SPLIT = {"train": 0.7, "test": 0.15, "val": 0.15}


class CustomDataset(Dataset):
    def __init__(self, data, classes=2):
        self.input = torch.tensor(data[0])
        self.targets = torch.tensor(data[1])
        self.eye = torch.eye(classes)

    def __getitem__(self, index):
        x = self.input[index].float()
        y = self.targets[index].float()
        # self.eye[self.targets[index]]
        return x, y

    def __len__(self):
        return len(self.input)


def load_datasets(X, Y):

    size = np.array(X).shape[0]
    permute = np.random.permutation(size)

    X_p = X[permute]
    Y_p = Y[permute]
    train_size = int(size * DATA_SPLIT["train"])
    val_size = int(size * DATA_SPLIT["val"])
    test_size = int(size * DATA_SPLIT["test"])
    train = X_p[:train_size], Y_p[:train_size]
    val = (
        X_p[train_size : train_size + val_size],
        Y_p[train_size : train_size + val_size],
    )
    test = X_p[train_size + val_size :], Y_p[train_size + val_size :]
    return train, val, test


# Function to calculate the team expected utility
def expected_team_utility_loss(pred, y):
    pos_pos_mask = torch.relu(pred - confidence) / (pred - confidence)
    neg_pos_mask = torch.relu(-pred + confidence) / (-pred + confidence)

    pos_y = y * (
        ((1 + beta) * pred - beta) * pos_pos_mask
        + ((1 + beta) * alpha - beta - gamma) * (neg_pos_mask)
    )

    npred = 1 - pred
    pos_neg_mask = torch.relu(npred - confidence) / (npred - confidence)
    neg_neg_mask = torch.relu(-npred + confidence) / (-npred + confidence)

    neg_y = (1 - y) * (
        ((1 + beta) * npred - beta) * (pos_neg_mask)
        + ((1 + beta) * alpha - beta - gamma) * (neg_neg_mask)
    )

    #print(-torch.sum((pos_y + neg_y)), pred, y)
    return -torch.sum((pos_y + neg_y))/len(pred)

class FeedForward(torch.nn.Module):
    def __init__(self, input_size, layer1_size, layer2_size):
        super().__init__()
        self.input_size = input_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.fc1 = torch.nn.Linear(self.input_size, self.layer1_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.layer1_size, self.layer2_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(self.layer2_size, 1)
    
    def forward(self, x):
        layer1 = self.fc1(x)
        relu1 = self.relu(layer1)
        layer2 = self.fc2(relu1)
        relu2 = self.relu(layer2)
        output = self.fc3(relu2)
        output = self.sigmoid(output)
        return output

class PLModel(LightningModule):
    def __init__(
        self, X, Y, input_size, layer1_size, layer2_size, train_type="initial"
    ):
        super().__init__()
        if train_type == "initial":
            self.loss = torch.nn.BCELoss()
        else:
            self.loss = expected_team_utility_loss

        self.input_size = input_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.model = FeedForward(input_size=input_size, layer1_size=layer1_size, layer2_size=layer2_size)

        self.train_set, self.val_set, self.test_set = load_datasets(X, Y)

    def load_model(self, path):
        path_dict = torch.load(path)["state_dict"]
        prefix = "model."
        state_dict = {
             key[len(prefix):] : path_dict[key] 
            for key in path_dict
        }
        self.model.load_state_dict(state_dict)
        print('Model Created!')

    def configure_optimizers(self):
        return torch.optim.SGD(model.parameters(), lr=0.1)

    def train_dataloader(self):
        return DataLoader(CustomDataset(self.train_set), batch_size=64)

    def val_dataloader(self):

        return DataLoader(CustomDataset(self.val_set), batch_size=64)

    def test_dataloader(self):
        return DataLoader(CustomDataset(self.test_set), batch_size=64)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1)
        predicted = self(x)
        loss = self.loss(predicted, y)
        self.log("test_loss", loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1)
        predicted = self(x)
        loss = self.loss(predicted, y)

        self.log("val_loss", loss)
        self.log("val_utility",  expected_team_utility_loss(predicted, y))
        return {
            "val_loss": loss,
            "val_y": y,
            "val_y_hat": predicted,
        }
    def validation_epoch_end(self, out):
        y_hat = torch.cat([out[i]["val_y_hat"] for i in range(len(out))])
        y = torch.cat([out[i]["val_y"] for i in range(len(out))])
        
        accuracy = FM.accuracy(y_hat, y.int())

        self.log("val_acc", accuracy)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1)
        predicted = self(x)
        loss = self.loss(predicted, y)
        return loss

    def forward(self, x):
        return self.model(x)


# Scale data around zero for Mulit-layered perceptron classifier
sc_X = StandardScaler()
# X_trainscaled = sc_X.fit_transform(X_train)
# X_valscaled = sc_X.transform(X_test)


seed_everything(42)


checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
    monitor="val_loss",
    dirpath="models_saved",
    filename="sst-{epoch:02d}-{val_utility:.2f}-{val_loss:.2f}-{val_acc:.2f}",
    mode="min",
)
model = PLModel(X, y, 2, 50, 10)
trainer = Trainer(max_epochs=100, callbacks=[checkpoint_callback])
trainer.fit(model)
print(checkpoint_callback.best_model_path)

# print(model.state_dict())
model = PLModel(X, y, 2, 50, 10, train_type="experiment")
model.load_model(
    checkpoint_callback.best_model_path,
)


checkpoint_callback_1 = pl.callbacks.model_checkpoint.ModelCheckpoint(
    monitor="val_loss",
    dirpath="models_saved",
    filename="sst2-{epoch:02d}-{val_utility:.2f}-{val_loss:.2f}-{val_acc:.2f}",
    mode="min",
)

trainer = Trainer(max_epochs=10000, callbacks=[checkpoint_callback_1])
trainer.fit(model)
print(checkpoint_callback_1.best_model_path)
