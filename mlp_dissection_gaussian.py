from ctypes import util
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

RATIONALITY = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Create make_moons dataset and split training and test data setts
def get_Xy(dtype="moons"):
    if dtype == "moons":
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
    def __init__(self, data, classes=2, human_rationality=1):
        self.input = torch.tensor(data[0])
        self.targets = torch.tensor(data[1])
        self.is_rational = torch.tensor(data[2])
        self.g_confidence = torch.tensor(data[3])

        self.eye = torch.eye(classes)
        self.human_rationality = human_rationality

    def __getitem__(self, index):
        x = self.input[index].float()
        y = self.targets[index].float()
        # assumes fixed irrationality
        is_rational = self.is_rational[index].float()
        g_confidence = self.g_confidence[index].float()
        # assumes the rational point 
        return x, y, is_rational, g_confidence

    def __len__(self):
        return len(self.input)


def load_datasets(X, Y, human_rationality=1, std=0.1):
    
    size = np.array(X).shape[0]
    g_confidence = torch.normal(mean=torch.ones(size) * confidence, std=torch.ones(size) * std)
    permute = np.random.permutation(size)
    a = torch.empty(size).uniform_(0, 1)
    is_rational = torch.ones(size) * (a < human_rationality)

    X_p = X[permute]
    Y_p = Y[permute]
    is_rational_p = is_rational[permute]

    train_size = int(size * DATA_SPLIT["train"])
    val_size = int(size * DATA_SPLIT["val"])

    train = X_p[:train_size], Y_p[:train_size], is_rational_p[:train_size], g_confidence[:train_size]
    val = (
        X_p[train_size : train_size + val_size],
        Y_p[train_size : train_size + val_size],
        is_rational_p[train_size : train_size + val_size],
        g_confidence[train_size : train_size + val_size]
    )
    test = (
        X_p[train_size + val_size :],
        Y_p[train_size + val_size :],
        is_rational_p[train_size + val_size :],
        g_confidence[train_size : train_size + val_size]
    )
    return train, val, test


# Function to calculate the team expected utility
def expected_team_utility_g_loss(pred, y, is_rational, g_confidence):
    g_confidence = g_confidence.view(-1, 1)
    pos_pos_mask = torch.relu(pred - g_confidence) / (pred - g_confidence)
    neg_pos_mask = torch.relu(-pred + g_confidence) / (-pred + g_confidence)

    pos_y = y * (
        ((1 + beta) * pred - beta) * pos_pos_mask
        + ((1 + beta) * alpha - beta - gamma) * (neg_pos_mask)
    )

    npred = 1 - pred
    pos_neg_mask = torch.relu(npred - g_confidence) / (npred - g_confidence)
    neg_neg_mask = torch.relu(-npred + g_confidence) / (-npred + g_confidence)

    neg_y = (1 - y) * (
        ((1 + beta) * npred - beta) * (pos_neg_mask)
        + ((1 + beta) * alpha - beta - gamma) * (neg_neg_mask)
    )
    is_rational = is_rational.view(-1, 1)
    irrational = torch.ones((len(pred), 1)) * (1 - is_rational) *  ((1 + beta) * alpha - beta - gamma)
    rational = ((pos_y + neg_y)) * (is_rational)
    #assert (torch.max(is_rational) <= 1)
    #assert (torch.max(pos_y + neg_y) <= 1)
    #assert (torch.max(rational + irrational) <= 1)
    #assert len(rational + irrational) == len(pred)
    return -torch.sum(rational + irrational)/len(pred)



# Function to calculate the team expected utility
def expected_team_utility_loss(pred, y, is_rational):
    #print(list(is_rational))

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
    is_rational = is_rational.view(-1, 1)
    irrational = torch.ones((len(pred), 1)) * (1 - is_rational) *  ((1 + beta) * alpha - beta - gamma)
    rational = ((pos_y + neg_y)) * (is_rational)
    assert (torch.max(is_rational) <= 1)
    assert (torch.max(pos_y + neg_y) <= 1)
    assert (torch.max(rational + irrational) <= 1)
    assert len(rational + irrational) == len(pred)
    return -torch.sum(rational + irrational)/len(pred)


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
        self,
        X,
        Y,
        input_size,
        layer1_size,
        layer2_size,
        train_type="initial",
        human_rationality=1,
        std=0.1
    ):
        super().__init__()
        self.train_type = train_type

        if train_type == "initial":
            self.loss = torch.nn.BCELoss()
        else:
            self.loss = expected_team_utility_loss

        self.input_size = input_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.model = FeedForward(
            input_size=input_size, layer1_size=layer1_size, layer2_size=layer2_size
        )

        self.train_set, self.val_set, self.test_set = load_datasets(
            X, Y, human_rationality=human_rationality, std=std
        )

    def load_model(self, path):
        path_dict = torch.load(path)["state_dict"]
        prefix = "model."
        state_dict = {key[len(prefix) :]: path_dict[key] for key in path_dict}
        self.model.load_state_dict(state_dict)
        print("Model Created!")

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def train_dataloader(self):
        return DataLoader(CustomDataset(self.train_set), batch_size=64)

    def val_dataloader(self):
        return DataLoader(CustomDataset(self.val_set), batch_size=64)

    def test_dataloader(self):
        return DataLoader(CustomDataset(self.test_set), batch_size=64)

    def test_step(self, batch, batch_idx):
        x, y, is_rational, g_confidence = batch
        y = y.view(-1, 1)
        predicted = self(x)
        if self.train_type == "initial":
            loss = self.loss(predicted, y)
        else:
            loss = self.loss(predicted, y, is_rational)
        self.log("test_loss", loss)

    def validation_step(self, batch, batch_idx):
        x, y, is_rational, g_confidence = batch
        y = y.view(-1, 1)
        predicted = self(x)
        if self.train_type == "initial":
            loss = self.loss(predicted, y)
        else:
            loss = self.loss(predicted, y, is_rational)

        self.log("val_loss", loss)
        utility = expected_team_utility_g_loss(predicted, y, is_rational, g_confidence)
        #print(utility)
        self.log("val_utility", utility)
        return {
            "val_loss": loss,
            "val_y": y,
            "val_y_hat": predicted,
            "is_rational": is_rational
        }

    def validation_epoch_end(self, out):
        y_hat = torch.cat([out[i]["val_y_hat"] for i in range(len(out))])
        y = torch.cat([out[i]["val_y"] for i in range(len(out))])
        is_rational = torch.cat([out[i]["is_rational"] for i in range(len(out))])
        val_loss = torch.tensor([out[i]["val_loss"] for i in range(len(out))])
        accuracy = FM.accuracy(y_hat, y.int())


        self.log("ep_val_loss", torch.mean(val_loss))
        self.log("val_acc", accuracy)

    def training_step(self, batch, batch_idx):
        x, y, is_rational, g_confidence = batch
        y = y.view(-1, 1)
        predicted = self(x)
        if self.train_type == "initial":
            loss = self.loss(predicted, y)
        else:
            loss = self.loss(predicted, y, is_rational)
        return loss

    def forward(self, x):
        return self.model(x)


# Scale data around zero for Mulit-layered perceptron classifier
sc_X = StandardScaler()
# X_trainscaled = sc_X.fit_transform(X_train)
# X_valscaled = sc_X.transform(X_test)




# print(model.state_dict())
def train_experiment_models():
    for std in [1e-3, 1e-2, 1e-1, 0.2, 0.5]:
        seed_everything(42)
        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            monitor="val_loss",
            dirpath=f"models_saved/experiments_gaussian/s-{std}/og",
            filename="sst-{epoch:02d}-{val_utility:.2f}-{ep_val_loss:.2f}-{val_acc:.2f}",
            mode="min",
        )
        model = PLModel(X, y, 2, 50, 10, human_rationality=1, std=std)
        trainer = Trainer(max_epochs=100, callbacks=[checkpoint_callback])
        trainer.fit(model)
        checkpt = checkpoint_callback.best_model_path
        print(checkpoint_callback.best_model_path)
        model = PLModel(X, y, 2, 50, 10, train_type="experiment", human_rationality=1, std=std)
        model.load_model(checkpt)
        checkpoint_callback_1 = pl.callbacks.model_checkpoint.ModelCheckpoint(
            monitor="val_loss",
            dirpath=f"models_saved/experiments_gaussian/s-{std}/exp",
            filename="sst-{epoch:02d}-{val_utility:.2f}-{ep_val_loss:.2f}-{val_acc:.2f}",
            mode="min",
        )
        trainer = Trainer(max_epochs=100, callbacks=[checkpoint_callback_1])
        trainer.fit(model)
        print(checkpoint_callback_1.best_model_path)


train_experiment_models()
