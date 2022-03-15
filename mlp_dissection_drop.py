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
# 0 for expected, 1 for emperical
loss_type = int(sys.argv[4])
# 0 for no, 1 for yes
use_dropout = int(sys.argv[5])


# Calculate human confidence
confidence = alpha - gamma / (1 + beta)

rng = np.random.default_rng()

RATIONALITY = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Create make_moons dataset and split training and test data setts
def get_Xy(dtype="moons"):
    if dtype == "moons":
        return make_moons(n_samples=10000, noise=0.3, random_state=1)
    if dtype == "german":
        Path = "datasets/german_credit.csv"
        Data = pd.read_csv(Path)
       
        X_ = np.array(Data.drop(columns=["Creditability"]))
        y_ = np.array(Data["Creditability"])
        return X_, y_
    if dtype == "FICO":
        Path = "datasets/FICO_RiskData.csv"
        
        Data = pd.read_csv(Path)
        print(Data.columns)
        Data.dropna()

        X_ = np.array(Data.drop(columns=["Risk_Flag"]))
        y_ = np.array(Data["Risk_Flag"])
        return X_, y_


dtype = "FICO"
X, y = get_Xy(dtype)
if dtype == "moons:":
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    target = df.label
    df.drop(["label"], axis=1, inplace=True)
    y = np.array(target)
    X = np.array(df, dtype=float)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=1)

DATA_SPLIT = {"train": 0.7, "test": 0.15, "val": 0.15}


def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def get_monte_carlo_predictions(
    data_loader, forward_passes, model, n_classes, n_samples
):
    """Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """

    dropout_predictions = np.empty((0, n_samples, n_classes))
    softmax = nn.Softmax(dim=1)
    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        model.eval()
        enable_dropout(model)
        for i, (image, label) in enumerate(data_loader):

            image = image.to(torch.device("cuda"))
            with torch.no_grad():
                output = model(image)
                output = softmax(output)  # shape (n_samples, n_classes)
            predictions = np.vstack((predictions, output.cpu().numpy()))

        dropout_predictions = np.vstack(
            (dropout_predictions, predictions[np.newaxis, :, :])
        )
        # dropout predictions - shape (forward_passes, n_samples, n_classes)

    # Calculating mean across multiple MCD forward passes
    mean = torch.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes
    variance = torch.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)
    epsilon = sys.float_info.min


class CustomDataset(Dataset):
    def __init__(self, data, classes=2, human_rationality=1):
        self.input = torch.tensor(data[0])
        self.targets = torch.tensor(data[1])
        self.is_rational = torch.tensor(data[2])
        self.human_effort = torch.tensor(data[3])

        self.eye = torch.eye(classes)
        self.human_rationality = human_rationality

    def __getitem__(self, index):
        x = self.input[index].float()
        y = self.targets[index].float()
        # assumes fixed irrationality
        is_rational = self.is_rational[index].float()
        human_effort = self.human_effort[index].float()
        return x, y, is_rational, human_effort

    def __len__(self):
        return len(self.input)


DATA_RATIOS = {"moon": [0.5, 0.5], "german": [0.3, 0.7], "FICO": [0.52, 0.48]}


def load_datasets(X, Y, human_rationality=1):

    size = np.array(X).shape[0]
    permute = np.random.permutation(size)
    a = torch.empty(size).uniform_(0, 1)
    is_rational = torch.ones(size) * (a < human_rationality)

    human_correctness = torch.empty(size).uniform_(0, 1)
    human_effort = ((1 - gamma) * torch.ones(size) * (human_correctness <= alpha)) + (
        (-beta - gamma) * torch.ones(size) * (human_correctness > alpha)
    )

    X_p = X[permute]
    Y_p = Y[permute]
    is_rational_p = is_rational[permute]

    train_size = int(size * DATA_SPLIT["train"])
    val_size = int(size * DATA_SPLIT["val"])

    train = (
        X_p[:train_size],
        Y_p[:train_size],
        is_rational_p[:train_size],
        human_effort[:train_size],
    )
    val = (
        X_p[train_size : train_size + val_size],
        Y_p[train_size : train_size + val_size],
        is_rational_p[train_size : train_size + val_size],
        human_effort[train_size : train_size + val_size],
    )
    test = (
        X_p[train_size + val_size :],
        Y_p[train_size + val_size :],
        is_rational_p[train_size + val_size :],
        human_effort[train_size + val_size :],
    )
    return train, val, test


def emperical_team_utility_loss(pred, y, is_rational, human_effort):
    human_effort = human_effort.view(-1, 1)
    is_rational = is_rational.view(-1, 1)

    pos_pos_mask = torch.relu(pred - confidence) / (pred - confidence)
    neg_pos_mask = torch.relu(-pred + confidence) / (-pred + confidence)

    pred_pos = torch.relu(pred - 0.5) / (pred - 0.5)
    pred_neg = torch.relu(-pred + 0.5) / (-pred + 0.5)

    pos_y = y * (pred_pos - beta * pred_neg)
    neg_y = (1 - y) * (pred_neg - beta * pred_pos)

    irrational = human_effort * (1 - is_rational)
    rational = ((pos_y + neg_y)) * (is_rational)
    # print((neg_pos_mask * human_effort).shape)
    # print((pos_pos_mask * (pos_y + neg_y)).shape)
    return -torch.sum(
        neg_pos_mask * human_effort + pos_pos_mask * (rational + irrational)
    ) / len(pred)


# Function to calculate the team expected utility
def expected_team_utility_loss(pred, y, is_rational, human_effort):
    # print(list(is_rational))

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
    irrational = (
        torch.ones((len(pred), 1))
        * (1 - is_rational)
        * ((1 + beta) * alpha - beta - gamma)
    )
    rational = ((pos_y + neg_y)) * (is_rational)
    assert torch.max(is_rational) <= 1
    assert torch.max(pos_y + neg_y) <= 1
    assert torch.max(rational + irrational) <= 1
    assert len(rational + irrational) == len(pred)
    return -torch.sum(rational + irrational) / len(pred)


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


def BCELoss_class_weighted(weights):
    def loss(input, target):
        input = torch.clamp(input, min=1e-7, max=1 - 1e-7)
        bce = -weights[1] * target * torch.log(input) - (1 - target) * weights[
            0
        ] * torch.log(1 - input)
        return torch.mean(bce)

    return loss


class FeedForward(torch.nn.Module):
    def __init__(self, input_size, layer1_size, layer2_size, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.fc1 = torch.nn.Linear(self.input_size, self.layer1_size)
        self.drop1 = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(self.layer1_size, self.layer2_size)
        self.drop2 = torch.nn.Dropout(dropout)
        self.sigmoid = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(self.layer2_size, 1)

    def forward(self, x):
        layer1 = self.drop1(self.fc1(x))
        relu1 = self.relu(layer1)
        layer2 = self.drop2(self.fc2(relu1))
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
    ):
        super().__init__()
        self.train_type = train_type

        if train_type == "initial":
            self.loss = BCELoss_class_weighted(weights=DATA_RATIOS[dtype])
        elif loss_type == 0:
            self.loss = expected_team_utility_loss
        else:
            self.loss = emperical_team_utility_loss

        self.input_size = input_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size

        self.model = FeedForward(
            input_size=input_size, layer1_size=layer1_size, layer2_size=layer2_size
        )
        enable_dropout(self.model)

        self.train_set, self.val_set, self.test_set = load_datasets(
            X, Y, human_rationality=human_rationality
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

        x, y, is_rational, human_effort = batch
        y = y.view(-1, 1)
        predicted = self(x)
        if self.train_type == "initial":
            loss = self.loss(predicted, y)
        else:
            loss = self.loss(predicted, y, is_rational, human_effort)
        self.log("test_loss", loss)
        if loss_type == 0:
            utility = expected_team_utility_loss(
                predicted, y, is_rational, human_effort
            )
        else:
            utility = emperical_team_utility_loss(
                predicted, y, is_rational, human_effort
            )
        # print(utility)
        self.log("test_utility", utility)
        return {
            "test_loss": loss,
            "test_y": y,
            "test_y_hat": predicted,
            "is_rational": is_rational,
        }

    def validation_step(self, batch, batch_idx):
        x, y, is_rational, human_effort = batch
        y = y.view(-1, 1)
        predicted = self(x)
        if self.train_type == "initial":
            loss = self.loss(predicted, y)
        else:
            loss = self.loss(predicted, y, is_rational, human_effort)

        self.log("val_loss", loss)
        if loss_type == 0:
            utility = expected_team_utility_loss(
                predicted, y, is_rational, human_effort
            )
        else:
            utility = emperical_team_utility_loss(
                predicted, y, is_rational, human_effort
            )
        # print(utility)
        self.log("val_utility", utility)
        return {
            "val_loss": loss,
            "val_y": y,
            "val_y_hat": predicted,
            "is_rational": is_rational,
        }

    def test_epoch_end(self, out):
        y_hat = torch.cat([out[i]["test_y_hat"] for i in range(len(out))])
        y = torch.cat([out[i]["test_y"] for i in range(len(out))])
        is_rational = torch.cat([out[i]["is_rational"] for i in range(len(out))])
        val_loss = torch.tensor([out[i]["test_loss"] for i in range(len(out))])
        accuracy = FM.accuracy(y_hat, y.int())

        self.log("ep_test_loss", torch.mean(val_loss))
        self.log("test_acc", accuracy)

    def validation_epoch_end(self, out):
        y_hat = torch.cat([out[i]["val_y_hat"] for i in range(len(out))])
        y = torch.cat([out[i]["val_y"] for i in range(len(out))])
        is_rational = torch.cat([out[i]["is_rational"] for i in range(len(out))])
        val_loss = torch.tensor([out[i]["val_loss"] for i in range(len(out))])
        accuracy = FM.accuracy(y_hat, y.int())

        self.log("ep_val_loss", torch.mean(val_loss))
        self.log("val_acc", accuracy)

    def training_step(self, batch, batch_idx):
        x, y, is_rational, human_effort = batch
        y = y.view(-1, 1)
        predicted = self(x)
        if self.train_type == "initial":
            loss = self.loss(predicted, y)
        else:
            loss = self.loss(predicted, y, is_rational, human_effort)
        return loss

    def forward(self, x):
        v = []

        if self.train_type == "initial" or not use_dropout:
            return self.model(x)

        for i in range(1000):
            v.append(self.model(x))

        return torch.mean(torch.stack(v), dim=0)

        return self.model(x)


# Scale data around zero for Mulit-layered perceptron classifier
sc_X = StandardScaler()
# X_trainscaled = sc_X.fit_transform(X_train)
# X_valscaled = sc_X.transform(X_test)

hparams = {"moon": [50, 10], "german": [1000, 100], "FICO": [100, 100]}
# print(model.state_dict())
def train_experiment_models():
    for r in [1]:
        seed_everything(42)

        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            monitor="val_loss",
            dirpath=f"models_saved/{dtype}/experiments/moon_drop/{loss_type}/d-{use_dropout}/og",
            filename="sst-{epoch:02d}-{val_utility:.3f}-{ep_val_loss:.3f}-{val_acc:.3f}",
            mode="min",
        )
        print(X.shape)
        model = PLModel(
            X,
            y,
            X[0].shape[0],
            hparams[dtype][0],
            hparams[dtype][1],
            human_rationality=r,
        )
        trainer = Trainer(max_epochs=300, callbacks=[checkpoint_callback])
        trainer.fit(model)
        checkpt = checkpoint_callback.best_model_path
        print(checkpoint_callback.best_model_path)
        model = PLModel(
            X,
            y,
            X[0].shape[0],
            hparams[dtype][0],
            hparams[dtype][1],
            train_type="experiment",
            human_rationality=r,
        )
        model.load_model(checkpt)
        checkpoint_callback_1 = pl.callbacks.model_checkpoint.ModelCheckpoint(
            monitor="val_loss",
            dirpath=f"models_saved/{dtype}/experiments/moon_drop/{loss_type}/d-{use_dropout}/experiment",
            filename="sst-{epoch:02d}-{val_utility:.3f}-{ep_val_loss:.3f}-{val_acc:.3f}",
            mode="min",
        )
        trainer = Trainer(max_epochs=300, callbacks=[checkpoint_callback_1])
        trainer.fit(model)
        print(checkpoint_callback_1.best_model_path)


def test_experiment_models(checkpt):
    for r in [1]:
        seed_everything(42)
        model = PLModel(
            X, y, X[0].shape[0], 50, 10, train_type="experiment", human_rationality=r
        )
        model.load_model(checkpt)
        trainer = Trainer(max_epochs=300)
        trainer.test(model=model, ckpt_path=checkpt)


train_experiment_models()

"""
exp_paths = {
    "drop_1_exp": "/Users/shreya/Desktop/15780/dissection_1/models_saved/experiments/moon_drop_1/0/r-1/experiment/sst-epoch=298-val_utility=-0.928-ep_val_loss=-0.927-val_acc=0.869.ckpt",
    "drop_all_exp": "/Users/shreya/Desktop/15780/dissection_1/models_saved/new/experiments/moon_drop_1/0/d-1/experiment/sst-epoch=56-val_utility=-0.927-ep_val_loss=-0.926-val_acc=0.881.ckpt",
    "drop_1_og": "/Users/shreya/Desktop/15780/dissection_1/models_saved/experiments/moon_drop_1/0/r-1/og/sst-epoch=105-val_utility=-0.857-ep_val_loss=0.241-val_acc=0.905.ckpt",
    "drop_all_og": "/Users/shreya/Desktop/15780/dissection_1/models_saved/new/experiments/moon_drop_1/0/d-1/og/sst-epoch=299-val_utility=-0.836-ep_val_loss=0.236-val_acc=0.907.ckpt"
}
test_experiment_models(exp_paths["drop_all_exp"])
"""
