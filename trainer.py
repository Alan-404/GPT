import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics import Perplexity
from typing import Callable
from gpt import GPT
import os
from torch.utils.data import TensorDataset, DataLoader, Dataset
import math
from sklearn.model_selection import train_test_split
import mlflow
from tqdm import tqdm
import pandas as pd
import json
from preprocessing.data import Tokenizer


class GPTTrainer:
    def __init__(self,
                 token_size: int,
                 n: int = 12,
                 d_model: int = 768,
                 heads: int = 12,
                 d_ff: int = 3072,
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
                 dropout_rate: float = 0.1,
                 eps: float = 0.02,
                 device: str = 'cpu',
                 pad_value: int = 0,
                 learning_rate: float = 3e-5,
                 checkpoint: str = None) -> None:
        # Setup Model
        self.model = GPT(
            token_size=token_size,
            n=n,
            d_model=d_model,
            heads=heads,
            d_ff=d_ff,
            activation=activation,
            dropout_rate=dropout_rate,
            eps=eps
        )

        # Setup Optimizer and its Scheduler
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)

        # Declare Loss Function and Metric Function
        self.cost = GPTLoss()
        self.metric = Perplexity(ignore_index=pad_value)

        # Training Config
        self.device = device
        self.model.to(self.device)
        self.checkpoint = checkpoint
        self.epoch = 0

        # Loss and Score Statistical
        self.loss = 0.0
        self.score = 0.0
        
        # Loss and Score History in Train Set and Validation Set
        self.losses = []
        self.scores = []

        self.val_losses = []
        self.val_scores = []

        # Load Checkpoint if having
        if self.checkpoint is not None:
            self.load_model(self.checkpoint)
    
    def __save(self, path: str):
        torch.save({
            ModelInfo.MODEL_STATE_DICT: self.model.state_dict(),
            ModelInfo.OPTIMIZER_STATE_DICT: self.optimizer.state_dict(),
            ModelInfo.EPOCH: self.epoch,
            ModelInfo.LOSS: self.losses,
            ModelInfo.METRIC: self.scores,
            ModelInfo.VAL_LOSS: self.val_losses,
            ModelInfo.VAL_METRIC: self.val_scores
        }, path)
    
    def save_model(self, path: str):
        try:
            self.__save(path)
        except:
            print("Folder is not Found")
            self.__save("./model.pt")

    def __load(self, path: str, location: str):
        checkpoint = torch.load(path, location)
        self.model.load_state_dict(checkpoint[ModelInfo.MODEL_STATE_DICT])
        self.optimizer.load_state_dict(checkpoint[ModelInfo.OPTIMIZER_STATE_DICT])
        self.epoch = checkpoint[ModelInfo.EPOCH]
        self.losses = checkpoint[ModelInfo.LOSS]
        self.scores = checkpoint[ModelInfo.METRIC]
        self.val_losses = checkpoint[ModelInfo.VAL_LOSS]
        self.val_scores = checkpoint[ModelInfo.VAL_METRIC]
        checkpoint = None

    def load_model(self, path: str, location: str = None):
        if os.path.exists(path):
            self.__load(path, location)

    def build_dataset(self, tensor: torch.Tensor, batch_size: int):
        return DataLoader(dataset=TensorDataset(tensor), batch_size=batch_size, shuffle=True)
    
    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        # Feed Forward
        self.optimizer.zero_grad()
        outputs = self.model(inputs)

        # Back Propagation
        loss = self.cost(outputs, labels)
        loss.backward()
        self.optimizer.step()

        # Calculate Metrical Score
        _, preds = torch.max(outputs, dim=-1)
        score = self.metric(preds, labels)

        self.loss += loss.item()
        self.score += score

    def fit(self, train_dataset: Dataset, valid_dataset: Dataset = None, epochs: int = 1, batch_size: int = 1, epoch_save: int = None, tracking_config: dict = None):
        if epoch_save is not None:
            if epoch_save > epochs:
                epoch_save = epochs
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

        num_batches = len(train_dataloader)

        for _ in range(epochs):
            print(f"======================{self.epoch + 1}======================")
            for index, data in tqdm(enumerate(train_dataloader)):
                inputs = data[0][:-1].to(self.device)
                labels = data[0][1:].to(self.device)

                self.train_step(inputs, labels)
            
            print(f"Epoch {self.epoch + 1}: Train Loss: {(self.loss/num_batches):.4f} Train Score: {(self.metric/num_batches):.4f}")

    
    def validate_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        outputs = self.model(inputs)

        loss = self.cost(outputs, labels).item()

        _, preds = torch.max(outputs, dim=-1)
        score = self.metric(preds, labels)

        return loss, score
        
    def validate(self, dataloader: DataLoader, tracking: bool = False):
        total_batches = len(dataloader)
        total_loss = 0.0
        total_score = 0.0
        for _, data in enumerate(dataloader, 0):
            inputs = data[0][:, :-1].to(self.device)
            labels = data[0][:, 1:].to(self.device)

            loss, score = self.validate_step(inputs, labels)
            total_loss += loss
            total_score += score
        
        if self.model.training:
            print(f"Epoch: {self.epoch+1} Validation Loss: {(total_loss/total_batches):.4f} Validation Score: {(total_score/total_batches):.4f}")
            self.val_losses.append(total_loss/total_batches)
            self.val_scores.append(total_score/total_batches)

            if tracking:
                mlflow.log_metric("Validation Loss", total_loss/total_batches, step=self.epoch)
                mlflow.log_metric("Validation BLEU Score", total_score/total_batches, step=self.epoch)
        else:
            print(f"Evaluation Result: Loss: {(total_loss/total_batches):.4f} Score: {(total_score/total_batches):.4f}")
            if tracking:
                mlflow.log_metric("Test Loss", total_loss/total_batches, step=self.epoch)
                mlflow.log_metric("Test Score", total_score/total_batches, step=self.epoch)
        

    def evaluate(self, data: torch.Tensor, batch_size: int = 1):
        self.model.eval()
        dataloader = self.build_dataset(data, batch_size=batch_size)
        self.validate(dataloader)


    def generate(self, text: torch.Tensor, max_length: int, end_token: int):
        self.model.eval()
        for _ in range(max_length):
            output = self.model(text)
            _, pred = torch.max(output[:, -1, :], dim=-1)

            if pred == end_token:
                break

            text = torch.cat((text, pred.unsqueeze(0)), dim=-1)
        return text


class GPTDataset(Dataset):
    def __init__(self, manifest_path: str, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.prompts = pd.read_csv(manifest_path, sep="\t")

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index: int):
        digits = self.tokenizer.text2sequence(self.prompts.loc[index]['input'], self.prompts.loc[index]['output'])
        return digits

class GPTLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        batch_size = labels.size(0)
        loss = 0.0
        for batch in range(batch_size):
            loss += self.criterion(outputs[batch], labels[batch])
        loss = loss / batch_size

        return loss


activation_functions_dict = {
    "gelu": F.gelu,
    "relu": F.relu,
    "tanh": torch.tanh,
    "selu": F.selu
}

optimizer_dict = {
    'adam': optim.Adam,
    'ada': optim.Adagrad
}


class ModelInfo:
    MODEL_STATE_DICT = 'model_state_dict'
    OPTIMIZER_STATE_DICT = 'optimizer_state_dict'
    SCHEDULER_STATE_DICT= 'scheduler_state_dict'
    EPOCH = 'epoch'
    LOSS = 'loss'
    METRIC = 'metric'
    VAL_LOSS = 'val_loss'
    VAL_METRIC = 'val_metric'