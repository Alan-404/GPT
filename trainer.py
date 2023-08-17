import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Callable
from gpt import GPT
import os
from metric import GPTMetric
from torch.utils.data import TensorDataset, DataLoader
from loss import GPTLoss
import math
from sklearn.model_selection import train_test_split
import mlflow

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
                 optim_betas: tuple[float, float] = [0.9, 0.999],
                 optim_eps: float = 1e-8,
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
        # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

        # Declare Loss Function and Metric Function
        self.cost = GPTLoss()
        self.metric = GPTMetric()

        # Training Config
        self.device = device
        self.model.to(self.device)
        self.checkpoint = checkpoint
        self.epoch = 0
        
        # Loss and Score History in Train Set and Validation Set
        self.losses = []
        self.scores = []

        self.val_losses = []
        self.val_scores = []

        # Load Checkpoint if having
        if self.checkpoint is not None:
            self.load_model(self.checkpoint)
    
    def save_model(self, path: str):
        try:
            torch.save({
                ModelInfo.MODEL_STATE_DICT: self.model.state_dict(),
                ModelInfo.OPTIMIZER_STATE_DICT: self.optimizer.state_dict(),
                # ModelInfo.SCHEDULER_STATE_DICT: self.scheduler.state_dict(),
                ModelInfo.EPOCH: self.epoch,
                ModelInfo.LOSS: self.losses,
                ModelInfo.METRIC: self.scores,
                ModelInfo.VAL_LOSS: self.val_losses,
                ModelInfo.VAL_METRIC: self.val_scores
            }, path)
        except:
            print("Folder is not Found")
            torch.save({
                ModelInfo.MODEL_STATE_DICT: self.model.state_dict(),
                ModelInfo.OPTIMIZER_STATE_DICT: self.optimizer.state_dict(),
                # ModelInfo.SCHEDULER_STATE_DICT: self.scheduler.state_dict(),
                ModelInfo.EPOCH: self.epoch,
                ModelInfo.LOSS: self.losses,
                ModelInfo.METRIC: self.scores,
                ModelInfo.VAL_LOSS: self.val_losses,
                ModelInfo.VAL_METRIC: self.val_scores
            }, './model.pt')

    def load_model(self, path: str, location: str = None):
        if os.path.exists(path):
            checkpoint = torch.load(path, location)

            self.model.load_state_dict(checkpoint[ModelInfo.MODEL_STATE_DICT])
            self.optimizer.load_state_dict(checkpoint[ModelInfo.OPTIMIZER_STATE_DICT])
            # self.scheduler.load_state_dict(checkpoint[ModelInfo.SCHEDULER_STATE_DICT])
            self.epoch = checkpoint[ModelInfo.EPOCH]
            self.losses = checkpoint[ModelInfo.LOSS]
            self.scores = checkpoint[ModelInfo.METRIC]
            self.val_losses = checkpoint[ModelInfo.VAL_LOSS]
            self.val_scores = checkpoint[ModelInfo.VAL_METRIC]

            checkpoint = None

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
        score = self.metric.bleu_score(preds, labels)

        return loss.item(), score

    def train(self, dataloader: DataLoader, epochs: int, mini_batch: int, val_dataloader: DataLoader = None):
        # Beginning Config
        total_batches = len(dataloader)
        batch_loss = 0.0
        batch_score = 0.0

        epoch_loss = 0.0
        epoch_score = 0.0
        # Start Tranining
        for _ in range(epochs):
            count = 0
            # Handle per epoch
            for index, data in enumerate(dataloader, 0):
                # Get Input and Corresponding Label
                inputs = data[0][:, :-1].to(self.device)
                labels = data[0][:, 1:].to(self.device)
                # Train in Batch
                loss, score = self.train_step(inputs, labels)
                batch_loss += loss
                batch_score += score
                count += 1
                # Statistical
                if index%mini_batch == mini_batch-1 or index == total_batches-1:
                    print(f"Epoch: {(self.epoch+1)} Batch: {index + 1} Loss: {(batch_loss/count):.4f} Score: {(batch_score/count):.4f}")

                    epoch_loss += batch_loss
                    epoch_score += batch_score

                    batch_loss = 0.0
                    batch_score = 0.0
                    count = 0
            # Finish an Epoch
            print(f"Epoch: {self.epoch+1} Train Loss: {(epoch_loss/total_batches):.4f} Train Score: {(epoch_score/total_batches):.4f}")
            self.losses.append(epoch_loss/total_batches)
            self.scores.append(epoch_score/total_batches)

            epoch_loss = 0.0
            epoch_score = 0.0
            self.epoch += 1

            if val_dataloader is not None:
                self.validate(val_dataloader)

            # Step Scheduler to change learning rate
            # self.scheduler.step()

    def fit(self, data: torch.Tensor, epochs: int = 1, batch_size: int = 1, mini_batch: int = 1, **validation):
        # Set model in Training mode
        self.model.train()

        use_validate = len(validation.keys()) != 0 and 'val_type' in validation and validation['val_type'] is not None
        # Validation
        if use_validate:
            if validation['val_type'] == 'holdout':
                test_size = 0.2
                if 'val_size' in validation and type(validation['val_size']) == 'float':
                    test_size = validation['val_size']
                # Split data into 2 parts
                train_set, val_set = train_test_split(data, test_size=test_size, random_state=41)
                # Train and Validate
                train_dataloader = self.build_dataset(train_set, batch_size=batch_size)
                val_dataloader = self.build_dataset(val_set, batch_size=batch_size)

                self.train(train_dataloader, epochs=epochs, mini_batch=mini_batch, val_dataloader=val_dataloader)
            elif validation['val_type'] == 'kfold':
                num_folds = 1
                if 'num_folds' in validation and type(validation['num_folds'] == 'int'):
                    num_folds = validation['num_folds']
                assert epochs > num_folds

                epochs = epochs // num_folds
                num_per_fold = math.ceil(data.size(0)/num_folds)

                for fold in range(num_folds):
                    val_start_idx = fold * num_per_fold
                    val_end_idx = (fold + 1) * num_per_fold

                    val_set = data[val_start_idx:val_end_idx, :]
                    train_set = torch.cat((data[0:val_start_idx, :], data[val_end_idx:, :]), dim=0)


                    train_dataloader = self.build_dataset(train_set, batch_size=batch_size)
                    val_dataloader = self.build_dataset(val_set, batch_size=batch_size)
                    self.train(train_dataloader, epochs=epochs, mini_batch=mini_batch, val_dataloader=val_dataloader)
        # No Validation
        else:
            dataloader = self.build_dataset(data, batch_size=batch_size)
            self.train(dataloader, epochs=epochs, mini_batch=mini_batch)

        # Save Checkpoint
        if self.checkpoint is not None:
            self.save_model(self.checkpoint)
        else:
            self.save_model("./model.pt")
    
    
    def validate_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        outputs = self.model(inputs)

        loss = self.cost(outputs, labels).item()

        _, preds = torch.max(outputs, dim=-1)
        score = self.metric.bleu_score(preds, labels)

        return loss, score
        
    def validate(self, dataloader: DataLoader):
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
            print(f"Epoch: {self.epoch} Validation Loss: {(total_loss/total_batches):.4f} Validation Score: {(total_score/total_batches):.4f}")
            self.val_losses.append(total_loss/total_batches)
            self.val_scores.append(total_score/total_batches)
        else:
            print(f"Evaluation Result: Loss: {(total_loss/total_batches):.4f} Score: {(total_score/total_batches):.4f}")
        

    def evaluate(self, data: torch.Tensor, batch_size: int = 1):
        self.model.eval()
        dataloader = self.build_dataset(data, batch_size=batch_size)
        self.validate(dataloader)


@staticmethod
def generate(model: nn.Module, text: torch.Tensor, max_length: int, end_token: int):
    model.eval()
    for _ in range(max_length):
        output = model(text)
        _, pred = torch.max(output[:, -1, :], dim=-1)

        if pred == end_token:
            break

        text = torch.cat((text, pred.unsqueeze(0)), dim=-1)
    return text

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