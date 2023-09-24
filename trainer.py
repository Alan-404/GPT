import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics import Perplexity
from typing import Callable
from gpt import GPT
import os
from torch.utils.data import TensorDataset, DataLoader
import math
from sklearn.model_selection import train_test_split
import mlflow
from tqdm import tqdm


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

    def train(self, dataloader: DataLoader, epochs: int, mini_batch: int, val_dataloader: DataLoader = None, tracking: bool = False):
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

            if tracking:
                mlflow.log_metric("Train Loss", epoch_loss/total_batches, step=self.epoch)
                mlflow.log_metric("Train BLEU Score", epoch_score/total_batches, step=self.epoch)

            if val_dataloader is not None:
                self.validate(val_dataloader)
            
            epoch_loss = 0.0
            epoch_score = 0.0
            self.epoch += 1

            
            # Step Scheduler to change learning rate
            # self.scheduler.step()

    def fit(self, data: torch.Tensor, epochs: int = 1, batch_size: int = 1, mini_batch: int = 1, **validation):
        tracking = False
        if 'tracking' in validation and validation['tracking'] == True:
            tracking = True
            if 'tracking_uri' in validation and validation['tracking_uri'] is not None:
                mlflow.set_tracking_uri(validation['tracking_uri'])
            
            if 'experiment_name' in validation and validation['experiment_name'] is not None:
                mlflow.set_experiment(validation['experiment_name'])
            else:
                mlflow.set_experiment("GPT Model")
            
            if "run_id" in validation and validation['run_id'] is not None:
                mlflow.start_run(validation['run_id'])
            elif 'run_name' in validation and validation['run_name'] is not None:
                mlflow.start_run(run_name=validation['run_name'])
            else:
                mlflow.start_run(run_name="Version 1")

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

                self.train(train_dataloader, epochs=epochs, mini_batch=mini_batch, val_dataloader=val_dataloader, tracking=tracking)
            elif validation['val_type'] == 'kfold':
                num_folds = 1
                if 'num_folds' in validation and type(validation['num_folds'] == 'int'):
                    num_folds = validation['num_folds']
                assert epochs >= num_folds

                epochs = epochs // num_folds
                
                num_per_fold = math.ceil(data.size(0)/num_folds)

                for fold in range(num_folds):
                    val_start_idx = fold * num_per_fold
                    val_end_idx = (fold + 1) * num_per_fold

                    val_set = data[val_start_idx:val_end_idx, :]
                    train_set = torch.cat((data[0:val_start_idx, :], data[val_end_idx:, :]), dim=0)


                    train_dataloader = self.build_dataset(train_set, batch_size=batch_size)
                    val_dataloader = self.build_dataset(val_set, batch_size=batch_size)
                    self.train(train_dataloader, epochs=epochs, mini_batch=mini_batch, val_dataloader=val_dataloader, tracking=tracking)
        # No Validation
        else:
            dataloader = self.build_dataset(data, batch_size=batch_size)
            self.train(dataloader, epochs=epochs, mini_batch=mini_batch, tracking=tracking)

        # Save Checkpoint
        if self.checkpoint is not None:
            self.save_model(self.checkpoint)
        else:
            self.save_model("./model.pt")

        if tracking:
            mlflow.pytorch.log_model(self.model, "model")
    
    
    def validate_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        outputs = self.model(inputs)

        loss = self.cost(outputs, labels).item()

        _, preds = torch.max(outputs, dim=-1)
        score = self.metric.bleu_score(preds, labels)

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


class GPTLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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