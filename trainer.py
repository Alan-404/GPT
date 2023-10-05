import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
from typing import Callable
from gpt import GPT
import os
from torch.utils.data import DataLoader, Dataset
import mlflow
from tqdm import tqdm
import pandas as pd
from preprocessing.tokenizer import Tokenizer
import numpy as np
import yaml
from yaml.loader import SafeLoader


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
                 init_learning_rate: float = 3e-5,
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
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=init_learning_rate)

        # Declare Loss Function and Metric Function
        self.cost = GPTLoss()
        self.metric = GPTMetric()

        self.pad_value = pad_value

        # Training Config
        self.device = device
        self.model.to(self.device)
        self.checkpoint = checkpoint
        self.epoch = 0

        # Loss and Score Statistical
        self.loss = 0.0
        self.score = {
            MetricInfo.BLEU_SCORE: 0.0,
            MetricInfo.PERPLEXITY_SCORE: 0.0
        }
        
        # Loss and Score History in Train Set and Validation Set
        self.losses = []
        self.scores = []

        self.val_losses = []
        self.val_scores = []

        # Load Checkpoint if having
        if self.checkpoint is not None:
            self.load_model(self.checkpoint)
        else:
            self.checkpoint = "./model.pt"
    
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

    def __load_config(self, path: str):
        with open(path, 'r') as file:
            return yaml.load(file, Loader=SafeLoader)

    def load_config(self, path: str):
        if os.path.exists(path):
            return self.__load_config(path)
        return None
    
    def __save_config(self, config: dict, path: str):
        with open(path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

    def save_config(self, config: dict, path: str):
        if os.path.exists(path):
            self.__save_config(config, path)

    def load_model(self, path: str, location: str = None):
        if os.path.exists(path):
            self.__load(path, location)

    def freeze_pretrain(self):
        self.model.decoder.requires_grad_(False)

    def show_info_config(self, info: dict, name: str):
        print(f"============== {name} Information ==============")
        for key in info.keys():
            print(f"\t{key}: {info[key]}")
        print("\n")

    def build_dataloader(self, dataset: Dataset, batch_size: int):
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: get_batch_with_padding(batch, self.pad_value))
    
    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        # Feed Forward
        self.optimizer.zero_grad()
        outputs = self.model(inputs)

        # Back Propagation
        loss = self.cost(outputs, labels)
        loss.backward()
        self.optimizer.step()

        score = self.metric(outputs, labels)

        self.loss += loss.item()
        self.score[MetricInfo.BLEU_SCORE] += score[MetricInfo.BLEU_SCORE]
        self.score[MetricInfo.PERPLEXITY_SCORE] += score[MetricInfo.PERPLEXITY_SCORE]


    def fit(self, train_dataset: Dataset, val_dataset: Dataset = None, epochs: int = 1, batch_size: int = 1, epoch_save: int = None, step_save: int = None, validation_config_path: str = None, tracking_config_path: str = None):
        validation_config = None
        if validation_config_path is not None:
            validation_config = self.load_config(validation_config_path)

        tracking_config = None
        if tracking_config_path is not None:
            tracking_config = self.load_config(tracking_config_path)
            self.show_info_config(tracking_config, "Tracking")
        
        save_strategy = None
        if epoch_save is not None:
            save_strategy = "epoch"
        elif step_save is not None:
            save_strategy = "step"
        
        train_dataloader = self.build_dataloader(train_dataset, batch_size)
        num_batches = len(train_dataloader)
        
        num_save = 0
        if save_strategy is not None:
            if save_strategy == "step":
                if step_save > num_batches:
                    num_save = num_batches
            elif save_strategy == "epoch":
                if epoch_save > epochs:
                    num_save = epochs

        if tracking_config is not None:
            mlflow.set_tracking_uri(tracking_config['tracking_uri'])
            if tracking_config['experiment_name'] is None:
                tracking_config['experiment_name'] = 'GPT'
            mlflow.set_experiment(tracking_config['experiment_name'])

            if tracking_config['run_id'] is not None:
                mlflow.start_run(run_id=tracking_config['run_id'])
            else:
                if tracking_config['run_name'] is None:
                    tracking_config['run_name'] = "Version 1"
                mlflow.start_run(run_name=tracking_config['run_name'])

        self.model.train()
        for epoch in range(epochs):
            print(f"====================== Epoch {self.epoch + 1} ======================")
            for index, data in enumerate(tqdm(train_dataloader)):
                inputs = data[:, :-1].to(self.device)
                labels = data[:, 1:].to(self.device)

                self.train_step(inputs, labels)
                if save_strategy is not None and save_strategy == "step" and index%num_save == num_save+1:
                    self.save_model(self.checkpoint)

            train_loss = self.loss/num_batches
            bleu_score = self.score[MetricInfo.BLEU_SCORE]/num_batches
            perplexity_score = self.score[MetricInfo.PERPLEXITY_SCORE]/num_batches
            print("=========== Training Statistical ===========")
            print(f"Epoch {self.epoch + 1}: \n\tTrain Loss: {(train_loss):.4f} \n\tTrain BLEU Score: {(bleu_score):.4f} \n\tTrain Perplexity Score: {(perplexity_score):.4f}")
            self.loss = 0.0
            self.score = {
                MetricInfo.BLEU_SCORE: 0.0,
                MetricInfo.PERPLEXITY_SCORE: 0.0
            }

            if tracking_config is not None:
                mlflow.log_metric(f"Train Loss", train_loss, self.epoch)
                mlflow.log_metric(f"Train BLEU Score", bleu_score, self.epoch)
                mlflow.log_metric(f"Train Perplexity Score", perplexity_score, self.epoch)

            if val_dataset is not None:
                val_batch_size = batch_size
                if validation_config is not None and validation_config['batch_size'] is not None:
                    val_batch_size = validation_config['batch_size']
                val_loss, val_score = self.validate(val_dataset, batch_size=val_batch_size)
                if tracking_config is not None:
                    mlflow.log_metric(f"Validate Loss", val_loss, self.epoch)
                    mlflow.log_metric(f"Validate Score", val_score, self.epoch)
            
            if save_strategy is not None and save_strategy == "epoch" and epoch%num_save == num_save + 1:
                self.save_model(self.checkpoint)
            
            print(f"====================== Done Epoch {self.epoch + 1} ======================")
            print("\n")
            self.epoch += 1
        
        print(f"====================== Done Training ======================")
        print("\n")
        
        
        self.save_model(self.checkpoint)
        print(f"Model is saved at {self.checkpoint}")

        if tracking_config_path is not None:
            mlflow.pytorch.log_model(self.model, artifact_path="model")
            mlflow.pytorch.log_state_dict(self.model.state_dict(), artifact_path="model_state_dict")
            mlflow.pytorch.log_state_dict(self.optimizer.state_dict(), artifact_path="optimizer_state_dict")
    
    def validate_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        outputs = self.model(inputs)

        loss = self.cost(outputs, labels)
        score = self.metric(outputs, labels)

        return loss, score
    
    def validate(self, val_dataset: Dataset, batch_size: int):
        val_dataloader = self.build_dataloader(val_dataset, batch_size=batch_size)
        loss = 0.0
        score = 0.0
        num_batches = len(val_dataloader)
        print(f"================ Validate at Epoch {self.epoch+1} ================")
        for index, data in enumerate(tqdm(val_dataloader)):
            inputs = data[:, :-1].to(self.device)
            labels = data[:, 1:].to(self.device)
            batch_loss, batch_score = self.validate_step(inputs, labels)

            loss += batch_loss
            score += batch_score
        print(f"================ Done Validate at Epoch {self.epoch+1} ================")
        return loss/num_batches, score/num_batches
    
    
    def evaluate(self, data: torch.Tensor, batch_size: int = 1):
        self.model.eval()
        dataloader = self.build_dataloader(data, batch_size=batch_size)
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
    
class GPTMetric(nn.Module):
    def __init__(self, n_gram: int = 4, smooth: bool = False, pad_value: int = 0) -> None:
        super().__init__()
        self.bleu_assessor = torchmetrics.BLEUScore(n_gram=n_gram, smooth=smooth)
        self.perplexity_assessor = torchmetrics.Perplexity(ignore_index=pad_value)
        self.pad_value = pad_value

    def handle_padding(self, x: torch.Tensor):
        last_non_zero_index = len(x) - 1
        for i in range(len(x) - 1, -1, -1):
            if x[i] != self.pad_value:
                last_non_zero_index = i
                break

        x = x[:last_non_zero_index + 1]
        
        return x
    
    def create_sentence(self, x: torch.Tensor):
        return " ".join(x.cpu().numpy().astype(str))

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = outputs.cpu()
        labels = labels.cpu()

        batch_size = labels.size(0)
        _, predicted_tokens  = torch.max(outputs, dim=-1)
        
        bleu_score = 0.0

        for batch in range(batch_size):
            ref = self.create_sentence(self.handle_padding(labels[batch]))
            hypo = self.create_sentence(self.handle_padding(predicted_tokens[batch]))

            bleu_score += self.bleu_assessor([hypo], [[ref]])
    
        perplexity_score = self.perplexity_assessor(outputs, labels)

        return {
            MetricInfo.BLEU_SCORE: bleu_score.item()/batch_size,
            MetricInfo.PERPLEXITY_SCORE: perplexity_score.item()
        }

def get_batch_with_padding(batch ,padding_value: int = 0):
    max_length = np.max([len(item) for item in batch])
    data = []
    for item in batch:
        data.append(np.pad(item, (0, max_length-len(item)), constant_values=padding_value))
    return torch.tensor(np.array(data))

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


class MetricInfo:
    BLEU_SCORE = 'bleu_score'
    PERPLEXITY_SCORE = 'perplexity_score'


class TrackingInfo:
    pass