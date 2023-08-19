import torch
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

class GPTMetric:
    def __init__(self) -> None:
        pass

    def handle_padding(self, x: torch.Tensor):
        last_non_zero_index = len(x) - 1
        for i in range(len(x) - 1, -1, -1):
            if x[i] != 0:
                last_non_zero_index = i
                break

        x = x[:last_non_zero_index + 1]
        return x

    def bleu_score(self, outputs: torch.Tensor, labels: torch.Tensor):
        batch_size = labels.size(0)
        outputs = outputs.cpu().numpy()
        labels = labels.cpu().numpy()
        
        score = 0.0

        for batch in range(batch_size):
            ref = self.handle_padding(labels[batch])
            hypo = self.handle_padding(outputs[batch])
            score += sentence_bleu([ref], hypo)
        
        return score/batch_size
    
    def perplexity_score(self, entropy: float):
        return np.exp(entropy)