#%%
from gpt import GPT
import torch
import torchsummary
# %%
gpt = GPT(
    token_size=1800,
    n=12,
    d_model=768,
    heads=12,
    d_ff=3072,
    activation=torch.nn.functional.gelu,
    dropout_rate=0.1,
    eps=0.02
)
# %%
torchsummary.summary(gpt)
# %%
