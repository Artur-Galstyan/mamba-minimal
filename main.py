from model import Mamba, ModelArgs
from tinyshakespeareloader.hamlet import get_data
from torch import nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import torch

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


max_seq_len = 8
batch_size = 64
tinyshakespeare = get_data(batch_size=batch_size, block_size=max_seq_len, shuffle=True)
train_dataloader, test_dataloader = (
    tinyshakespeare.train_dataloader,
    tinyshakespeare.test_dataloader,
)

n_dims = tinyshakespeare.vocab_size if tinyshakespeare.vocab_size else 256
n_embd = 256  # 384
learning_rate = 3e-4
num_heads = 2  # 6
query_multihead_dim = num_heads
kv_multihead_dim = 2
n_layers = 3  # 6
max_new_tokens = 2000

model_args = ModelArgs(
    d_model=n_embd,
    n_layer=n_layers,
    vocab_size=n_dims,
)

mamba = Mamba(model_args).to(device)


loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
optimizer = optim.Adam(mamba.parameters(), lr=learning_rate)
start_time = time.time()
print(len(train_dataloader))
for i, (x, y) in tqdm(enumerate(train_dataloader)):
    x = x.to(device)
    y = y.to(device)
    outputs = mamba(x)
    #outputs = outputs.view(-1, outputs.shape[-1])  # Reshape to [64*8, 72]
    #y = y.view(-1)  # Flatten y to [64*8]
    #loss = loss_fn(outputs, y)
    #if i % 100 == 0:
    #    print(f"Epoch {i} loss: {loss.item()}")
    #optimizer.zero_grad()
    #loss.backward()
    #optimizer.step()
    if i > 3000:
        break


print(f"time to train to 3000 steps = {time.time() - start_time}")
