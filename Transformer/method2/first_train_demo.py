import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from transformer import make_model, subsequent_mask


class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]   #解码器的输入，去掉了句尾终止符
            self.trg_y = trg[:, 1:]   #解码器的输出，去掉了句首起始符
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()   #计算token数

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))   # padding_mask和sequence_mask，mask都用0表示
        return tgt_mask


# Synthetic Data
def data_gen(V, slen, batch, nbatches, device):
    """
    Generate random data for a src-tgt copy task.
    V: 词典数量，取值范围[0, V-1]，约定0作为特殊符号使用代表padding
    slen: 生成的序列数据的长度
    batch: batch_size
    nbatches: number of batches to generate
    """
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(2, V, size=(batch, slen)))   #从2开始是因为规定0是padding，1是起始符
        tgt_data = data.clone()
        tgt_data[:, 0] = 1   # 规定起始符为1，但这里并未规定终止符，可能有误？
        src = Variable(data, requires_grad=False)
        tgt = Variable(tgt_data, requires_grad=False)
        if device == "cuda":
            src = src.cuda()
            tgt = tgt.cuda() 
        yield Batch(src, tgt, 0)   # yield使得每次训练完一个batch再进行下一个
    

# test data_gen
data_iter = data_gen(V=5, slen=10, batch=2, nbatches=10, device="cpu")
for i, batch in enumerate(data_iter):
    print("\nbatch.src")
    print(batch.src.shape)
    print(batch.src)
    print("\nbatch.trg")
    print(batch.trg.shape)
    print(batch.trg)
    print("\nbatch.trg_y")
    print(batch.trg_y.shape)
    print(batch.trg_y)
    print("\nbatch.src_mask")
    print(batch.src_mask.shape)
    print(batch.src_mask)
    print("\nbatch.trg_mask")
    print(batch.trg_mask.shape)
    print(batch.trg_mask)
    break
#raise RuntimeError()


def run_epoch(data_iter, model, loss_compute, device=None):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        """
        norm: loss的归一化系数，用batch中所有有效token数即可
        """
        x = self.generator(x)
        x_ = x.contiguous().view(-1, x.size(-1))
        y_ = y.contiguous().view(-1)
        loss = self.criterion(x_, y_)
        loss /= norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss.item() * norm


# -----------------------------------
# A Easy Example
# -----------------------------------
# Train the simple copy task.
device = "cpu"
nrof_epochs = 40
batch_size = 32
V = 11    # 词典的数量
sequence_len = 15  # 生成的序列数据的长度
nrof_batch_train_epoch = 30    # 训练时每个epoch多少个batch
nrof_batch_valid_epoch = 10    # 验证时每个epoch多少个batch
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
#optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
if device == "cuda":
    model.cuda()

for epoch in range(nrof_epochs):
    print(f"\nepoch {epoch}")
    print("train...")
    model.train()
    data_iter = data_gen(V, sequence_len, batch_size, nrof_batch_train_epoch, device)
    loss_compute = SimpleLossCompute(model.generator, criterion, optimizer)
    train_mean_loss = run_epoch(data_iter, model, loss_compute, device)
    print("valid...")
    model.eval()
    valid_data_iter = data_gen(V, sequence_len, batch_size, nrof_batch_valid_epoch, device)
    valid_loss_compute = SimpleLossCompute(model.generator, criterion, None)
    valid_mean_loss = run_epoch(valid_data_iter, model, valid_loss_compute, device)
    print(f"valid loss: {valid_mean_loss}")


# greedy decode
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    # ys代表目前已生成的序列，最初为仅包含一个起始符的序列，不断将预测结果追加到序列最后
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)   
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

print("greedy decode")
model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]])).cuda()
src_mask = Variable(torch.ones(1, 1, 10)).cuda()
pred_result = greedy_decode(model, src, src_mask, max_len=10, start_symbol=1)
print(pred_result[:, 1:])


