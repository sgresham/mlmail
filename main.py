)impot pandas as pd
import torch
import time
import numpy as np
from glob import glob
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

#import and format dataframe
file_name = "mailbox-export.json"
full = pd.read_json(file_name)
y = full.drop(columns=["thread","timestamp","date_relative", "matched","total", "query","tags"])
y['category'] = y['authors'].astype('category')
y['catcode'] = y.category.cat.codes

#custom dataloader full
class CustomDatasetFull(Dataset):
    def __init__(self, dataframe):
        self.authors = dataframe["authors"]
        self.subject = dataframe["subject"]
        self.category = dataframe["category"]
        self.catcode = dataframe["catcode"]
    def __len__(self):
        return len(self.text)
    def __getitem__(self, idx):
        authors = self.authors.iloc[idx]
        subject = self.subject.iloc[idx]
        category = self.category.iloc[idx]
        catcode = self.catcode .iloc[idx]
        return authors, subject, category, catcode

class CustomDatasetMin(Dataset):
    def __init__(self, dataframe):
        self.authors = dataframe["authors"]
        self.subject = dataframe["subject"]
        self.category = dataframe["category"]
        self.catcode = dataframe["catcode"]
    def __len__(self):
        return len(self.text)
    def __getitem__(self, idx):
        authors = self.authors.iloc[idx]
        subject = self.subject.iloc[idx]
        category = self.category.iloc[idx]
        catcode = self.catcode .iloc[idx]
        return catcode, subject
    
prepped = CustomDatasetMin(y)    

#splitting to train/Validation
mask = np.random.rand(len(y)) < 0.8
trainDF = pd.DataFrame(y[mask])
validationDF = pd.DataFrame(y[~mask])

#turn subject into words, then tokenize them to numbers
tokenizer = get_tokenizer('basic_english')
def yield_tokens(data):
    for _, subject in data:
        yield tokenizer(subject)

vocab = build_vocab_from_iterator(yield_tokens(prepped), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
subject_pipeline = lambda x: vocab(tokenizer(x))
authors_pipeline = lambda x: int(x) - 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    authors_list, subject_list, offsets = [], [], [0]
    for (_authors, _subject) in batch:
        authors_list.append(authors_pipeline(_authors))
        processed_subject = torch.tensor(subject_pipeline(_subject), dtype=torch.int64)
        subject_list.append(processed_subject)
        offsets.append(processed_subject.size(0))
    authors_list = torch.tensor(authors_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    #authors_list = torch.cat(authors_list)
    return authors_list.to(device), subject_list.to(device), offsets.to(device)

#dataloaderdemo = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)
dataloaderemail = DataLoader(prepped, batch_size=8, shuffle=False, collate_fn=collate_batch)

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

num_class = len(set([authors for (authors, subject) in prepped]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()
    for idx, (authors, subject, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_authors = model(subject, offsets)
        loss = criterion(predicted_authors, authors)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_authors.argmax(1) == authors).sum().item()
        total_count += authors.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for idx, (authors, subject, offsets) in enumerate(dataloader):
            predicted_authors = model(subject, offsets)
            loss = criterion(predicted_authors, authors)
            total_acc += (predicted_authors.argmax(1) == authors).sum().item()
            total_count += authors.size(0)
    return total_acc/total_count

# Hyperparameters
EPOCHS = 10 # epoch
LR = 5  # learning rate
BATCH_SIZE = 1 # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter = prepped
test_iter = prepped
train_dataset = prepped
test_dataset = prepped
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)r
