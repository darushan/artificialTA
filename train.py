import torch
import torch.optim as optim
import time
import json
from tqdm import tqdm
from routing_transformer import RoutingTransformerEncDec
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

path = ""

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

data_path = path + "data/qa_w_retrievals/datascience_qa.json"

qa_data = {}

with open(data_path, "r", encoding='utf-8') as f:
    qa_data = json.load(f)

qa_data_arr = []

for i in qa_data:
  question = qa_data[i]["question:"]
  question += "---"
  answer = qa_data[i]["answer:"]
  retrievals = qa_data[i]["retrievals:"] # Top 5
  retrievals = [x[0]+"---"+x[1] for x in retrievals] # Concat title + body
  retrievals = retrievals[::-1] # Reverse retrievals (hurdles paper)

  ques_and_rets = ""
  for ret in retrievals:
    ques_and_rets += ret + " "
  ques_and_rets += question # Concat question to end of rets (hurdles paper)
  
  query= str(ques_and_rets).strip()
  answer = str(answer).strip()

  if answer != "nan":
    # len(tokenizer(query)['input_ids'])
    qa_data_arr.append([query, answer])

# Split dataset into training and validation set.
train_data, valid_data = train_test_split(
    qa_data_arr, train_size=0.99, shuffle=True, random_state=47)

len_train_data = len(train_data)
print(f"Samples in training set: {len_train_data}")
print(f"Samples in validation set: {len(valid_data)}")

# Constants / Model Parameters
BATCH_SIZE = 1 # 128 Mini batch size in (hurdles paper)
NUM_BATCHES = len_train_data//BATCH_SIZE
LEARNING_RATE = 5e-5 # (hurdles paper)
EVALUATE_EVERY  = NUM_BATCHES//4
NUM_TOKENS = 65536 #~32k, possibly use 65536
ENC_SEQ_LEN = 4096 # 8192 Max tokens in tokenized sequence. #(hurdles paper)
DEC_SEQ_LEN = 2048
DROPOUT = .15 # (hurdles paper)
WINDOW_SIZE = 256 # 512(RT paper)
HEADS = 8 # (RT paper)
LAYERS = 18 # 22(RT paper)
start_token = (torch.zeros((1, 1)) * 1).long().cuda()

# Set up train/valid input and target sets.
train_inp_data = []
train_tgt_data = []

for qa_pair in train_data:
  train_inp_data.append(qa_pair[0]) # Query
  train_tgt_data.append(qa_pair[1]) # Answer

valid_inp_data = []
valid_tgt_data = []

for qa_pair in valid_data:
  valid_inp_data.append(qa_pair[0]) # Query
  valid_tgt_data.append(qa_pair[1]) # Answer

train_inp_data = np.array(train_inp_data)
train_tgt_data = np.array(train_tgt_data)
valid_inp_data = np.array(valid_inp_data)
valid_tgt_data = np.array(valid_tgt_data)

del qa_data
del qa_data_arr
del train_data
del valid_data

# Class definition for PyTorch Dataset
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class QADataset(Dataset):
    def __init__(self, input_data, target_data):
        super().__init__()
        self.input_data = input_data
        self.target_data = target_data

    def __getitem__(self, index):
        # Encode Queries
        enc = tokenizer.encode_plus(
            self.input_data[index], 
            None, 
            max_length=ENC_SEQ_LEN, 
            padding="max_length", 
            truncation=True)
        inp = enc['input_ids']
        inp = torch.tensor(inp).long().cuda()
        inp_mask = torch.tensor(enc['attention_mask']).bool().cuda()

        # Encode Answers
        enc = tokenizer.encode_plus(
            self.target_data[index], 
            None, 
            max_length=DEC_SEQ_LEN, 
            padding="max_length",
            truncation=True)
        tgt = enc['input_ids']
        tgt = torch.tensor(tgt).long().cuda()
        tgt_mask = torch.tensor(enc['attention_mask']).bool().cuda()

        return inp, inp_mask, tgt, tgt_mask

    def __len__(self):
        return len(self.input_data)

# Convert datasets to PyTorch Datasets and Loaders
train_dataset = QADataset(train_inp_data, train_tgt_data)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataset = QADataset(valid_inp_data, valid_tgt_data)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Routing transformer w/ encoder decoder stack.
model = RoutingTransformerEncDec(
    dim=512,
    enc_num_tokens=NUM_TOKENS,
    enc_depth=LAYERS // 2, # Half to enc half to dec
    enc_heads=HEADS,
    enc_max_seq_len=ENC_SEQ_LEN,
    enc_window_size=WINDOW_SIZE,
    dec_num_tokens=NUM_TOKENS,
    dec_depth=LAYERS // 2,
    dec_heads=HEADS,
    dec_max_seq_len=DEC_SEQ_LEN,
    dec_window_size=WINDOW_SIZE,
    reversible=True,
    shift_tokens=True,
    attn_dropout=DROPOUT,
    ff_dropout=DROPOUT,
    layer_dropout=DROPOUT,
    causal=True # Auto-regressive
).cuda()

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Function for evaluating model.
def evaluate(model, data_loader, batch_size=BATCH_SIZE, sample_size=1):
  model.eval()
  total = batch_size*sample_size

  for _ in range(sample_size):
    inp, inp_mask, tgt, _ = next(data_loader)

    for i, x in enumerate(inp):
      predict = model.generate(
          inp[i:i+1], start_token, DEC_SEQ_LEN, eos_token=102)
      
      gen = tokenizer.decode(predict[0])
      actual = tokenizer.decode(tgt[0][1:])
      print("\n\nGeneration: ", gen)
      print("\nActual: ", actual)

# Training loop for one epoch.
def train():
  loss_tot = 0
  aux_tot = 0
  for i in tqdm(range(NUM_BATCHES)):
      model.train()

      inp, inp_mask, tgt, tgt_mask = next(iter(train_loader))
      loss, aux_loss = model(
          inp, tgt, enc_input_mask=inp_mask, dec_input_mask=tgt_mask, 
          return_loss = True, randomly_truncate_sequence = True)
      loss.backward()
      aux_loss.backward()

      loss_tot += loss.item()
      aux_tot += aux_loss.item()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
      optim.step()
      optim.zero_grad()

      if i != 0 and i % EVALUATE_EVERY == 0:
          print(f"\n   loss at step {i}: {loss_tot/EVALUATE_EVERY}")
          print(f"auxloss at step {i}: {aux_tot/EVALUATE_EVERY}")
          loss_tot = 0
          aux_tot = 0

          print("\nGenerate from Training:")
          evaluate(model, iter(train_loader))

# Load last checkpoint
models_path = path + "models/pytorch/pytorch"
model_name = "qa_rt150.pt"
model.load_state_dict(torch.load(models_path+model_name))


# Run Training
starting_epoch = 151
epochs = 160
for epoch in range(starting_epoch, epochs+1):
  print("\n------------------------------------------\n")
  print(f"Epoch: {epoch}")
  start = time.time()
  train()
  total_time = time.time() - start
  print(f"\nEpoch elapsed time: {total_time}")
  print("--- Generate from Validation ---")
  evaluate(model, iter(valid_loader))
  model_name = "qa_rt" + str(epoch) + ".pt"
  torch.save(model.state_dict(), models_path+model_name)