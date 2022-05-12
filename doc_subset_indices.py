import tensorflow as tf
import pandas as pd
from tensor2tensor.data_generators import text_encoder
import tensorflow_hub as hub
from transformers import AutoTokenizer

path = ''

VOCAB_PATH = path + "/models/vocab.pg19_length8k.32768.subwords"
retrieval_corpus = "eli5_train"
retriever_path = path + "/models/retriever"
total = 1
max_seq_length = 2816
retrieval_corpus = "kilt"

blocks_file = path + "models/retrieval_train/blocks_and_titles_pg19_dash_sep.tfr"
encoded_path = path + "models/retriever/encoded_kilt_wiki_pg19_vocab_dash_sep/encoded.ckpt"
encoded_weights = tf.train.load_variable(encoded_path, "block_emb")

print("Loading retrieval corpus...")
pg19_vocab_encoder = text_encoder.SubwordTextEncoder(VOCAB_PATH)
dataset = tf.data.TFRecordDataset(blocks_file)

print("Loading retriever...")
# encode_queries and encode_candidates are the same since the encoders are shared
retriever = hub.KerasLayer(retriever_path, signature="encode_candidates", signature_outputs_as_dict=True)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

data_path = path + "data/datascience_qa.csv"
data = pd.read_csv(data_path)
data = data.to_numpy()

ds_retrieval_indices = []

count = 1
for qa_pair in data:
  print(f"Current: {count}/{len(data)}")
  question = [qa_pair[0]]
  str_tokens = bert_tokenizer(question[0], truncation=True, padding="max_length",
                            max_length=288, return_tensors="tf")
  
  input_map = {
    "input_ids": str_tokens["input_ids"],
    "segment_ids": str_tokens["token_type_ids"],
    "input_mask": str_tokens["attention_mask"]
  }
  retrieved_emb = retriever(input_map)["default"]
  retrieval_scores = tf.matmul(retrieved_emb, tf.transpose(encoded_weights))
  top_retrievals = tf.math.top_k(retrieval_scores, k=10).indices.numpy()

  print(f"\nInput question = {question[0]}\n")
  print(top_retrievals)

  ds_retrieval_indices.append(top_retrievals[0])
  count += 1

out_file_path = path + "data/qa_w_retrievals/ds_retrieval_indices.txt"
with open(out_file_path, 'w') as out_file:
  for indices in ds_retrieval_indices:
    for i in indices:
      out_file.write(str(i) + " ")
    out_file.write(str(i) + "\n")