import tqdm
import json
import tensorflow as tf
import pandas as pd

from tensor2tensor.data_generators import text_encoder
import tensorflow_hub as hub
from transformers import AutoTokenizer

path = ''

ds_index_retrievals = {}

ind_retreivals_path = path + "data/qa_w_retrievals/"
for i in range(1, 5):
  file_name = f"ds_retrieval_indices_{i}.txt"
  with open(ind_retreivals_path + file_name, "r") as f:
    lines = f.readlines()
    for line in lines:
      retreivals = line.split(' ')[:10]
      for index in retreivals:
        ds_index_retrievals[int(index)] = 1

index_arr = []

for i in ds_index_retrievals:
  index_arr.append(i)

index_arr.sort()

VOCAB_PATH = "/content/drive/MyDrive/Senior Design/models/vocab.pg19_length8k.32768.subwords"
retriever_path = "/content/drive/MyDrive/Senior Design/models/retriever"
total = 1
max_seq_length = 2816
retrieval_corpus = "kilt"

print("Loading retrieval index...")

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

max = index_arr[-1]

documents = {}
count = 0

for dd in tqdm.tqdm(dataset.as_numpy_iterator()):
  if count in ds_index_retrievals:
    dd2 = tf.train.Example.FromString(dd)
    processed_list = dd2.features.feature['block_and_title'].int64_list.value
    decoded_str = pg19_vocab_encoder.decode(processed_list).split("---")[:2]
    documents[count] = decoded_str
  if count == max:
    break
  count += 1

with open(path + 'data/document_subsets/documents_datascience.bak', 'w', encoding='utf-8') as f:
    json.dump(documents, f, ensure_ascii=False, indent=4)

rt_dataset_format = {}
data_path = path + "data/datascience_qa.csv"
qa_data = pd.read_csv(data_path)
qa_data = qa_data.to_numpy()

def format_dict(question, answer, retreivals):
  dict_format = {"question:": "",
                "answer:": "",
                "retrievals:": [] }
  
  dict_format["question:"] = question
  dict_format["answer:"] = answer
  dict_format["retrievals:"] = retreivals

  return dict_format

ind_retreivals_path = path + "data/qa_w_retrievals/"
q_index = 0
for i in range(1, 5):
  file_name = f"ds_retrieval_indices_{i}.txt"
  with open(ind_retreivals_path + file_name, "r") as f:
    lines = f.readlines()
    for line in lines:
      question = qa_data[q_index][0]
      answer = qa_data[q_index][1]
      retrieval_ind = line.split(' ')[:7] # top 7 retrievals
      retrievals = []
      for ind in retrieval_ind:
        retrievals.append(documents[str(ind)])

      rt_dataset_format[q_index] = format_dict(
          question, answer, retrievals)
      q_index += 1

with open(path + 'data/qa_w_retrievals/datascience_qa.json.bak', 'w', encoding='utf-8') as f:
    json.dump(rt_dataset_format, f, ensure_ascii=False, indent=4)