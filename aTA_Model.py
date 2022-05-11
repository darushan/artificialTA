import torch # import torch before tf
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # Hide tf warnings. TODO: doesn't work?
tf.config.set_visible_devices([], 'GPU') # Hide GPU for tf, retriever does not fit in GPU memory

import json
import tensorflow_hub as hub
import time
from routing_transformer import RoutingTransformerEncDec
from transformers import AutoTokenizer

class ATA_Model():
    """
    Class for aTA deployment.
    Creating instance of class loads model components.
    Call .get_answer(query) to generate from aTA instance.
    """
    def __init__(self):
        # Constants / Model Parameters
        self.path = ""
        self.DIM = 512
        self.MAX_SEQ_LEN = 4096
        self.DEC_SEQ_LEN = 2048
        self.DROPOUT = 0 # 0 for eval
        self.NUM_TOKENS = 65536
        self.WINDOW_SIZE = 256 
        self.HEADS = 8
        self.LAYERS = 18 
        self.NUM_RETRIEVALS = 7

        print("Initializing CUDA for torch...") # First torch .cuda() call takes a bit.
        self.start_token = (torch.zeros((1, 1)) * 1).long().cuda()
        
        print("Loading tokenizers...")
        self.tokenizer_uncased = AutoTokenizer.from_pretrained(
            "bert-base-uncased")
        self.tokenizer_cased = AutoTokenizer.from_pretrained(
            "bert-base-cased")
        
        print("Loading retrieval corpus...")
        self.documents = self.get_document_set()
        
        print("Loading retriever...")
        self.crealm, self.weights = self.get_crealm()
        
        print("Loading generator...")
        self.rt_model = self.get_rt_model()
    
    def get_document_set(self):
        """
        Get closed-domain subset of KILT Wikipedia corpus.
        """
        documents = {}
        with open(self.path + 'data/document_subsets/documents_datascience.json', "r", encoding='utf-8') as f:
          documents = json.load(f)
        return documents

    def get_crealm(self):
        """
        Load C-REALM model from checkpoint.
        """
        retriever_path = self.path + "models/retriever"
        encoded_path = self.path + "models/retriever/encoded_kilt_wiki_pg19_vocab_dash_sep/encoded.ckpt"
        encoded_weights = tf.train.load_variable(encoded_path, "block_emb")
        retriever = hub.KerasLayer(retriever_path, signature="encode_candidates", signature_outputs_as_dict=True)
        return retriever, encoded_weights

    def retrieve_documents(self, query):
        """
        Inference on C-REALM to retrieve documents from query.
        """
        str_tokens = self.tokenizer_uncased(
            query, 
            truncation=True, 
            padding="max_length",
            max_length=288, 
            return_tensors="tf")
        
        input_map = {
            "input_ids": str_tokens["input_ids"],
            "segment_ids": str_tokens["token_type_ids"],
            "input_mask": str_tokens["attention_mask"]
        }
    
        retrieved_emb = self.crealm(input_map)["default"]
        retrieval_scores = tf.matmul(retrieved_emb, tf.transpose(self.weights))
        top_retrievals = tf.math.top_k(retrieval_scores, k=self.NUM_RETRIEVALS).indices.numpy()
        
        retrievals = []
        for retrieval in top_retrievals[0]:
            if str(retrieval) in self.documents:
                retrievals.append(self.documents[str(retrieval)])

        # Not enough related documents to answer.
        if len(retrievals) < 5:
            return None
        
        # Pad documents with top retrieval if only a couple documents missing.
        while len(retrievals) < 7:
            retrievals.append(self.documents[str(top_retrievals[0][0])])

        retrievals = [x[0]+"---"+x[1] for x in retrievals]
        return retrievals


    def format_query(self, query, retrievals):
        """
        Concatenates documents + query for RT inference.
        """
        query += "---"
        retrievals = retrievals[::-1] # Reverse retrievals (hurdles paper)

        ques_and_rets = ""
        for ret in retrievals:
          ques_and_rets += ret + " "
        ques_and_rets += query # Concat question to end of rets (hurdles paper)
        
        query = str(ques_and_rets).strip()
        return query
    
    def get_rt_model(self):
        """
        Loads routing transformer module from checkpoint.
        """
        model = RoutingTransformerEncDec(
            dim=self.DIM,
            enc_num_tokens=self.NUM_TOKENS,
            enc_depth=self.LAYERS // 2,
            enc_heads=self.HEADS,
            enc_max_seq_len=self.MAX_SEQ_LEN,
            enc_window_size=self.WINDOW_SIZE,
            dec_num_tokens=self.NUM_TOKENS,
            dec_depth=self.LAYERS // 2,
            dec_heads=self.HEADS,
            dec_max_seq_len=self.DEC_SEQ_LEN,
            dec_window_size=self.WINDOW_SIZE,
            reversible=True,
            shift_tokens=True,
            attn_dropout=self.DROPOUT,
            ff_dropout=self.DROPOUT,
            layer_dropout=self.DROPOUT,
            causal=True
        ).cuda()

        models_path = self.path + "models/pytorch/"
        model_name = "pytorchqa_rt100.pt"
        model.load_state_dict(torch.load(models_path+model_name))

        return model

    def rt_inference(self, query):
        """
        Generates answer using RT from documents + query.
        """
        self.rt_model.eval()

        # Encode Query
        enc = self.tokenizer_cased.encode_plus(
            query, 
            None, 
            max_length=self.MAX_SEQ_LEN, 
            padding="max_length", 
            truncation=True)
        
        inp = enc['input_ids']
        inp = torch.tensor([inp]).long().cuda()

        predict = self.rt_model.generate(
            inp[0:1], self.start_token, self.DEC_SEQ_LEN, eos_token=102)
        
        gen = self.tokenizer_cased.decode(predict[0][1:-1])
        gen = gen.capitalize()

        return gen

    def get_answer(self, query):
        """
        Get answer from aTA given a query. 
        """
        retrievals = self.retrieve_documents(query)
        
        if not retrievals:
            answer = "Sorry I lack the knowledge to answer this question."
        else:
            print(f"Retrievals:\n{retrievals}")
            query = self.format_query(query, retrievals)
            start = time.time()
            print("\nGetting answer...")
            answer = self.rt_inference(query)
            total_time = time.time() - start
            print(f"\nTime to answer: {total_time:.2f}\n")

        return answer