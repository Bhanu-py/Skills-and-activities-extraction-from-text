from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm


def read_file(path):
    with open(path, 'r') as f:
        sent = [sent for sent in f.read().split("\n") if len(sent) != 0]
        return(sent)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_cos_sim(a, b) -> float:
    return np.dot(a, b) / (norm(a) * norm(b))


def get_embeds(sentences, model = "jjzha/jobbert-base-cased"):

  # Load model from HuggingFace Hub
  tokenizer = AutoTokenizer.from_pretrained(model)
  model = AutoModel.from_pretrained(model)

  # Check if a GPU is available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Tokenize sentences
  encoded_input = tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')
  attention_mask = encoded_input['attention_mask']

  # Move input to GPU
  encoded_input.to(device)
  attention_mask.to(device)

  # Compute token embeddings
  # with torch.no_grad():
  #     model_output = model(encoded_input.to(device), attention_mask=attention_mask.to(device))
  model.to(device)
  with torch.no_grad():
        model_output = model(**encoded_input)

  # Perform pooling
  sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

  # Normalize embeddings
  sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

  # Move embeddings back to CPU
  sentence_embeddings = sentence_embeddings.to("cpu")

  return sentence_embeddings

