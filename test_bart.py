from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch


def load_model(model_name_or_path):
    tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
    model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
    return tokenizer, model


def format_result(self, result):
    query = result[1]
    prediction = result[0]
    prediction_meta = {"model": self.__class__.__name__, "sparql_query": query}
    return {"answer": prediction, "prediction_meta": prediction_meta}

def process_query(query, tokenizer, model, top_k=2):
    inputs = tokenizer([query], max_length=100, truncation=True, return_tensors='pt')
    # generate self.top_k+2 SPARQL queries so that we can dismiss some queries with wrong syntax
    temp = model.generate(inputs['input_ids'], num_beams=5, max_length=100, num_return_sequences=top_k+2, early_stopping=True)
    sparql_queries = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in temp]
    
    return sparql_queries

question = "Who are the buyers for this category: L357"
# model_name_or_path = '/opt/notebooks/viona/test-knowledge-graph1/MK-SQuIT/test_bart/data/lcquad_full_wikidata'
model_name_or_path = '/opt/notebooks/viona/test-knowledge-graph1/test_bart/data/output/model_2'
tokenizer, model = load_model(model_name_or_path)
data = ''
while True:
    data = input("Enter your query: ")
    answer = process_query(data, tokenizer, model)
    print(answer)
    if data == 'Y':
        break
    print("\n\n\n")