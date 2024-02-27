import json
import requests
from langchain.text_splitter import RecursiveJsonSplitter
from transformers import AutoTokenizer, AutoModel


from data_processing.data_processing import json_to_dataframe
from embeddings.embeddings import get_embeddings_create_index
from embeddings.embeddings import get_nearest_neighbors_index
from embeddings.embeddings import get_nearest_neighbors_data
from embeddings.embeddings import construct_prompt
json_data = requests.get("https://gist.githubusercontent.com/xapss/f1bc847ed57236c11f1e810095fa7555/raw/57bfab76abdecb0de83476fae953fac8b8c68378/menu.json").json()
splitter = RecursiveJsonSplitter(max_chunk_size=300)
json_chunks = splitter.split_json(json_data=json_data)


model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# Convert JSON chunks to DataFrame
df = json_to_dataframe(json_chunks)

index,df=get_embeddings_create_index(df)



def main(user_query):
    D,I= get_nearest_neighbors_index(user_query,index)
    nearest_neighbors_data = get_nearest_neighbors_data(I,df)
    prompt = construct_prompt(user_query, nearest_neighbors_data)
    return prompt

if __name__ == '__main__':
    user_query = "How many calories does the Fire Zinger stacker have?"  # Placeholder for actual user query
    prompt = main(user_query)
    print(prompt)
