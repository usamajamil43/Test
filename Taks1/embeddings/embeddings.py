import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    if torch.cuda.is_available():
        model.cuda()
        inputs = inputs.to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    return embeddings

def get_embeddings_create_index(df):
    embeddings_list = [get_embeddings(text) for text in df["Details"].tolist()]
    embeddings_array = np.vstack(embeddings_list).astype('float32')

    # Add embeddings as a new column in the DataFrame
    df['Embeddings'] = embeddings_list

    # Initialize FAISS index
    dimension = embeddings_array.shape[1]
    nlist = 47  # Adjust based on your dataset size for IVFFlat index
    quantizer = faiss.IndexFlatL2(dimension)  # Using L2 distance
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    index.train(embeddings_array)
    index.add(embeddings_array)
    index.nprobe = 10  # Adjust based on dataset and desired balance between speed and accuracy
    return index,df

# Import the embreddings here as well
def get_nearest_neighbors_index(user_query,index):
    query_embedding = get_embeddings(user_query)
    query_embedding_np = query_embedding.astype('float32').reshape(1, -1)
    k = 10  # Number of nearest neighbors
    D, I = index.search(query_embedding_np, k)  # D: distances, I: indices
    return D,I



def get_nearest_neighbors_data(indices,df):
    neighbor_indices = indices[0]
    neighbors_data = df.iloc[neighbor_indices][['Name', 'Details']]
    return neighbors_data

# Retrieve and print nearest neighbors

def create_details_string(nearest_neighbors_data):
    details_string = "\n".join(nearest_neighbors_data["Details"].tolist())
    return details_string

def construct_prompt(user_query, nearest_neighbors_data):
    details_string = create_details_string(nearest_neighbors_data)
    prompt = f"Based on the following menu items:\n{details_string}\nAnswer the question by the customer. If the user is asking to make an order, reply with the order completion message, including the price of the food ordered by the customer. If the ordered food is not present in the menu, return an apology message User Query: '{user_query}'\nResponse:"
    return prompt