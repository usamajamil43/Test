
# Menu-Based Query Handling System

## Overview:
This project is designed to handle queries related to menu items using NLP and nearest neighbors search. It processes JSON menu data, converts it to a structured format, creates text embeddings, and uses these embeddings to find the most relevant menu items to a user's query.

- We are using faiss vector db, we can easily apply CRUD operations on this. 

## Project Structure:
- `main.py`: The main script that orchestrates the data fetching, processing, and query handling.
- `data_processing/`: Contains modules for processing JSON data into a structured format.
  - `data_processing.py`: Functions for converting JSON to DataFrame and processing menu items.
- `embeddings/`: Contains modules for handling embeddings.
  - `embeddings.py`: Functions for creating embeddings, constructing nearest neighbors index, and generating responses based on user queries.

## How It Works:
1. **main.py**:
   - Fetches menu data in JSON format.
   - Splits and processes the JSON data into manageable chunks.
   - Converts the data into a structured DataFrame.
   - Generates embeddings for the menu items and constructs a nearest neighbors index.
   - Accepts user queries and finds the most relevant menu items based on the embeddings.
   - Constructs and returns a prompt answering the user's query based on the nearest menu items.

2. **data_processing/data_processing.py**:
   - `process_item()`: Extracts and processes details from each menu item.
   - `json_to_dataframe()`: Converts JSON chunks into a pandas DataFrame for analysis and processing.

3. **embeddings/embeddings.py**:
   - `get_embeddings()`: Generates embeddings for given textual content.
   - `get_embeddings_create_index()`: Generates embeddings for all items in a DataFrame and prepares a FAISS index for efficient nearest neighbor searches.
   - `get_nearest_neighbors_index()`, `get_nearest_neighbors_data()`: Functions to retrieve nearest neighbors based on a query.
   - `construct_prompt()`: Constructs a response prompt based on the nearest neighbors' data and the user query.

## Setup and Execution:
- Ensure all dependencies are installed as per `requirements.txt`.
- Run `main.py` to start the program. Modify the `user_query` in `main.py` as needed to test different queries.

## Notes:
- The project is set up to handle menu-based queries but can be adapted for other types of data.
- Embeddings and nearest neighbor search parameters can be adjusted based on different datasets or requirements.
