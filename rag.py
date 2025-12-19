import requests

# ./llama-server -m ~/Git/llama.cpp/models/bge-base-en-v1.5-f32.gguf --embeddings --host 127.0.0.1 --port 8080 -c 2048 -ngl 99
EMBEDDING_SERVER = "http://localhost:8080/v1/embeddings"

# ./llama-server -m ~/Git/llama.cpp/models/Phi-3-mini-4k-instruct-fp16.gguf --host 127.0.0.1 --port 8081 -n 500 --gpu-layers 40 -sm none -mg 0 -c 4096
CHAT_SERVER = "http://localhost:8081/v1/chat/completions"

# ../embeddings	                Generate embeddings
# ../v1/embeddings	            OpenAI-compatible embeddings
# ../v1/chat/completions	    Chat completions
# ../v1/completions	            Text completions
# ../health	                    Server health check

VECTOR_DB = []

dataset = []

def call_to_embedding_server(chunk):
    payload = {
        "input": chunk
    }
    
    response = requests.post(EMBEDDING_SERVER, json=payload)
    response.raise_for_status() # try-catch HTTP err
    
    data = response.json()
    embedding = data["data"][0]["embedding"]
    
    return embedding

def add_chunk_to_database(chunk):
    embedding = call_to_embedding_server(chunk)

    VECTOR_DB.append((chunk, embedding))
    
    #print("Embedding length:", len(embedding))
    #print("First 5 values:", embedding[:5])

def load():
    with open('data/cat-facts.txt') as file:
        dataset = file.readlines()
        print(f'Loaded {len(dataset)} entries')
        
        for i, chunk in enumerate(dataset):
            add_chunk_to_database(chunk)
            print(f'Added chunk {i+1}/{len(dataset)} to the database')
        
def cosine_similarity(a, b): 
    dot_product = sum([x * y for x, y in zip(a, b)]) # Semantic similarity of chunks, obscured by magnitude i.e. length.
    norm_a = sum([x ** 2 for x in a]) ** 0.5 # Normalizes chunk a length
    norm_b = sum([x ** 2 for x in b]) ** 0.5 # Normalizes chunk b length
    return dot_product / (norm_a * norm_b) # Measured semantic similarity 

#   Example:
#
#   a = [1, 2, 3] (x, y, z)
#   b = [4, 5, 6] (x, y, z) 
#   
#   list(zip(a,b)) --> [(1,4), (2,5), (3,6)]
#   
#   [x * y for x, y in zip(a, b)] --> 1*4, 2*5, 3*6 --> [4, 10, 18]
#   
#   dot_product --> 4 + 10 + 18 = 32
#       
#   norm_a == square of (1*1 + 2*2 + 3*3) == 3.7417 
#   norm_b == square of (4*4 + 5*5 + 6*6) == 8.775
#   
#   norms can be thought of as arrows on a graph pointing from 0, 0, 0, and return is the difference in direction they are pointing in
#   return is 1 || 0 || -1 with 1 implying same dirtection, 0 implying random direction but not opposite or the the same, and -1 being opposite direction

def retrieve(query, top_n=3):
    query_embedding = call_to_embedding_server(query)
    
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
        
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_n]


def main():
    load()
    print(retrieve(input("Query: ")))

main()