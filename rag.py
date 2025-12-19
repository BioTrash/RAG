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

def add_chunk_to_database(chunk):
    payload = {
        "input": chunk
    }
    
    response = requests.post(EMBEDDING_SERVER, json=payload)
    response.raise_for_status() # try-catch HTTP err
    
    data = response.json()
    embedding = data["data"][0]["embedding"]
    
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
        

def main():
    load()

main()