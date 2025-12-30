import requests, json, time


# ./llama-server -m ~/Git/llama.cpp/models/bge-base-en-v1.5-f32.gguf --embeddings --host 127.0.0.1 --port 8080 -c 2048 -ngl 99
EMBEDDING_SERVER = "http://localhost:8080/v1/embeddings"

# ./llama-server -m ~/Git/llama.cpp/models/MODEL_GGUF --host 127.0.0.1 --port 808X --gpu-layers -1 -mg 0 -c 4096
CHAT_SERVER = "http://localhost:8081/v1/chat/completions"
CLASS_SERVER = "http://localhost:8082/v1/chat/completions"
REWRITE_SERVER = "http://localhost:8083/v1/chat/completions"
AMBIGIOUS_SERVER = "http://localhost:8084/v1/chat/completions"
COMPLEX_SERVER = "http://localhost:8085/v1/chat/completions"

# ../embeddings	                Generate embeddings
# ../v1/embeddings	            OpenAI-compatible embeddings
# ../v1/chat/completions	    Chat completions
# ../v1/completions	            Text completions
# ../health	                    Server health check

VECTOR_DB = []

dataset = []

# Pre-Retrieval:
#
#   Potential Problems:
#       Poorly worded queries
#       Complex queries
#       Ambigious queries
#
#   Potential Solutions:
#       Expand original query into multiple queries in order to further the context (Multi-Query)
#           Assign greater weight to original query
#           Validate each query via LLM by a comparison to the original so as to reduce hallucination and increasce relevance
#           Validate final responce-query via the same method as above
#           Least-to-Most Prompting vs Chain-of-Thought Promting vs Chain-of-Verification Promting
#               Least-to-Most runs the generated queries sequentially, using the answer to the previous one as the basis for the next one until a satifsying solution to the original query is reached
#               Chain-of-Thought runs the generated queries in parallel, fusing them at the end (Unsuitable for smaller models)
#               Chain-of-Verification generates queries on a draft-solution query instead of the original query itself (Unsuitable for smaller models)
#       Rewrite the original query based on assumed meaning
#           Prompt User for confirmation about assumed meaning if the rewriting is too semantically different from the original (Retrieval dependant)
#       Generate hypothetical query for each chunk in the database (Unsuitable for smaller models)
#           Leads to longer load time initially depending on chunk amount
#           Focus on semantical similarity between original query and hypothetical query instead of between original query and chunk
#       Generate a high-level concept of the original query, use both for retrieval and generate the answer based on both
#
#   Combined Solution:  
#       Avoid parallel executions for the sake of performance on low-end hardware
#       Determine query type (Orchestration-preferable)
#           Skip to retrieval if it considered simple 
#           Rewrite the original query if it is considered poorly worded 
#           Generate a high-level concept if it is considered ambigious
#           Employ Least-to-Most sequential prompting if it is considered complex
#               If a query is considered multiple types, prioritize: Poorly Worded > Ambigious > Complex
#           Recursively check type of the generated query unless it is considered simple.
#               Set a max. amount of reccursions so as to avoid endless loops, long wait times and/or oversimplification
#                  If max. amount is reached, prompt the use for confirmation on final deduced prompt-meaning before retrieval
#                      Upon rejection, clarification from user is asked, alternatively provided with the rejection, the process restarts but the original query is appended with the clarification.
#             Temporarily save all generated final-queries at the end of each reccursion
#             Validate generated querie's relevance and accuracy compared to the original querie at the end of each reccursion
#                 Go back to the final-query generated in the latest reccursion if current final-query is considered irrelevant and/or inaccurate and try again

#   ToDo:
#       Implement an LLM-Judge (DONE)
#           LLM Decides whether the User's prompt is Simple, Poorly Worded, Ambigious or Complex via internal reasoning
#           LLM outputs a JSON-formated string, for consistency, with the determined type
#           The determined type is stored in a local variable
#       Type-Based Orchestration
#           If SIMPLE skip to retrieval and generation (DONE)
#           if POORLY_WORDED prompt LLM to rewrite the query to the best of its abilities
#           if AMBIGIOUS propmt LLM to generate a 'step-back' query
#           if COMPLEX implement Least-To-Most prompting method
#
#   
#

def llm_judge(query): # Orchestration
    
    with open('data/llm_query_classifier.json') as file:
        classifier = json.load(file)
        
    guide = f'You are a classifier. Your job is to classify the USER query type. Here is the JSON-formated classifying instruction that you are to follow EXACTLY: {json.dumps(classifier)}'
    
    determination = call_to_chat_server(guide, query, 64, 0.1, CLASS_SERVER)
    
    parsed = extract_json(determination)
    query_type = parsed["type"]
    
    ALLOWED = {"SIMPLE", "POORLY_WORDED", "AMBIGUOUS", "COMPLEX"}
    
    if query_type not in ALLOWED:
        print(f'NOT IN ALLOWED | QUERY TYPE: {query_type}')
        return pathing(query)
    else:
        print(f'ALLOWED | QUERY TYPE: {query_type}')
        return pathing(query, query_type)

def pathing(query, query_type:str="SIMPLE"):
    match query_type:
        case "SIMPLE":
            retrieved = retrieve(query)
                
            guide = f'Use only the following pieces of context to answer the user query. Do not make any new information: {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved])}'

            return call_to_chat_server(guide, query)
        
        case "POORLY_WORDED":
            with open('data/llm_query_rewriter.json') as file: # Move out to a seprate file-loader instead of loading at each iteration
                rewriter = json.load(file)
                
            guide = f'You are a rewriter. Your job is to rewrite the USER query. Here is the JSON-formated rewriting instruction that you are to follow EXACTLY: {json.dumps(rewriter)}'

            response = call_to_chat_server(guide, query, 512, 0.1, REWRITE_SERVER)
            
            parsed = extract_json(response)
            rewritten_query = parsed["rewritten_query"]
            
            return llm_judge(rewritten_query) 
            
                    

def extract_json(text):
    start = text.find('{')
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth+=1
        elif text[i] == '}':
            depth-=1
            if depth == 0:
                candidate = text[start:i+1]
                return json.loads(candidate)
    
    return None


def call_to_chat_server(guide_prompt, user_query, max_tokens:int=512, temperature:float=0.1, server=CHAT_SERVER):
    
    payload = {
        "messages": [
            {"role": "system", "content": guide_prompt},
            {"role": "user", "content": user_query}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    response = requests.post(server, json=payload)
    response.raise_for_status() # try-catch HTTP err

    
    data = response.json()
    print("Raw response:", json.dumps(data, indent=2))
    reply = data["choices"][0]["message"]["content"]
    
    time.sleep(1.0)
    
    return reply

def call_to_embedding_server(chunk):
    payload = {
        "input": chunk
    }
    
    response = requests.post(EMBEDDING_SERVER, json=payload)
    response.raise_for_status() # try-catch HTTP err
    
    data = response.json()
    embedding = data["data"][0]["embedding"]
    
    return embedding

def add_chunk_to_database(chunk, database:list): 
    embedding = call_to_embedding_server(chunk)

    database.append((chunk, embedding))

def load():
    with open('data/cat-facts.txt') as file:
        dataset = file.readlines()
        print(f'Loaded {len(dataset)} entries')
        
        for i, chunk in enumerate(dataset):
            add_chunk_to_database(chunk, VECTOR_DB)
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
    input_query = input("Query: ") 
    print(llm_judge(input_query))
    
main()