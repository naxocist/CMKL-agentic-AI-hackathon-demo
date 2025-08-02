# API specification
### API Tool: Fast API
### Endpoint
  - POST: /eval
    - Header Content-Type: application/json
    - Accepted request body: {"question": &lt;string&gt;} 
    - Response body: {"answer": {ก,ข,ค,ง}, "reason": &lt;string&gt;}

# Used Model
### Model Tool
  - langchain
  - qdrant
  - transformer pipeline

### Model
  - embedding: Qwen/Qwen3-Embedding-4B
  - sparse embedding: Qdrant/bm25
  - reranker: BAAI/bge-reranker-v2-m3
  - answer LLM: google/medgemma-27b-it