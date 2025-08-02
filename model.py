import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import re


from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings

def find_first_thai_choice(s: str) -> str | None:
    for ch in s:
        if ch in {'ก', 'ข', 'ค', 'ง'}:
            return ch
    return None


client = QdrantClient(host="172.16.30.145", port=6333)

embeddings = HuggingFaceBgeEmbeddings(model_name='Qwen/Qwen3-Embedding-4B')
sparse_embeddings = FastEmbedSparse(model_name='Qdrant/bm25')

vector_store = QdrantVectorStore(
    client=client,
    collection_name="CMKL_hackathon_Qwen4B",
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense_vector",
    sparse_vector_name="sparse_vector",
)

test = pd.read_csv('/home/siamai/cmkl-agentic-ai-for-healthcare-hackathon/test.csv')

retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
            search_kwargs={
                "k": 10,
                'score_threshold': 0.2
                })

results = vector_store.similarity_search_with_score('ผมปวดท้องมาก อ้วกด้วย ตอนนี้ตีสองยังมีแผนกไหนเปิดอยู่ไหมครับ?  ก. Endocrinology ข. Orthopedics ค. Emergency ง. Internal Medicine', k=20, score_threshold=0.2)
context = "\n---\n".join([doc.page_content for doc, _ in results])
print(context)

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

compressor = CrossEncoderReranker(model=reranker, top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke('ผมปวดท้องมาก อ้วกด้วย ตอนนี้ตีสองยังมีแผนกไหนเปิดอยู่ไหมครับ?  ก. Endocrinology ข. Orthopedics ค. Emergency ง. Internal Medicine')

from transformers import pipeline
import torch
import asyncio

pipe = pipeline("image-text-to-text", model="google/medgemma-27b-it")
# pipe = pipeline("text-generation", model="Qwen/Qwen3-14B")

async def call_pipeline(messages):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: pipe(messages, temperature=0.1))


# Prepare context text from compressed documents
context_text = ""
for j in range(len(compressed_docs)):
    context_text += compressed_docs[j].page_content + '\n\n-------------------------------------------------------------------\n\n'

async def infer(question: str):
    prompt = f"""
Context:
{context_text}

Question:
{question}
"""
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """
คุณเป็นแพทย์ผู้เชี่ยวชาญและต้องตอบคำถามโดยใช้เฉพาะข้อมูลจากบริบทที่ให้มาเท่านั้น พร้อมเลือกคำตอบที่ถูกต้องจากตัวเลือกต่อไปนี้: ก, ข, ค, ง
หากข้อมูลในบริบทไม่เพียงพอที่จะตอบอย่างเด็ดขาด คุณยังคงต้องเลือกคำตอบที่เป็นไปได้มากที่สุดโดยใช้เหตุผลทางการแพทย์
รูปแบบคำตอบที่ต้องการ: ตัวอักษรคำตอบเดียวหรือมากกว่าหนึ่งตัว (ไม่มีวงเล็บ ไม่มีจุด ไม่มีข้อความอื่นๆ) ตามด้วยคำอธิบายสั้น ๆ เป็นภาษาไทยในบรรทัดเดียว
ห้ามแสดงตัวเลือกอื่น ๆ ห้ามขึ้นบรรทัดใหม่ ห้ามใช้วงเล็บหรืออักขระพิเศษใด ๆ รอบตัวอักษรคำตอบ ห้ามใช้ bullet point หรือคำว่า "คำตอบคือ"
ตัวอย่างผลลัพธ์ที่ถูกต้อง:
"ก เป็นสิ่งที่สมควรทำเพราะข้อมูลในบริบทสนับสนุนการเลือกนี้",
"ข ไม่มีข้อมูลในบริบทเกี่ยวกับราคานี้แต่เป็นตัวเลือกที่ใกล้เคียงที่สุดตามเหตุผลทางการแพทย์",
"ก,ข,ง เพราะสามารถเข้าใช้บริการได้ในช่วงเวลาที่ระบุ", (จะสังเกตว่าต้องไม่มีช่องว่างระหว่างตัวอักษรและจุลภาคเลย)

ย้ำอีกครั้งว่าคุณต้องอธิบายเหตุผลของคำตอบของคุณหลังจากตัวอักษรคำตอบในบรรทัดเดียว
และหลังจากตัวอักษรคำตอบห้ามมีข้อความอื่น ๆ เช่น "คำตอบคือ" หรือ "ตัวเลือกที่ถูกต้องคือ" หรือ "คำตอบที่ถูกต้องคือ" หรือ "คำตอบคือ" หรือ "ตัวเลือกที่ถูกต้องคือ" หรือ ข้อมูลของตัวเลือกนั้น
รวมถึงต้องไม่มีช่องว่างระหว่างตัวอักษรคำตอบและจุลภาคหากมีมากกว่าหนึ่งตัวเลือก

YOU MUST NOT ADD ANY SPACE AFTER A COMMA, e.g. "ก,ข,ง" is correct but "ก, ข, ง" is incorrect.
YOU MUST COMPLY WITH THE FORMAT EXACTLY AS DESCRIBED ABOVE
IF YOU DO NOT KNOW THE ANSWER, YOU MUST ALWAYS STILL CHOOSE THE MOST LIKELY OPTION BASED ON THE CONTEXT PROVIDED, SO YOU MUST NOT ANSWER WITH "ไม่มีข้อมูลในบริบทที่ให้มาเกี่ยวกับโปรแกรมที่ใช้สำหรับบริการ" OR SOMETHING SIMILAR IT SHOULD ALWAYS BE A CHOICE FROM ก,ข,ค,ง
"""
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        },
    ]

    ans = await call_pipeline(messages)
    ans = ans[0]['generated_text'][-1]['content'].strip()

    sp = ans.find(' ')
    answer = ans[:sp]
    reason = ans[sp+1:].strip()
    # answer = find_first_thai_choice(ans)
    # reason = ans.replace(answer, '').strip()

    filtered = re.sub(r'[^กขคง,]', '', answer)
    return {"answer": filtered, "reason": reason}

'''
คุณเป็นแพทย์ผู้เชี่ยวชาญและต้องตอบคำถามโดยใช้เฉพาะข้อมูลจากบริบทที่ให้มาเท่านั้น พร้อมเลือกคำตอบที่ถูกต้องที่สุดจากตัวเลือกต่อไปนี้: ก, ข, ค, ง
หากข้อมูลในบริบทไม่เพียงพอที่จะตอบอย่างเด็ดขาด คุณยังคงต้องเลือกคำตอบที่เป็นไปได้มากที่สุดโดยใช้เหตุผลทางการแพทย์
รูปแบบคำตอบที่ต้องการ: ตัวอักษรคำตอบเดียว (ไม่มีวงเล็บ ไม่มีจุด ไม่มีข้อความอื่นๆ) ตามด้วยคำอธิบายสั้น ๆ เป็นภาษาไทยในบรรทัดเดียว
ห้ามแสดงตัวเลือกอื่น ๆ ห้ามขึ้นบรรทัดใหม่ ห้ามใช้วงเล็บหรืออักขระพิเศษใด ๆ รอบตัวอักษรคำตอบ ห้ามใช้ bullet point หรือคำว่า "คำตอบคือ"
ตัวอย่างผลลัพธ์ที่ถูกต้อง:
"ข ไม่มีข้อมูลในบริบทเกี่ยวกับราคานี้แต่เป็นตัวเลือกที่ใกล้เคียงที่สุดตามเหตุผลทางการแพทย์"

ย้ำอีกครั้งว่าคุณต้องอธิบายเหตุผลของคำตอบของคุณหลังจากตัวอักษรคำตอบในบรรทัดเดียว
และหลังจากตัวอักษรคำตอบห้ามมีข้อความอื่น ๆ เช่น "คำตอบคือ" หรือ "ตัวเลือกที่ถูกต้องคือ" หรือ "คำตอบที่ถูกต้องคือ" หรือ "คำตอบคือ" หรือ "ตัวเลือกที่ถูกต้องคือ" หรือ ข้อมูลของตัวเลือกนั้น'''