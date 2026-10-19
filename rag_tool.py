import os
import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()

# 初始化 Chroma 本地向量库
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="company_knowledge")

API_KEY = os.getenv("SILICONFLOW_API_KEY")
EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"
MODEL_NAME = "BAAI/bge-m3"

def get_embedding(text: str) -> list:
    """调用硅基流动 API 获取文本向量"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "input": text
    }
    response = requests.post(EMBEDDING_URL, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Embedding API 调用失败: {response.text}")
    
    return response.json()["data"][0]["embedding"]

def build_knowledge_base(file_path: str = "knowledge.txt") -> str:
    """读取本地文件，切分并存入向量库"""
    if not os.path.exists(file_path):
        return f"知识库文件不存在: {file_path}"
    
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 切分策略：按段落（双换行）切
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    
    # 清空旧数据
    if collection.count() > 0:
        collection.delete(where={})

    ids = []
    embeddings = []
    documents = []
    
    for i, chunk in enumerate(chunks):
        try:
            emb = get_embedding(chunk)
            embeddings.append(emb)
            documents.append(chunk)
            ids.append(f"chunk_{i}")
        except Exception as e:
            return f"存入第 {i} 段时失败: {e}"
    
    if ids:
        collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )
        return f"成功存入 {len(ids)} 个知识片段"
    return "没有可存入的内容"

def search_knowledge(query: str, top_k: int = 2) -> str:
    """根据问题检索最相关的知识片段"""
    if collection.count() == 0:
        return "知识库为空，请先调用 build_knowledge_base 存入知识。"

    try:
        query_embedding = get_embedding(query)
    except Exception as e:
        return f"查询向量化失败: {e}"
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    if not results['documents'] or not results['documents'][0]:
        return "没有检索到相关内容"

    docs = results['documents'][0]
    return "\n---\n".join(docs)