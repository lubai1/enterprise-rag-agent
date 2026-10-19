import streamlit as st
import os
import requests
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

# ================= 1. 页面基础配置 =================
st.set_page_config(page_title="企业智能知识库助手", page_icon="🤖", layout="wide")
st.title("🤖 企业智能知识库助手")
st.caption("基于 DeepSeek + LangChain + Chroma 的流式 RAG 问答系统")

# ================= 2. 初始化 Embedding 和 向量库 =================
load_dotenv()

class SiliconFlowEmbeddings(Embeddings):
    def __init__(self):
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        self.url = "https://api.siliconflow.cn/v1/embeddings"
        self.model = "BAAI/bge-m3"

    def _get_emb(self, text: str) -> list:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": text}
        resp = requests.post(self.url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get_emb(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._get_emb(text)

embeddings = SiliconFlowEmbeddings()
# 注意这里路径要和 agent_langchain.py 里的保持一致
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db_lc")
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ================= 3. 初始化大模型 =================
llm = ChatDeepSeek(
    model="deepseek-flash", 
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    streaming=True
)

# ================= 4. 渲染历史对话 =================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= 5. 核心交互逻辑（RAG + 流式） =================
if prompt := st.chat_input("请输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # ====== 核心：1. 检索知识库 ======
        with st.spinner("正在查阅公司制度..."):
            docs = retriever.invoke(prompt)
            context = "\n---\n".join([d.page_content for d in docs])
        
        # 👇 加上这段，把检索到的原文展示在网页上（面试演示神器）
        if docs:
            with st.expander("📚 点击查看检索到的公司制度原文"):
                st.write(context)
        else:
            st.warning("⚠️ 向量库里没有检索到任何内容！")
        
        # 更强硬的 Prompt，防止模型自由发挥
        system_prompt = f"""你是一个严格的企业规章制度问答助手。
你必须且只能根据下面提供的【参考资料】来回答用户问题。
绝对禁止使用你内置的其他知识。
如果【参考资料】中没有直接提到相关内容，你必须回答：“抱歉，公司制度中没有相关规定，建议咨询HR。”

【参考资料】
{context}
"""
        
        messages_to_send = [("system", system_prompt)]
        for m in st.session_state.messages:
            messages_to_send.append((m["role"], m["content"]))

        # ====== 核心：2. 流式调用 ======
        response_stream = llm.stream(messages_to_send)
        full_response = st.write_stream(response_stream)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})