import os
import subprocess
import sys
import csv
import requests
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

load_dotenv()

# ================= 1. 自定义硅基流动的 Embedding 组件 =================
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

# 初始化向量库
embeddings = SiliconFlowEmbeddings()
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db_lc")

# ================= 2. 定义工具集 =================
@tool
def build_knowledge_base(file_path: str = "knowledge.txt") -> str:
    """读取本地文本文件，存入向量数据库。"""
    global vectorstore  # 声明使用全局变量
    
    if not os.path.exists(file_path):
        return f"文件不存在: {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    
    # 清空旧数据并重新初始化
    vectorstore.delete_collection()
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db_lc")
    
    vectorstore.add_texts(chunks)
    return f"成功存入 {len(chunks)} 个知识片段"

@tool
def search_knowledge(query: str) -> str:
    """在本地知识库中检索相关的内容。"""
    docs = vectorstore.similarity_search(query, k=2)
    if not docs:
        return "没有检索到相关内容"
    return "\n---\n".join([d.page_content for d in docs])

@tool
def read_file(path: str) -> str:
    """读取指定路径的文本文件内容。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()[:4000]
    except Exception as e:
        return f"读取失败：{e}"

@tool
def write_file(path: str, content: str) -> str:
    """把内容写入文件。"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已成功写入 {path}"
    except Exception as e:
        return f"写入失败：{e}"

@tool
def analyze_csv(path: str) -> str:
    """读取 CSV 文件并计算第二列的总和与最大值。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2: return "文件为空"
        values = [int(r[1]) for r in rows[1:] if len(r) > 1]
        return f"总和：{sum(values)}，最大值：{max(values)}"
    except Exception as e:
        return f"分析失败：{e}"

@tool
def run_python(code: str) -> str:
    """执行一段 Python 代码并返回输出。"""
    if "open(" not in code and code.count(",") > 20:
        return "提示：不要硬编码数据，请用 open() 读取。"
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10, encoding="utf-8")
        return result.stdout or result.stderr
    except Exception as e:
        return f"执行失败：{e}"

# 把所有工具打包成一个列表
tools = [read_file, write_file, analyze_csv, run_python, search_knowledge, build_knowledge_base]

# ================= 3. 初始化大模型并绑定工具 =================
llm = ChatDeepSeek(model="deepseek-flash", api_key=os.getenv("DEEPSEEK_API_KEY"))
llm_with_tools = llm.bind_tools(tools)
tools_map = {t.name: t for t in tools}

# ================= 4. LangChain 版 Agent 主循环 =================
def run_agent(user_input: str, verbose: bool = True) -> str:
    messages = [
        SystemMessage(content="你是一个本地助手，负责读写文件、分析数据和检索知识库。回答公司规定时请先用 search_knowledge 检索。"),
        HumanMessage(content=user_input)
    ]

    for step in range(8):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # 如果模型不调工具了，直接返回答案
        if not response.tool_calls:
            return response.content

        # 如果模型决定调工具
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if verbose:
                print(f"  [第 {step + 1} 步] 调用工具：{tool_name}，参数：{tool_args}")
            
            # 执行工具
            tool_result = tools_map[tool_name].invoke(tool_args)
            
            if verbose:
                print(f"           返回：{str(tool_result)[:150]}...")

            # 把工具结果塞回对话中
            messages.append({
                "role": "tool",
                "content": str(tool_result),
                "tool_call_id": tool_call["id"]
            })

    return "任务步数超过上限"

# ================= 5. 命令行入口 =================
if __name__ == "__main__":
    print("=" * 50)
    print("LangChain 版本地助手（输入 exit 退出）")
    print("=" * 50)
    while True:
        try:
            user_input = input("\n你：").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue
            print("Agent 正在思考...")
            print(f"\nAgent：{run_agent(user_input)}")
        except KeyboardInterrupt:
            break