# 企业智能知识库助手 (Enterprise RAG Agent)

基于 LangChain 与 Streamlit 构建的流式 RAG（检索增强生成）问答系统。本项目结合了 DeepSeek 大模型、Chroma 向量数据库以及硅基流动 Embedding 模型，实现了企业私有知识库的精准检索与逐字流式回答。

## 🌟 核心功能
- **流式输出**：前端基于 Streamlit，后端基于 LangChain，实现类似 ChatGPT 的打字机式回答体验。
- **RAG 防幻觉**：严格约束 Prompt，强制模型仅基于检索到的知识库内容作答，并支持展开查看引用原文。
- **向量检索**：接入 BAAI/bge-m3 向量模型与 Chroma 本地向量库，实现毫秒级语义检索。
- **多轮对话**：支持 Session 状态管理，保留上下文记忆。

## 🛠️ 技术栈
- **大模型**: DeepSeek API (`deepseek-flash`)
- **框架**: LangChain
- **前端**: Streamlit
- **向量库**: ChromaDB
- **Embedding**: 硅基流动 API (`BAAI/bge-m3`)

## 🚀 本地运行指南
1. 克隆项目并安装依赖：
   ```bash
   pip install -r requirements.txt