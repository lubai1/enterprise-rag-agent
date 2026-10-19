import os
import subprocess
import sys
import csv
from rag_tool import search_knowledge, build_knowledge_base

def read_file(path: str) -> str:
    """读取指定路径的文本文件内容。"""
    if not os.path.exists(path):
        return f"错误：文件不存在 -> {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 4000:
            return content[:4000] + "\n...(内容过长已截断)"
        return content
    except Exception as e:
        return f"读取失败：{e}"

def write_file(path: str, content: str) -> str:
    """把 content 写入指定路径的文件。"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已成功写入 {path}"
    except Exception as e:
        return f"写入失败：{e}"

def analyze_csv(path: str) -> str:
    """读取 CSV 文件，计算指定数值列的总和与最大值。"""
    if not os.path.exists(path):
        return f"错误：文件不存在 -> {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) < 2:
            return "文件为空或只有表头"

        header = rows[0]
        data_rows = rows[1:]

        values = []
        names = []
        for r in data_rows:
            if len(r) < 2:
                continue
            try:
                values.append(int(r[1]))
                names.append(r[0])
            except ValueError:
                continue

        if not values:
            return "没有可解析的数值"

        total = sum(values)
        max_idx = values.index(max(values))
        max_name = names[max_idx]
        max_value = values[max_idx]

        return (
            f"列名：{header[1] if len(header) > 1 else '第二列'}\n"
            f"总和：{total}\n"
            f"最大值：{max_value}（{max_name}）"
        )
    except Exception as e:
        return f"分析失败：{e}"

def run_python(code: str) -> str:
    """执行一段 Python 代码并返回标准输出。"""
    if "open(" not in code and code.count(",") > 20:
        return "提示：检测到可能把文件内容硬编码进了代码。请改用 open() 或专用工具。"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8"
        )
        output = result.stdout.strip()
        error = result.stderr.strip()
        if error:
            return f"执行出错：\n{error}"
        return output if output else "执行成功，无输出"
    except subprocess.TimeoutExpired:
        return "执行超时（超过 10 秒）"
    except Exception as e:
        return f"执行失败：{e}"

TOOL_FUNCTIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "analyze_csv": analyze_csv,
    "run_python": run_python,
    "search_knowledge": search_knowledge,
    "build_knowledge_base": build_knowledge_base,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定路径的文本文件内容。当用户要求查看、分析某个文件时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要读取的文件路径"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "把指定内容写入指定路径的文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要写入的文件路径"},
                    "content": {"type": "string", "description": "要写入的完整内容"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_csv",
            "description": "读取 CSV 文件并计算第二列数值的总和与最大值。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要分析的 CSV 文件路径"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "执行一段 Python 代码并返回输出。仅用于纯计算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的 Python 代码"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "build_knowledge_base",
            "description": "读取本地文本文件，将其切分并存入向量知识库。初始化知识库时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "知识库文本文件路径，默认 knowledge.txt"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在本地知识库中检索相关的内容。回答规章制度、说明文档等问题时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的问题或关键词"}
                },
                "required": ["query"]
            }
        }
    }
]