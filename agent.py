import json
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOLS, TOOL_FUNCTIONS

# ========== 日志配置 ==========
logging.basicConfig(
    filename="agent.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8"
)
logger = logging.getLogger(__name__)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

SYSTEM_PROMPT = """你是一个本地文件助手 Agent。

你可以调用以下工具：
- read_file：读取文件内容
- write_file：把内容写入文件
- analyze_csv：分析 CSV 文件，计算第二列的总和与最大值
- run_python：执行纯 Python 计算

工作原则：
1. 先理解用户目标，再决定调用哪个工具。
2. 需要多步完成时，一步一步来，每次只调必要的工具。
3. 工具返回结果后，判断任务是否完成；没完成就继续调工具。
4. 任务完成后，用中文简洁地告诉用户结果。
5. 如果工具报错，尝试换一种方式，不要重复同样的错误调用。

【重要规则】
- 分析 CSV 文件时，优先使用 analyze_csv 工具，不要自己写 Python 读文件。
- 不要把文件内容硬编码进任何代码或参数里。
- 需要处理文件时，直接把文件路径传给对应工具。
"""

MAX_STEPS = 8


def execute_tool(name: str, args: dict) -> str:
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        logger.error(f"未知工具：{name}")
        return f"错误：未知工具 {name}"
    try:
        return func(**args)
    except Exception as e:
        logger.error(f"工具 {name} 执行异常：{e}")
        return f"工具执行异常：{e}"


def run_agent(user_input: str, verbose: bool = True) -> str:
    logger.info("=" * 60)
    logger.info(f"用户输入：{user_input}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    for step in range(MAX_STEPS):
        try:
            response = client.chat.completions.create(
                model="deepseek-flash",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )
        except Exception as e:
            logger.error(f"API 调用失败：{e}")
            return f"API 调用失败：{e}"

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                logger.info(f"第 {step + 1} 步 | 调用工具：{name} | 参数：{args}")

                if verbose:
                    print(f"  [第 {step + 1} 步] 调用工具：{name}，参数：{args}")

                result = execute_tool(name, args)

                logger.info(f"第 {step + 1} 步 | 返回：{str(result)[:300]}")

                if verbose:
                    preview = str(result)[:200].replace("\n", " ")
                    print(f"           返回：{preview}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
        else:
            logger.info(f"Agent 最终回复：{msg.content}")
            return msg.content

    logger.warning("步数超过上限，已停止")
    return "任务步数超过上限，已停止。请把任务拆得更简单一些。"