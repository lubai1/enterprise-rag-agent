from agent import run_agent


def main():
    print("=" * 50)
    print("本地文件助手 Agent（输入 exit 退出）")
    print("=" * 50)
    print("示例：")
    print("  - 读一下下 data.csv 的内容")
    print("  - 帮我算一下 1 到 100 的和")
    print("  - 读 data.csv，算一下销售额总和，写到 result.txt")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n你：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "退出"):
            print("再见！")
            break

        print("Agent 正在思考...")
        answer = run_agent(user_input)
        print(f"\nAgent：{answer}")


if __name__ == "__main__":
    main()