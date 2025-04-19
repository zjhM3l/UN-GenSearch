import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Deepseek API configuration
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {deepseek_api_key}",
    "Content-Type": "application/json"
}


def call_deepseek_api(messages, model="deepseek-chat", temperature=0.2, max_tokens=1024):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        resp = requests.post(deepseek_api_url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Deepseek API call failed: {e}")
        return None


def get_original_response(user_question):
    """获取原始版本响应并写入文件"""
    # 原始版本系统提示
    system_prompt = "You are a helpful assistant that provides general knowledge and information."

    # 消息结构
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    # 获取响应
    response = call_deepseek_api(messages)

    # 解析结果
    result = "Error: No response received"
    if response and "choices" in response:
        result = response["choices"][0]["message"]["content"]

    # 写入文件（UTF-8编码）
    try:
        with open("original_response.txt", "w", encoding="utf-8") as f:
            f.write(result)
    except Exception as e:
        print(f"文件写入失败: {e}")

    return result


# 使用示例
if __name__ == "__main__":
    # 示例问题
    test_question = "What was the outcome of UN vote on nuclear disarmament?"

    # 获取并保存原始版本响应
    original_response = get_original_response(test_question)

    # 输出结果
    print("=== Original Version Response ===")
    print(original_response)