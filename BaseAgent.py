"""
Hello Agents LLM 模块

为本书 "Hello Agents" 定制的LLM客户端模块。
提供一个统一的、可复用的大语言模型客户端接口。
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# 自动加载 .env 文件中的环境变量
load_dotenv()


class BaseAgent:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        
        Args:
            model: 模型ID，如未提供则从 LLM_MODEL_ID 环境变量加载
            apiKey: API密钥，如未提供则从 LLM_API_KEY 环境变量加载
            baseUrl: 服务地址，如未提供则从 LLM_BASE_URL 环境变量加载
            timeout: 超时时间（秒），如未提供则从 LLM_TIMEOUT 环境变量加载，默认60秒
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        
        Args:
            messages: 消息列表，格式为 [{"role": "user/system/assistant", "content": "..."}]
            temperature: 温度参数，控制输出的随机性，默认为0（确定性输出）
            
        Returns:
            模型的响应文本，如果发生错误则返回None
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None
