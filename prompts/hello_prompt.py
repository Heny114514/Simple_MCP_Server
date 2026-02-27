from typing import Any, Dict

from prompts import TimeSeriesMCPServer_Prompt


@TimeSeriesMCPServer_Prompt(
    name="greet_user",
    title="Greeting Prompt",
    description="生成一个问候消息",
)
async def hello_prompt(name: str) -> Dict[str, Any]:
    """当用户发起问候时，调用此prompt。
    参数name默认为"用户"

    Args:
        name (str): 用户的名字。

    Returns:
        Dict[str, str]: 包含问候语的字典。

    Example:
        {
            "greeting": "Hello, Alice!"
        }
    """
    return f"你好，{name}！欢迎使用时间序列预测MCP服务器。"
