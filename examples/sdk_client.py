"""
TimeSeriesMCP Python SDK

简化大模型调用MCP服务器的Python客户端封装
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client


class TimeSeriesMCPClient:
    """时间序列预测MCP客户端"""
    
    def __init__(
        self,
        command: str = "uv",
        args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        sse_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        初始化客户端
        
        Args:
            command: 运行命令 (默认: uv)
            args: 命令参数
            cwd: 工作目录
            sse_url: SSE模式下的服务器URL
            api_key: API密钥 (如启用安全认证)
        """
        self.command = command
        self.args = args or ["run", "server.py"]
        self.cwd = cwd
        self.sse_url = sse_url
        self.api_key = api_key
        self.session: Optional[ClientSession] = None
    
    async def connect(self):
        """建立连接"""
        if self.sse_url:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            async with sse_client(self.sse_url, headers=headers) as (read, write):
                self.session = ClientSession(read, write)
                await self.session.initialize()
        else:
            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                cwd=self.cwd,
            )
            async with stdio_client(server_params) as (read, write):
                self.session = ClientSession(read, write)
                await self.session.initialize()
    
    async def close(self):
        """关闭连接"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def train_rnn(
        self,
        data: List[float],
        epochs: int = 100,
        sequence_length: int = 10,
        hidden_size: int = 64,
        num_layers: int = 1,
        learning_rate: float = 0.01,
    ) -> Dict[str, Any]:
        """训练RNN模型"""
        return await self.session.call_tool(
            "train_rnn_model",
            {
                "data": data,
                "epochs": epochs,
                "sequence_length": sequence_length,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "learning_rate": learning_rate,
            }
        )
    
    async def train_lstm(
        self,
        data: List[float],
        epochs: int = 100,
        sequence_length: int = 10,
        hidden_size: int = 64,
        num_layers: int = 1,
        learning_rate: float = 0.01,
    ) -> Dict[str, Any]:
        """训练LSTM模型"""
        return await self.session.call_tool(
            "train_lstm_model",
            {
                "data": data,
                "epochs": epochs,
                "sequence_length": sequence_length,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "learning_rate": learning_rate,
            }
        )
    
    async def train_gru(
        self,
        data: List[float],
        epochs: int = 100,
        sequence_length: int = 10,
        hidden_size: int = 64,
        num_layers: int = 1,
        learning_rate: float = 0.01,
    ) -> Dict[str, Any]:
        """训练GRU模型"""
        return await self.session.call_tool(
            "train_gru_model",
            {
                "data": data,
                "epochs": epochs,
                "sequence_length": sequence_length,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "learning_rate": learning_rate,
            }
        )
    
    async def predict(
        self,
        data: List[float],
        model_type: str = "lstm",
    ) -> Dict[str, Any]:
        """使用模型预测"""
        return await self.session.call_tool(
            "predict_model",
            {
                "data": data,
                "model_type": model_type,
            }
        )
    
    async def compare_models(
        self,
        data: List[float],
        epochs: int = 100,
        sequence_length: int = 10,
        hidden_size: int = 64,
    ) -> Dict[str, Any]:
        """对比模型性能"""
        return await self.session.call_tool(
            "compare_models",
            {
                "data": data,
                "epochs": epochs,
                "sequence_length": sequence_length,
                "hidden_size": hidden_size,
            }
        )
    
    async def get_available_models(self) -> Dict[str, Any]:
        """获取可用模型列表"""
        return await self.session.call_tool("get_available_models", {})
    
    async def get_config(self, key: str, default: Any = None) -> Dict[str, Any]:
        """获取服务器配置"""
        return await self.session.call_tool(
            "get_server_config",
            {"key": key, "default": default}
        )


async def quick_train_lstm(data: List[float], epochs: int = 50) -> Dict[str, Any]:
    """
    快速训练LSTM模型的便捷函数
    
    Args:
        data: 时间序列数据
        epochs: 训练轮数
    
    Returns:
        训练结果
    """
    async with TimeSeriesMCPClient() as client:
        result = await client.train_lstm(data=data, epochs=epochs)
        return result


async def quick_predict(data: List[float], model_type: str = "lstm") -> List[float]:
    """
    快速预测的便捷函数
    
    Args:
        data: 待预测数据
        model_type: 模型类型
    
    Returns:
        预测结果
    """
    async with TimeSeriesMCPClient() as client:
        result = await client.predict(data=data, model_type=model_type)
        return result.get("predictions", [])


if __name__ == "__main__":
    async def demo():
        data = [float(i + (i % 10) * 0.5) for i in range(100)]
        
        async with TimeSeriesMCPClient(cwd=".") as client:
            result = await client.train_lstm(data, epochs=30)
            print(f"训练结果: {result}")
            
            models = await client.get_available_models()
            print(f"可用模型: {models}")
    
    asyncio.run(demo())
