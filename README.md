# 时间序列预测MCP服务器

基于MCP (Model Context Protocol) 的时间序列预测服务，支持RNN、LSTM、GRU三种深度学习算法。

**作者**: Heny(兰翔)、LovelyFlash(王李超)  
**课程**: 人工智能引论 25秋

---

## 一、项目结构

```
Simple_MCP_Server/
├── server.py              # 服务器入口
├── config.yaml           # 配置文件
├── pyproject.toml        # 依赖管理
├── README.md             # 本文档
│
├── tools/                # MCP工具
│   ├── time_series_models.py  # RNN/LSTM/GRU模型实现
│   └── time_series_tools.py  # 工具函数
│
├── prompts/              # 提示词模板
├── resources/           # 资源文件
├── core/                # 核心模块
├── utils/               # 公共工具
│   ├── config.py       # 配置管理
│   ├── logger.py       # 日志
│   └── security.py    # 安全认证
│
├── examples/            # 示例代码
│   └── sdk_client.py   # Python SDK
│
├── deploy.sh           # Linux部署脚本
└── deploy.bat          # Windows部署脚本
```

---

## 二、快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
uv venv

# 安装依赖
uv sync
```

### 2. 启动服务器

```bash
uv run server.py
```

服务器启动后访问 `http://127.0.0.1:12346`

---

## 三、可用工具

### 模型训练

| 工具 | 功能 |
| :-- | :-- |
| train_rnn_model | 训练RNN模型 |
| train_lstm_model | 训练LSTM模型 |
| train_gru_model | 训练GRU模型 |

### 模型预测

| 工具 | 功能 |
| :-- | :-- |
| predict_model | 使用训练好的模型预测 |
| compare_models | 对比三种模型性能 |
| get_available_models | 查看已训练的模型 |

---

## 四、参数说明

| 参数 | 类型 | 默认值 | 说明 |
| :-- | :-- | :-- | :-- |
| data | List[float] | 必填 | 时间序列数据 |
| epochs | int | 100 | 训练轮数 |
| sequence_length | int | 10 | 序列长度 |
| hidden_size | int | 64 | 隐藏层大小 |
| learning_rate | float | 0.01 | 学习率 |
| model_type | str | "lstm" | 模型类型 |

---

## 五、大模型调用配置

### Cursor/Cline

```json
{
  "mcpServers": {
    "timeseries-mcp": {
      "url": "http://127.0.0.1:12346/sse"
    }
  }
}
```

### Claude Desktop

`%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "timeseries-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "D:\\desktop\\Simple_MCP_Server", "server.py"]
    }
  }
}
```

### Python SDK

```python
from examples import TimeSeriesMCPClient
import asyncio

async def main():
    async with TimeSeriesMCPClient(cwd=".") as client:
        # 训练模型
        result = await client.train_lstm(data=[1,2,3,4,5], epochs=50)
        
        # 预测
        pred = await client.predict(data=[6,7], model_type="lstm")
        print(pred)

asyncio.run(main())
```

---

## 六、配置说明

修改 `config.yaml`:

```yaml
server:
  name: TimeSeriesMCPServer
  version: 0.1.0

transport:
  type: "sse"      # stdio 或 sse
  port: 12346

security:
  enabled: false   # 启用安全认证
  api_key: "your-key"
```

---

## 七、算法选择

| 场景 | 推荐 | 原因 |
| :-- | :-- | :-- |
| 短序列 | RNN | 简单快速 |
| 长序列 | LSTM | 记忆能力强 |
| 资源受限 | GRU | 参数量少 |
