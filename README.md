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
uv venv
uv sync
```

### 2. 启动服务器

```bash
uv run server.py
```

服务器地址: `http://127.0.0.1:12346`

---

## 三、工具详解

### 3.1 train_rnn_model

训练RNN回归模型。

**参数:**
| 参数 | 类型 | 必填 | 默认值 | 说明 |
| :-- | :--: | :--: | :-- | :-- |
| data | List[float] | ✅ | - | 时间序列数据 |
| epochs | int | - | 100 | 训练轮数 |
| sequence_length | int | - | 10 | 序列长度 |
| hidden_size | int | - | 64 | 隐藏层大小 |
| num_layers | int | - | 1 | 网络层数 |
| learning_rate | float | - | 0.01 | 学习率 |

**返回:**
```json
{
  "status": "success",
  "model_type": "rnn",
  "final_loss": 0.0234,
  "metrics": {"mse": 0.021, "rmse": 0.145, "mae": 0.112},
  "sample_predictions": [9.8, 10.2, 10.5],
  "sample_actuals": [10.0, 10.0, 11.0]
}
```

**示例:**
```python
# Python SDK
result = await client.train_rnn_model(
    data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    epochs=50,
    sequence_length=3,
    hidden_size=32
)
```

---

### 3.2 train_lstm_model

训练LSTM回归模型。参数与返回与RNN相同，适用于长序列预测。

**参数:** 同 train_rnn_model

**示例:**
```python
result = await client.train_lstm_model(
    data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    epochs=100,
    sequence_length=5,
    hidden_size=64
)
```

---

### 3.3 train_gru_model

训练GRU回归模型。参数与返回与RNN相同，参数量较少。

**参数:** 同 train_rnn_model

**示例:**
```python
result = await client.train_gru_model(
    data=[1.0, 2.0, 3.0, 4.0, 5.0],
    epochs=30,
    hidden_size=16
)
```

---

### 3.4 predict_model

使用已训练的模型进行预测。

**参数:**
| 参数 | 类型 | 必填 | 默认值 | 说明 |
| :-- | :--: | :--: | :-- | :-- |
| data | List[float] | ✅ | - | 待预测数据 |
| model_type | str | - | "lstm" | 模型类型 (rnn/lstm/gru) |

**返回:**
```json
{
  "status": "success",
  "model_type": "lstm",
  "predictions": [10.5, 11.2, 12.0],
  "prediction_count": 3
}
```

**错误返回:**
```json
{
  "status": "error",
  "message": "Model 'lstm' not trained yet. Please train the model first."
}
```

**示例:**
```python
# 先训练模型
await client.train_lstm_model(data=[1,2,3,4,5,6,7,8,9,10], epochs=50)

# 预测
result = await client.predict_model(
    data=[10.0, 11.0],
    model_type="lstm"
)
```

---

### 3.5 compare_models

对比三种模型的性能，自动选择最佳模型。

**参数:**
| 参数 | 类型 | 必填 | 默认值 | 说明 |
| :-- | :--: | :--: | :-- | :-- |
| data | List[float] | ✅ | - | 时间序列数据 |
| epochs | int | - | 100 | 训练轮数 |
| sequence_length | int | - | 10 | 序列长度 |
| hidden_size | int | - | 64 | 隐藏层大小 |

**返回:**
```json
{
  "status": "success",
  "results": {
    "rnn": {"mse": 0.031, "rmse": 0.176, "mae": 0.145},
    "lstm": {"mse": 0.018, "rmse": 0.134, "mae": 0.098},
    "gru": {"mse": 0.022, "rmse": 0.148, "mae": 0.112}
  },
  "best_model": "lstm",
  "best_mse": 0.018
}
```

**示例:**
```python
result = await client.compare_models(
    data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    epochs=100,
    sequence_length=3
)
print(f"最佳模型: {result['best_model']}")  # lstm
```

---

### 3.6 get_available_models

查看当前已训练的模型列表。

**参数:** 无

**返回:**
```json
{
  "status": "success",
  "available_models": ["lstm", "gru"],
  "model_count": 2
}
```

---

### 3.7 get_server_config

获取服务器配置。

**参数:**
| 参数 | 类型 | 必填 | 默认值 | 说明 |
| :-- | :--: | :--: | :-- | :-- |
| key | str | - | null | 配置键名 |
| default | Any | - | null | 默认值 |

---

## 四、大模型调用配置

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

asyncio.run(main())
```

---

## 五、配置说明

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

## 六、算法选择

| 场景 | 推荐 | 原因 |
| :-- | :-- | :-- |
| 短序列 | RNN | 简单快速 |
| 长序列 | LSTM | 记忆能力强 |
| 资源受限 | GRU | 参数量少 |
