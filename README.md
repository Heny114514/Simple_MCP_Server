# 时间序列预测MCP智能体

基于MCP (Model Context Protocol) 的时间序列预测智能体，支持RNN、LSTM、GRU三种深度学习回归算法。

### 组员信息

| 姓名 | 学号 | 分工 | 备注 |
| :--: | :--: | :--: | :--: |
| Heny(兰翔) | - | 智能体开发 | 人工智能引论25秋课程项目 |
| LovelyFlash(王李超) | - | 智能体开发 | 人工智能引论25秋课程项目 |

---

## 一、项目简介

### 1.1 选题背景

本项目是"人工智能引论25秋"课程项目，实现了基于循环神经网络的时间序列预测MCP Server。

**选题要求**: 从LSTM、RNN和GRU中选取两种回归算法进行智能体封装

**实现方案**: 完整实现了全部三种算法（RNN、LSTM、GRU），并提供了模型对比功能。

### 1.2 技术栈

| 技术 | 说明 |
| :--: | :-- |
| Python | 编程语言 |
| FastMCP | MCP Server框架 |
| PyTorch | 深度学习框架 |
| Uvicorn | ASGI服务器 |
| SSE/stdio | 传输协议 |

### 1.3 核心功能

本MCP Server提供以下核心功能：

1. **模型训练**: 支持RNN、LSTM、GRU三种深度学习模型的训练
2. **模型预测**: 使用训练好的模型进行时间序列预测
3. **模型对比**: 自动对比三种模型的性能，帮助选择最佳模型
4. **模型管理**: 查看当前已训练的模型列表

---

## 二、Tool 工具列表

### 2.1 时间序列预测工具

| 工具名称 | 功能描述 | 输入参数 | 输出 |
| :------: | :------: | :-- | :-- |
| train_rnn_model | 训练RNN回归模型 | data, epochs, sequence_length, hidden_size, num_layers, learning_rate | 训练结果和评估指标 |
| train_lstm_model | 训练LSTM回归模型 | data, epochs, sequence_length, hidden_size, num_layers, learning_rate | 训练结果和评估指标 |
| train_gru_model | 训练GRU回归模型 | data, epochs, sequence_length, hidden_size, num_layers, learning_rate | 训练结果和评估指标 |
| predict_model | 使用训练好的模型进行预测 | data, model_type | 预测结果 |
| compare_models | 对比RNN、LSTM、GRU三种模型效果 | data, epochs, sequence_length, hidden_size | 模型对比结果 |
| get_available_models | 获取当前已训练的模型列表 | 无 | 可用模型列表 |

### 2.2 配置工具

| 工具名称 | 功能描述 | 输入参数 | 输出 |
| :------: | :------: | :-- | :-- |
| get_server_config | 获取服务器的配置信息 | key, default | 配置值 |

### 2.3 参数说明

| 参数名称 | 类型 | 默认值 | 说明 |
| :--: | :--: | :--: | :-- |
| data | List[float] | 必填 | 时间序列数据列表 |
| epochs | int | 100 | 训练轮数 |
| sequence_length | int | 10 | 序列长度 |
| hidden_size | int | 64 | 隐藏层大小 |
| num_layers | int | 1 | 网络层数 |
| learning_rate | float | 0.01 | 学习率 |
| model_type | str | "lstm" | 模型类型 (rnn/lstm/gru) |

---

## 三、Resource 资源列表

| 资源名称 | 功能描述 | URI |
| :------: | :------: | :-- |
| readme | 项目README文档 | file:///README.md |
| server_logs | 服务器日志文件 | file:///logs/{path} |

---

## 四、Prompts 提示词列表

| 指令名称 | 功能描述 | 参数 |
| :------: | :------: | :-- |
| greet_user | 生成一个问候消息 | name |

---

## 五、环境配置与运行

### 5.1 环境要求

- Python 3.10+
- Windows / Linux / macOS

### 5.2 快速开始

```bash
# 1. 创建虚拟环境
uv venv

# 2. 激活虚拟环境（Windows）
venv\Scripts\activate

# 3. 激活虚拟环境（Linux/macOS）
source .venv/bin/activate

# 4. 安装依赖
uv sync

# 5. 启动服务器
uv run server.py
```

### 5.3 配置说明

修改 `config.yaml` 文件：

```yaml
server:
  name: TimeSeriesMCPServer
  author: Heny(兰翔) and LovelyFlash(王李超)
  version: 0.1.0

transport:
  type: "sse"      # 可选值: stdio, sse
  host: "127.0.0.1"
  port: 12346     # 端口号
```

**传输模式说明**:
- `sse`: SSE (Server-Sent Events) 模式，通过HTTP长连接通信
- `stdio`: 标准输入输出模式，适用于本地进程调用

---

## 六、大模型调用配置

### 6.1 什么是MCP？

MCP (Model Context Protocol) 是一个标准化协议，允许AI大模型与外部工具和服务进行交互。通过MCP，Claude、ChatGPT等大模型可以调用本项目提供的时间序列预测工具。

### 6.2 配置步骤

#### 方式一：使用Cursor/Cline配置

1. 启动MCP服务器：
```bash
uv run server.py
```

2. 在Cursor或Cline的配置文件中添加：

**Windows (cursor.json)**:
```json
{
  "mcpServers": {
    "timeseries-mcp": {
      "command": "cmd",
      "args": ["/c", "uv", "run", "server.py"],
      "env": {},
      "cwd": "D:\\desktop\\Simple_MCP_Server"
    }
  }
}
```

或者使用SSE模式：
```json
{
  "mcpServers": {
    "timeseries-mcp": {
      "url": "http://127.0.0.1:12346/sse"
    }
  }
}
```

#### 方式二：使用Claude Desktop

编辑配置文件：

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

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

#### 方式三：使用Python调用

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # 配置服务器参数
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", "D:\\desktop\\Simple_MCP_Server", "server.py"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 调用工具训练LSTM模型
            result = await session.call_tool(
                "train_lstm_model",
                {
                    "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    "epochs": 50,
                    "sequence_length": 3,
                    "hidden_size": 32
                }
            )
            print(result)

asyncio.run(main())
```

### 6.3 大模型调用示例

配置完成后，大模型可以自然语言调用工具：

**用户请求**: "用RNN模型预测接下来的5个数据点"

**模型调用流程**:
1. 模型识别需要使用 `train_rnn_model` 工具
2. 调用工具获取训练结果
3. 使用 `predict_model` 进行预测
4. 返回预测结果给用户

---

## 七、使用示例

### 7.1 训练模型

```python
import asyncio
from tools.time_series_tools import train_lstm_model

# 准备时间序列数据
data = [float(i + (i % 10) * 0.5) for i in range(100)]

async def main():
    result = await train_lstm_model(
        data=data,
        epochs=100,
        sequence_length=10,
        hidden_size=64,
        num_layers=2,
        learning_rate=0.01
    )
    print(f"训练状态: {result['status']}")
    print(f"最终损失: {result['final_loss']:.4f}")
    print(f"MSE: {result['metrics']['mse']:.4f}")
    print(f"RMSE: {result['metrics']['rmse']:.4f}")
    print(f"MAE: {result['metrics']['mae']:.4f}")

asyncio.run(main())
```

### 7.2 对比模型

```python
import asyncio
from tools.time_series_tools import compare_models

data = [float(i + (i % 10) * 0.5) for i in range(100)]

async def main():
    result = await compare_models(
        data=data,
        epochs=100,
        sequence_length=10,
        hidden_size=64
    )
    print(f"最佳模型: {result['best_model']}")
    print(f"最佳MSE: {result['best_mse']:.4f}")
    print("\n各模型性能对比:")
    for model, metrics in result['results'].items():
        print(f"  {model}: MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

asyncio.run(main())
```

### 7.3 模型预测

```python
import asyncio
from tools.time_series_tools import predict_model

# 先训练模型...
data = [float(i) for i in range(50)]
new_data = [float(i) for i in range(50, 60)]

async def main():
    result = await predict_model(
        data=new_data,
        model_type="lstm"
    )
    print(f"预测结果: {result['predictions']}")

asyncio.run(main())
```

---

## 八、项目结构

```
Simple_MCP_Server/
├── server.py                 # 服务器入口
├── config.yaml              # 配置文件
├── pyproject.toml           # 依赖管理
├── README.md                # 项目文档
│
├── tools/                   # 工具模块
│   ├── __init__.py          # 工具注册器
│   ├── hello_tool.py        # 配置获取工具
│   ├── time_series_models.py # 模型实现
│   └── time_series_tools.py  # 时间序列工具
│
├── prompts/                  # 提示词模块
│   ├── __init__.py
│   └── hello_prompt.py
│
├── resources/               # 资源模块
│   ├── __init__.py
│   └── hello_resource.py
│
├── core/                    # 核心模块
│   └── hello_secrets.py
│
└── modules/                 # 公共模块
    └── YA_Common/
        └── utils/
            ├── config.py
            ├── logger.py
            └── ...
```

---

## 九、算法说明

### 9.1 RNN (循环神经网络)

RNN是最基础的循环神经网络，通过隐藏状态将历史信息传递到当前时刻。

- **结构**: 输入层 → 隐藏层(循环) → 输出层
- **优点**: 结构简单、参数量少、训练速度快
- **缺点**: 存在梯度消失/爆炸问题，难以处理长序列

### 9.2 LSTM (长短期记忆网络)

LSTM通过引入门控机制解决了RNN的长期依赖问题。

- **核心**: 输入门、遗忘门、输出门
- **优点**: 能处理长序列、长期记忆能力强
- **缺点**: 参数量较大、训练较慢

### 9.3 GRU (门控循环单元)

GRU是LSTM的简化版本，将门数量从3个减少到2个。

- **核心**: 更新门、重置门
- **优点**: 参数量少、训练速度快、结构简单
- **缺点**: 表达能力略弱于LSTM

### 9.4 模型选择建议

| 场景 | 推荐模型 | 原因 |
| :--: | :--: | :-- |
| 短序列 | RNN | 简单高效 |
| 长序列 | LSTM | 记忆能力强 |
| 资源受限 | GRU | 参数量少 |
| 追求精度 | LSTM | 表达能力强 |
| 快速原型 | GRU | 训练快 |

---

## 十、评估指标

本项目使用以下指标评估模型性能：

| 指标 | 名称 | 说明 |
| :--: | :-- | :-- |
| MSE | 均方误差 | 预测值与真实值差的平方的均值 |
| RMSE | 均方根误差 | MSE的平方根 |
| MAE | 平均绝对误差 | 预测值与真实值差的绝对值的均值 |

**指标解读**: 值越小表示模型预测越准确

---

## 十一、常见问题

### Q1: 如何切换传输模式？

修改 `config.yaml` 中的 `transport.type` 为 `stdio` 或 `sse`。

### Q2: 端口被占用怎么办？

修改 `config.yaml` 中的 `port` 为其他端口号。

### Q3: 如何查看日志？

日志文件保存在 `logs/` 目录下，可通过 `server_logs` 资源访问。

### Q4: 模型训练时间过长怎么办？

- 减少 `epochs`
- 减小 `hidden_size`
- 减小 `num_layers`
- 增加 `learning_rate`

---

## 十二、许可与鸣谢

- 课程: 人工智能引论 25秋
- 框架: FastMCP / PyTorch
- 作者: Heny(兰翔), LovelyFlash(王李超)
