# 时间序列预测MCP智能体

基于MCP (Model Context Protocol) 的时间序列预测智能体，支持RNN、LSTM、GRU三种深度学习回归算法。

### 组员信息

| 姓名 | 学号 | 分工 | 备注 |
| :--: | :--: | :--: | :--: |
| Heny | - | 智能体开发 | 人工智能引论25秋课程项目 |
| LovelyFlash | - | 智能体开发 | 人工智能引论25秋课程项目 |

### 项目简介

本项目实现了基于循环神经网络的时间序列预测MCP Server，符合选题要求：
- **选题**: 从LSTM、RNN和GRU中选取两种回归算法进行智能体封装
- **技术栈**: Python + FastMCP + PyTorch
- **算法模型**: RNN、LSTM、GRU (3种)

---

### Tool 列表

| 工具名称 | 功能描述 | 输入 | 输出 |
| :------: | :------: | :--: | :--: |
| train_rnn_model | 训练RNN回归模型 | data: List[float], epochs, sequence_length, hidden_size | 训练结果和评估指标 |
| train_lstm_model | 训练LSTM回归模型 | data: List[float], epochs, sequence_length, hidden_size | 训练结果和评估指标 |
| train_gru_model | 训练GRU回归模型 | data: List[float], epochs, sequence_length, hidden_size | 训练结果和评估指标 |
| predict_model | 使用训练好的模型进行预测 | data: List[float], model_type | 预测结果 |
| compare_models | 对比RNN、LSTM、GRU三种模型效果 | data, epochs, sequence_length, hidden_size | 模型对比结果 |
| get_available_models | 获取当前已训练的模型列表 | 无 | 可用模型列表 |
| get_server_config | 获取服务器的配置信息 | key, default | 配置值 |

---

### Resource 列表

| 资源名称 | 功能描述 |
| :------: | :------: |
| server_info | 服务器信息 |
| server_config | 服务器配置信息 |
| readme | 项目README文档 |

---

### Prompts 列表

| 指令名称 | 功能描述 |
| :------: | :------: |
| hello_prompt | 生成一个问候消息 |

---

## 环境配置与运行

### 1. 创建并激活虚拟环境

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境（Windows）
venv\Scripts\activate

# 激活虚拟环境（Linux/macOS）
source .venv/bin/activate
```

### 2. 安装依赖

```bash
# 安装项目依赖
uv sync
```

### 3. 运行MCP Server

```bash
# 默认使用SSE模式启动服务器
uv run server.py

# 或直接运行
python server.py
```

**默认配置**:
- 地址: `127.0.0.1:12345
- 传输模式: SSE

### 4. 修改传输模式

如需切换到标准输入输出模式，修改 `config.yaml` 中的 `transport.type`:

```yaml
transport:
  type: "stdio"  # 可选值: stdio, sse
```

---

### 使用MCP工具

**示例：训练LSTM模型**

```python
import asyncio
from tools.time_series_tools import train_lstm_model, compare_models

# 生成测试数据
data = [float(i + (i % 10) * 0.5) for i in range(100)]

# 训练LSTM模型
async def main():
    result = await train_lstm_model(
        data=data,
        epochs=50,
        sequence_length=10,
        hidden_size=32
    )
    print(result)

asyncio.run(main())
```

**示例：对比模型效果**

```python
async def compare():
    data = [float(i + (i % 10) * 0.5) for i in range(100)]
    result = await compare_models(
        data=data,
        epochs=50,
        sequence_length=10,
        hidden_size=32
    )
    print(f"最佳模型: {result['best_model']}")
    print(f"各模型MSE: {result['results']}")

asyncio.run(compare())
```

---

### 项目结构与文件作用

| 文件/目录 | 作用 | 说明 |
| :-------: | :--: | :--: |
| **核心文件** | | |
| `server.py` | 服务器入口 | MCP Server启动与配置 |
| `tools/time_series_models.py` | 模型核心类 | RNN、LSTM、GRU回归模型实现 |
| `tools/time_series_tools.py` | MCP工具接口 | 时间序列预测相关工具 |
| `tools/hello_tool.py` | 示例工具 | 获取服务器配置信息 |
| `resources/hello_resource.py` | 资源文件 | 返回项目README、日志等 |
| `prompts/hello_prompt.py` | 提示词 | 生成问候消息 |
| `core/hello_secrets.py` | 密钥示例 | SOPS加密使用示例 |
| **配置** | | |
| `config.yaml` | 配置文件 | 服务器配置、日志设置等 |
| `pyproject.toml` | 依赖管理 | 项目依赖包版本管理 |
| `env.yaml` | 环境变量 | 密钥管理配置 |
| **公共模块** | | |
| `modules/YA_Common/` | 公共模块 | 配置、日志、MCP客户端等工具 |

---

### 算法说明

#### RNN (循环神经网络)
- 基础循环结构，处理序列数据
- 优点：结构简单、参数量少
- 缺点：存在梯度消失问题

#### LSTM (长短期记忆网络)
- 引入门控机制，解决长序列依赖问题
- 优点：能处理长序列、长期记忆
- 缺点：参数量较大、训练较慢

#### GRU (门控循环单元)
- LSTM的简化版本，合并了门控
- 优点：参数量少、训练速度快
- 缺点：表达能力略弱于LSTM

---

### 其他说明

- 使用了 **PyTorch** 深度学习框架
- 模型评估指标：MSE、RMSE、MAE
- 支持自定义超参数：epochs、learning_rate、hidden_size、num_layers
- 符合MCP Server规范，可与Claude等AI助手集成
