# 时间序列预测MCP智能体

基于MCP (Model Context Protocol) 的时间序列预测智能体，支持RNN、LSTM、GRU三种深度学习回归算法。

### 组员信息

| 姓名 | 学号 | 分工 | 备注 |
| :--: | :--: | :--: | :--: |
| 莫益军 | - | 智能体开发 | 人工智能引论25秋课程项目 |

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

---

### Resource 列表

| 资源名称 | 功能描述 |
| :------: | :------: |
| server_info | 服务器信息 |
| server_config | 服务器配置信息 |

---

### Prompts 列表

| 指令名称 | 功能描述 |
| :------: | :------: |
| hello_prompt | 欢迎提示词 |

---

### 使用方法

#### 1. 环境准备

```bash
# 安装依赖
pip install torch numpy pandas scikit-learn matplotlib
pip install mcp uvicorn starlette sse-starlette
pip install colorlog art
```

#### 2. 启动MCP Server

```bash
# 方式一：SSE模式（推荐）
python server.py

# 或设置端口
HOST=127.0.0.1 PORT=12345 python server.py

# 方式二：Stdio模式
python -c "from server import YA_MCPServer; YA_MCPServer().run_stdio()"
```

#### 3. 使用MCP工具

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

#### 4. 在Claude等AI助手中使用

配置MCP Server后，可直接调用以下工具：

- 使用 `train_lstm_model` 训练时间序列预测模型
- 使用 `compare_models` 对比不同模型效果
- 使用 `predict_model` 进行预测

---

### 项目结构与文件作用

| 文件/目录 | 作用 | 说明 |
| :-------: | :--: | :--: |
| **核心文件** | | |
| `server.py` | 服务器入口 | MCP Server启动与配置 |
| `tools/time_series_models.py` | 模型核心类 | RNN、LSTM、GRU回归模型实现 |
| `tools/time_series_tools.py` | MCP工具接口 | 6个时间序列预测相关工具 |
| `config.yaml` | 配置文件 | 服务器配置、日志设置等 |
| `pyproject.toml` | 依赖管理 | 项目依赖包版本管理 |
| `setup.py` | 安装脚本 | 项目安装配置 |
| `README.md` | 项目文档 | 本文档 |
| `PROJECT_PLAN.md` | 项目计划 | 开发进度与任务清单 |
| **功能模块** | | |
| `modules/YA_Common/` | 公共模块 | 配置、日志、MCP客户端等工具 |
| `modules/YA_Common/utils/` | 工具函数 | 配置读取、日志记录等 |
| `modules/YA_Common/mcp/` | MCP适配器 | OpenAI适配器等 |
| **模板文件** | | 可删除 |
| `core/` | 核心模块 | 模板代码，未使用 |
| `prompts/` | 提示词模块 | 模板提示词，未使用 |
| `resources/` | 资源模块 | 模板资源，未使用 |
| `tools/hello_tool.py` | 示例工具 | 模板工具，未使用 |
| **文档** | | 可删除 |
| `docs/` | 开发文档 | MCP开发指南 |
| **密钥管理** | | 可删除 |
| `modules/YA_Secrets/` | 密钥管理 | 未使用 |
| **平台脚本** | | 可删除 |
| `linux-macos.*.sh` | Linux脚本 | 未使用 |
| `windows.*.ps1` | Windows脚本 | 未使用 |

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
