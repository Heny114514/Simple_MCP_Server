# 时间序列预测MCP服务器

基于MCP (Model Context Protocol) 的时间序列预测服务，支持RNN、LSTM、GRU三种深度学习算法。

## 作者: `Heny(兰翔)`、`LovelyFlash(王李超)`

## 课程: `人工智能引论 25秋`

## 工具列表：

### 工具 (Tools)
- `train_rnn_model` - 训练RNN回归模型
- `train_lstm_model` - 训练LSTM回归模型
- `train_gru_model` - 训练GRU回归模型
- `predict_model` - 使用已训练模型进行预测
- `compare_models` - 对比三种模型性能并自动选择最佳
- `get_available_models` - 获取当前已训练可用的模型列表
- `get_server_config` - 获取服务器配置信息

### 资源 (Resources)
- `hello_resource` - 示例MCP资源

### 提示词 (Prompts)
- `greet_user` - 问候语生成提示
- `predict_workflow` - 时间序列预测完整工作流提示


---

## 一、项目结构

```
Simple_MCP_Server/
├── server.py              # 服务器入口
├── setup.py              # 安装脚本
├── config.yaml           # 配置文件
├── pyproject.toml        # 依赖管理
├── .gitignore           # Git忽略文件
├── README.md             # 本文档
│
├── tools/                # MCP工具
│   ├── __init__.py       # 工具包初始化
│   ├── hello_tool.py     # 示例工具
│   ├── time_series_models.py  # RNN/LSTM/GRU模型实现
│   └── time_series_tools.py  # 时间序列工具函数
│
├── prompts/              # MCP提示词模板
│   ├── __init__.py       # 提示包初始化
│   ├── hello_prompt.py   # 示例提示
│   └── predict_prompt.py # 时间序列预测工作流提示
│
├── resources/            # MCP资源
│   ├── __init__.py       # 资源包初始化
│   └── hello_resource.py # 示例资源
│
├── core/                 # 核心功能模块
│   ├── __init__.py       # 核心包初始化
│   └── hello_secrets.py  # 示例secrets管理
│
├── utils/                # 公共工具模块
│   ├── __init__.py       # 工具包初始化
│   ├── config.py         # 配置管理
│   ├── logger.py         # 日志管理
│   ├── errors.py         # 错误处理
│   ├── helpers.py        # 辅助函数
│   ├── middleware.py     # 中间件
│   ├── security.py       # 安全模块
│   └── system.py         # 系统工具
│
├── examples/             # 示例代码
│   ├── __init__.py       # 示例包初始化
│   └── sdk_client.py     # Python SDK客户端
│
├── logs/                 # 日志文件目录
├── modules/              # 模块目录
│
├── deploy.sh             # Linux部署脚本
└── deploy.bat            # Windows部署脚本
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

### 3.1 train_rnn_model - RNN模型训练对话示例

**用户:** 我有一组温度数据，想训练一个RNN模型。

**AI:** 好的，我们可以使用`train_rnn_model`工具来训练RNN回归模型。你需要提供时间序列数据，比如说: [20.5, 21.2, 19.8, 22.0, 21.5, 20.9, 22.1, 21.8, 20.4, 23.0]，以及设定训练轮数等参数。是否需要调整默认参数？

**AI:** 调用工具训练：
```python
result = await client.train_rnn_model(
    data=[20.5, 21.2, 19.8, 22.0, 21.5, 20.9, 22.1, 21.8, 20.4, 23.0],
    epochs=50,
    sequence_length=3,
    hidden_size=32
)
```

**AI:** 训练完成！
```json
{
  "status": "success",
  "model_type": "rnn",
  "final_loss": 0.0123,
  "metrics": {
    "mse": 0.011,
    "rmse": 0.105,
    "mae": 0.082
  },
  "sample_predictions": [21.1, 19.9, 22.2],
  "sample_actuals": [21.2, 19.8, 22.0]
}
```

**AI:** RNN模型训练成功！最终损失为0.0123，MSE为0.011。可以看到预测值与实际值很接近。

---

### 3.2 train_lstm_model - LSTM模型训练对话示例

**用户:** 我有一段时间序列数据，序列比较长，有50个点，想用LSTM模型做预测。

**AI:** 好的，LSTM适合处理长序列数据，记忆能力更强。我们可以使用`train_lstm_model`工具。我看到你有50个数据点，那么我们可以设置序列长度为5，隐藏层大小为64，训练轮数为100。数据准备好了吗？

**AI:** 开始训练LSTM模型：
```python
result = await client.train_lstm_model(
    data=[2.1, 2.3, 2.0, 2.5, 2.2, 2.8, 1.9, 3.0, 2.7, 2.4, 2.9, 2.1, 3.2, 2.6, 2.8, 3.1, 2.5, 3.3, 2.7, 3.0, 2.8, 2.9, 3.4, 2.6, 3.5, 2.9, 3.1, 3.2, 2.8, 3.6, 3.0, 3.3, 3.1, 2.9, 3.7, 3.2, 3.4, 3.1, 3.0, 3.8, 3.3, 3.5, 3.2, 3.1, 3.9, 3.4, 3.6, 3.2, 3.3, 4.0],
    epochs=100,
    sequence_length=5,
    hidden_size=64
)
```

**AI:** LSTM训练结果：
```json
{
  "status": "success",
  "model_type": "lstm",
  "final_loss": 0.0089,
  "metrics": {
    "mse": 0.007,
    "rmse": 0.084,
    "mae": 0.061
  },
  "sample_predictions": [3.35, 3.5, 3.2],
  "sample_actuals": [3.4, 3.5, 3.2]
}
```

**AI:** LSTM模型训练成功！相比RNN，LSTM在处理长序列时表现更稳定，最终损失为0.0089，MSE为0.007。看起来预测准确度很高。

---

### 3.3 train_gru_model - GRU模型训练对话示例

**用户:** 我设备性能有限，想用最少的参数做时间序列预测，GRU是不是适合我？

**AI:** 确实，GRU 参数量比LSTM少，适合资源受限的环境。我们可以使用`train_gru_model`工具。给你看一个简单的例子，只需要很少的数据点和小的隐藏层。

**AI:** 训练GRU模型：
```python
result = await client.train_gru_model(
    data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    epochs=30,
    sequence_length=2,
    hidden_size=16
)
```

**AI:** GRU训练结果：
```json
{
  "status": "success",
  "model_type": "gru",
  "final_loss": 0.0156,
  "metrics": {
    "mse": 0.014,
    "rmse": 0.118,
    "mae": 0.091
  },
  "sample_predictions": [9.8, 10.1, 10.5],
  "sample_actuals": [10.0, 10.0, 11.0]
}
```

**AI:** 太好了！GRU模型训练成功。虽然参数量少，但表现还不错，最终损失为0.0156，MSE为0.014。对于资源有限的设备来说，这是一个很好的选择。

---

### 3.4 predict_model - 使用模型进行预测对话示例

**用户:** 我已经训练了一个LSTM模型，现在想对未来数据做预测，怎么做？

**AI:** 好，我们可以使用`predict_model`工具来使用已训练的模型进行预测。首先你需要确保模型已经训练好了，然后提供待预测的数据序列。

**AI:** 首先检查可用模型：
```python
await client.get_available_models()
```

**AI:** 如果LSTM已训练，就可以进行预测：
```python
result = await client.predict_model(
    data=[10.0, 11.0, 12.5],
    model_type="lstm"
)
```

**AI:** 预测结果：
```json
{
  "status": "success",
  "model_type": "lstm",
  "predictions": [13.1, 14.2, 15.3, 16.5, 17.8],
  "prediction_count": 5
}
```

**AI:** 成功预测了未来5个数据点！根据输入序列[10.0, 11.0, 12.5]，模型预测下一阶段的值为[13.1, 14.2, 15.3, 16.5, 17.8]。

**用户:** 我想预测其他模型，可是我还没训练过。

**AI:** 这样的话会返回错误信息：
```json
{
  "status": "error",
  "message": "Model 'rnn' not trained yet. Please train the model first."
}
```

**AI:** 看来需要先训练相应的模型，然后再进行预测。

---

### 3.5 compare_models - 模型对比和自动选择对话示例

**用户:** 我很难决定用哪个模型，RNN、LSTM还是GRU？有没有工具可以帮我比较它们？

**AI:** 当然有！我们可以使用`compare_models`工具来同时训练三种模型，并自动选择表现最好的一个。这个工具会比较MSE（均方误差）来确定哪个模型最合适。

**AI:** 调用比较工具：
```python
result = await client.compare_models(
    data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    epochs=100,
    sequence_length=3,
    hidden_size=64
)
```

**AI:** 模型比较结果：
```json
{
  "status": "success",
  "results": {
    "rnn": {
      "mse": 0.031,
      "rmse": 0.176,
      "mae": 0.145
    },
    "lstm": {
      "mse": 0.012,
      "rmse": 0.109,
      "mae": 0.085
    },
    "gru": {
      "mse": 0.022,
      "rmse": 0.148,
      "mae": 0.112
    }
  },
  "best_model": "lstm",
  "best_mse": 0.012
}
```

**AI:** 太棒了！经过比较，LSTM模型表现最佳，MSE只有0.012，比RNN的0.031和GRU的0.022都要好。系统自动推荐使用LSTM模型进行后续预测，这对长序列数据来说是最合适的。

---

### 3.6 get_available_models - 检查可用模型对话示例

**用户:** 我想知道当前已经训练了哪些模型。

**AI:** 没问题，我们可以使用`get_available_models`工具来查看当前已训练并可用的模型列表。这个工具不需要任何参数。

**AI:** 检查可用模型：
```python
result = await client.get_available_models()
```

**AI:** 查询结果：
```json
{
  "status": "success",
  "available_models": ["lstm", "gru"],
  "model_count": 2
}
```

**AI:** 好的，当前已经训练了两个模型：LSTM和GRU。现在你可以直接使用这两个模型进行预测，而不需要重新训练。

---

### 3.7 get_server_config - 获取服务器配置对话示例

**用户:** 我想了解服务器的当前配置。

**AI:** 我们可以用`get_server_config`工具来获取服务器的配置信息。你可以指定具体要查询的配置键，如果不指定会返回默认值。

**AI:** 获取服务器配置：
```python
result = await client.get_server_config(key="server.name")
```

**AI:** 配置查询结果：
```json
{
  "status": "success",
  "config_key": "server.name",
  "config_value": "TimeSeriesMCPServer",
  "description": "当前服务器名称"
}
```

**AI:** 如果你想查看所有配置，也可以不传key参数来获取完整的配置信息。


### 3.8 predict_prompt - 时间序列预测工作流对话示例

**用户:** 预测时间序列[1.02, 2.11, 2.98, 4.05, 5.12, 6.03, 7.19, 8.25, 9.17, 10.28,
 11.33, 12.41, 13.29, 14.45, 15.52, 16.37, 17.58, 18.64, 19.71, 20.83,
 21.92, 22.78, 23.89, 25.01, 26.10, 27.05, 28.18, 29.27, 30.34, 31.42,
 32.51, 33.63, 34.72, 35.81, 36.90, 37.95, 39.08, 40.13, 41.27, 42.39,
 43.48, 44.55, 45.69, 46.77, 47.84, 48.91, 50.02, 51.15, 52.23, 53.31]的后五项

**AI:** 根据对时间序列数据的分析和模型训练，以下是预测结果和模型对比分析：

预测后五项

基于训练的LSTM模型，时间序列的后五项预测值为：
- 第51项：**53.31** → 预测为 **54.40**
- 第52项：**54.40** → 预测为 **55.49**
- 第53项：**55.49** → 预测为 **56.58**
- 第54项：**56.58** → 预测为 **57.67**
- 第55项：**57.67** → 预测为 **58.76**

（注：以上为基于模型趋势的连续预测，根据最后几个点的模式推算得出。）

模型性能对比

我们对比了RNN、LSTM和GRU三种模型在相同数据集上的表现，评估指标包括均方误差（MSE）、均方根误差（RMSE）和平均绝对误差（MAE）：

| 模型 | MSE | RMSE | MAE | 最终损失 |
|------|-----|------|-----|-----------|
| RNN  | 0.0671 | 0.2591 | 0.2192 | 0.0000276 |
| LSTM | **0.0236** | **0.1537** | **0.1346** | **0.0000095** |
| GRU  | 0.0249 | 0.1576 | 0.1375 | 0.0000083 |

结论与推荐

- **最佳模型**：**LSTM**，在所有指标中表现最优，尤其是在MSE和RMSE上显著优于RNN和GRU。
- **次优模型**：GRU，性能与LSTM非常接近，但略逊于LSTM。
- **最差模型**：RNN，误差明显高于其他两种模型，不适合此时间序列任务。

推荐

建议使用**LSTM模型**进行后续预测，其预测精度最高，对趋势的捕捉能力最强。

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
