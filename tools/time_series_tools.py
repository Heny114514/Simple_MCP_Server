from typing import Any, Dict, List, Optional
import json

from tools import YA_MCPServer_Tool
from tools.time_series_models import TimeSeriesModel, create_model


MODEL_REGISTRY: Dict[str, TimeSeriesModel] = {}


@YA_MCPServer_Tool(
    name="train_rnn_model",
    title="Train RNN Model",
    description="训练RNN回归模型，用于时间序列预测",
)
async def train_rnn_model(
    data: List[float],
    epochs: int = 100,
    sequence_length: int = 10,
    hidden_size: int = 64,
    num_layers: int = 1,
    learning_rate: float = 0.01,
) -> Dict[str, Any]:
    """训练RNN回归模型

    Args:
        data: 时间序列数据列表
        epochs: 训练轮数，默认100
        sequence_length: 序列长度，默认10
        hidden_size: 隐藏层大小，默认64
        num_layers: RNN层数，默认1
        learning_rate: 学习率，默认0.01

    Returns:
        包含训练结果和评估指标的字典
    """
    model = create_model(
        model_type="rnn",
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=learning_rate
    )
    result = model.train(data, epochs=epochs, verbose=False)
    MODEL_REGISTRY["rnn"] = model

    return {
        "status": "success",
        "model_type": result.model_type,
        "final_loss": result.final_loss,
        "metrics": result.metrics,
        "sample_predictions": result.predictions[:5],
        "sample_actuals": result.actuals[:5],
    }


@YA_MCPServer_Tool(
    name="train_lstm_model",
    title="Train LSTM Model",
    description="训练LSTM回归模型，用于时间序列预测",
)
async def train_lstm_model(
    data: List[float],
    epochs: int = 100,
    sequence_length: int = 10,
    hidden_size: int = 64,
    num_layers: int = 1,
    learning_rate: float = 0.01,
) -> Dict[str, Any]:
    """训练LSTM回归模型

    Args:
        data: 时间序列数据列表
        epochs: 训练轮数，默认100
        sequence_length: 序列长度，默认10
        hidden_size: 隐藏层大小，默认64
        num_layers: LSTM层数，默认1
        learning_rate: 学习率，默认0.01

    Returns:
        包含训练结果和评估指标的字典
    """
    model = create_model(
        model_type="lstm",
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=learning_rate
    )
    result = model.train(data, epochs=epochs, verbose=False)
    MODEL_REGISTRY["lstm"] = model

    return {
        "status": "success",
        "model_type": result.model_type,
        "final_loss": result.final_loss,
        "metrics": result.metrics,
        "sample_predictions": result.predictions[:5],
        "sample_actuals": result.actuals[:5],
    }


@YA_MCPServer_Tool(
    name="train_gru_model",
    title="Train GRU Model",
    description="训练GRU回归模型，用于时间序列预测",
)
async def train_gru_model(
    data: List[float],
    epochs: int = 100,
    sequence_length: int = 10,
    hidden_size: int = 64,
    num_layers: int = 1,
    learning_rate: float = 0.01,
) -> Dict[str, Any]:
    """训练GRU回归模型

    Args:
        data: 时间序列数据列表
        epochs: 训练轮数，默认100
        sequence_length: 序列长度，默认10
        hidden_size: 隐藏层大小，默认64
        num_layers: GRU层数，默认1
        learning_rate: 学习率，默认0.01

    Returns:
        包含训练结果和评估指标的字典
    """
    model = create_model(
        model_type="gru",
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=learning_rate
    )
    result = model.train(data, epochs=epochs, verbose=False)
    MODEL_REGISTRY["gru"] = model

    return {
        "status": "success",
        "model_type": result.model_type,
        "final_loss": result.final_loss,
        "metrics": result.metrics,
        "sample_predictions": result.predictions[:5],
        "sample_actuals": result.actuals[:5],
    }


@YA_MCPServer_Tool(
    name="predict_model",
    title="Predict with Model",
    description="使用已训练的时间序列模型进行预测",
)
async def predict_model(
    data: List[float],
    model_type: str = "lstm",
) -> Dict[str, Any]:
    """使用已训练的模型进行预测

    Args:
        data: 时间序列数据列表
        model_type: 模型类型 (rnn/lstm/gru)

    Returns:
        预测结果
    """
    if model_type not in MODEL_REGISTRY:
        return {
            "status": "error",
            "message": f"Model '{model_type}' not trained yet. Please train the model first."
        }

    model = MODEL_REGISTRY[model_type]
    predictions = model.predict(data)

    return {
        "status": "success",
        "model_type": model_type,
        "predictions": predictions,
        "prediction_count": len(predictions),
    }


@YA_MCPServer_Tool(
    name="compare_models",
    title="Compare Time Series Models",
    description="对比RNN、LSTM、GRU三种模型的预测效果",
)
async def compare_models(
    data: List[float],
    epochs: int = 100,
    sequence_length: int = 10,
    hidden_size: int = 64,
) -> Dict[str, Any]:
    """对比RNN、LSTM、GRU三种模型的预测效果

    Args:
        data: 时间序列数据列表
        epochs: 训练轮数
        sequence_length: 序列长度
        hidden_size: 隐藏层大小

    Returns:
        各模型的评估指标对比
    """
    results = {}

    for model_type in ["rnn", "lstm", "gru"]:
        model = create_model(
            model_type=model_type,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
        )
        result = model.train(data, epochs=epochs, verbose=False)
        results[model_type] = {
            "mse": result.metrics["mse"],
            "rmse": result.metrics["rmse"],
            "mae": result.metrics["mae"],
            "final_loss": result.final_loss,
        }

    best_model = min(results.keys(), key=lambda x: results[x]["mse"])

    return {
        "status": "success",
        "results": results,
        "best_model": best_model,
        "best_mse": results[best_model]["mse"],
    }


@YA_MCPServer_Tool(
    name="get_available_models",
    title="Get Available Models",
    description="获取当前已训练可用的模型列表",
)
async def get_available_models() -> Dict[str, Any]:
    """获取当前已训练可用的模型列表"""
    return {
        "status": "success",
        "available_models": list(MODEL_REGISTRY.keys()),
        "model_count": len(MODEL_REGISTRY),
    }
