from typing import Any, Dict, List, Optional

from prompts import TimeSeriesMCPServer_Prompt


@TimeSeriesMCPServer_Prompt(
	name="predict_workflow",
	title="Predict Workflow Prompt",
	description="执行序列预测的工作流：训练三种模型，选择最佳模型进行预测",
)
async def predict_prompt(sequence: Optional[List[float]] = None, num_predictions: Optional[int] = None) -> Dict[str, Any]:
	"""只要用户有序列预测相关的需求，必须调用并严格遵守此工作流，如果参数不足则要求用户给出缺失参数

	此提示描述了执行时间序列预测的完整工作流程

	Args:
		sequence (Optional[List[float]]): 历史时间序列数据列表，如果未提供则会要求用户输入
		num_predictions (Optional[int]): 需要预测的数据点数量，如果未提供则会要求用户输入

	Returns:
		Dict[str, Any]: 工作流执行结果，包含预测数据和使用的模型信息
	"""

	if sequence is None or num_predictions is None:
		missing_params = []
		if sequence is None:
			missing_params.append("历史时间序列数据 (sequence: List[float])")
		if num_predictions is None:
			missing_params.append("预测点数 (num_predictions: int)")

		return {
			"error": "参数不足",
			"missing_parameters": missing_params,
			"instruction": "请提供以下缺失参数以执行时间序列预测任务：" + ", ".join(missing_params),
			"workflow_description": "此工作流将训练三种模型（RNN、LSTM、GRU），比较性能，选择最佳模型进行预测。",
		}

	workflow_steps = f"""
请严格按照以下工作流程执行时间序列预测任务：

**输入数据：**
- 历史序列: {sequence}
- 预测点数: {num_predictions}

**工作流程步骤：**

1. **训练三种模型：**
   - 使用 train_rnn_model 工具训练RNN模型
   - 使用 train_lstm_model 工具训练LSTM模型
   - 使用 train_gru_model 工具训练GRU模型
   
   调用这些工具时使用相同的历史数据：{sequence}

2. **比较模型性能：**
   - 使用 compare_models 工具对比三种模型的预测精度
   - 工具将返回基于MSE的最优模型选择

3. **执行预测：**
   - 使用选择的最佳模型调用 predict_model 工具
   - 传递历史序列数据进行预测
   - 要求预测 {num_predictions} 个数据点

4. **返回结果：**
   - 预测的数值序列
   - 使用的模型类型
   - 模型的性能指标（如RMSE、MAE等）

请严格按照上述步骤执行，确保每个工具调用都使用正确的数据参数。
最终结果应包含预测数据和模型选择理由。
"""
	return {
		"workflow_prompt": workflow_steps,
		"input_sequence": sequence,
		"num_predictions": num_predictions,
	}
