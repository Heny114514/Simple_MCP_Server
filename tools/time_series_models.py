import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrainingResult:
    model_type: str
    final_loss: float
    predictions: List[float]
    actuals: List[float]
    metrics: Dict[str, float]


class RNNRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super(RNNRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class GRURegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super(GRURegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


class TimeSeriesModel:
    def __init__(
        self,
        model_type: str = "lstm",
        sequence_length: int = 10,
        hidden_size: int = 64,
        num_layers: int = 1,
        learning_rate: float = 0.01
    ):
        self.model_type = model_type.lower()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def _normalize(self, data: List[float]) -> np.ndarray:
        data = np.array(data)
        self.scaler = {"min": data.min(), "max": data.max()}
        return (data - data.min()) / (data.max() - data.min())

    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return data
        return data * (self.scaler["max"] - self.scaler["min"]) + self.scaler["min"]

    def train(self, data: List[float], epochs: int = 100, verbose: bool = True) -> TrainingResult:
        normalized_data = self._normalize(data)
        X, y = self._create_sequences(normalized_data)

        X = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
        y = torch.FloatTensor(y).unsqueeze(-1).to(self.device)

        input_size = 1
        output_size = 1

        if self.model_type == "rnn":
            self.model = RNNRegressor(input_size, self.hidden_size, output_size, self.num_layers)
        elif self.model_type == "lstm":
            self.model = LSTMRegressor(input_size, self.hidden_size, output_size, self.num_layers)
        elif self.model_type == "gru":
            self.model = GRURegressor(input_size, self.hidden_size, output_size, self.num_layers)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model = self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        losses = []
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy().flatten()
            actuals = y.cpu().numpy().flatten()

        predictions = self._denormalize(predictions).tolist()
        actuals = self._denormalize(actuals).tolist()

        mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))

        return TrainingResult(
            model_type=self.model_type,
            final_loss=losses[-1],
            predictions=predictions,
            actuals=actuals,
            metrics={
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "final_normalized_loss": float(losses[-1])
            }
        )

    def predict(self, data: List[float]) -> List[float]:
        if self.model is None:
            raise ValueError("Model not trained yet")

        normalized_data = self._normalize(data)
        
        if len(normalized_data) < self.sequence_length:
            padding = np.zeros(self.sequence_length - len(normalized_data))
            input_data = np.concatenate([padding, normalized_data])
            X = torch.FloatTensor([input_data]).unsqueeze(-1).to(self.device)
        else:
            X, _ = self._create_sequences(normalized_data)
            X = torch.FloatTensor(X).unsqueeze(-1).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy().flatten()

        return self._denormalize(predictions).tolist()


def create_model(
    model_type: str,
    sequence_length: int = 10,
    hidden_size: int = 64,
    num_layers: int = 1,
    learning_rate: float = 0.01
) -> TimeSeriesModel:
    return TimeSeriesModel(
        model_type=model_type,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=learning_rate
    )
