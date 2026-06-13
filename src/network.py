from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float32)


# ==========================================
# オプティマイザの実装
# ==========================================
class SGD:
    """確率的勾配降下法（Stochastic Gradient Descent）"""
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def update(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """パラメータと勾配の辞書を受け取り、パラメータを更新する"""
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class Adam:
    """Adam（PyTorch完全互換バージョン + Weight Decay対応）"""
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.0) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.iter = 0
        self.m: dict[str, np.ndarray] | None = None
        self.v: dict[str, np.ndarray] | None = None

    def update(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1

        for key in params.keys():
            # バイアス(b)にはペナルティをかけず、重み(W)にだけペナルティを勾配に足す
            if key.startswith("W"):
                grad = grads[key] + self.weight_decay * params[key]
            else:
                grad = grads[key]

            # 勢い(m)と勾配の2乗(v)を更新
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

            # バイアス補正
            m_hat = self.m[key] / (1 - self.beta1 ** self.iter)
            v_hat = self.v[key] / (1 - self.beta2 ** self.iter)

            # パラメータの更新
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


# ==========================================
# ネットワーク設定とモデルの実装
# ==========================================
@dataclass
class NetworkConfig:
    input_size: int = 784
    hidden_size: int = 1024
    hidden_size2: int = 512
    output_size: int = 10
    learning_rate: float = 0.001
    batch_size: int = 128
    seed: int = 42
    optimizer: str = "Adam"        # "SGD", "Momentum", "Adam" から選択
    weight_decay: float = 0.0001   # Weight Decayの強さ（追加）


class SimpleMLP:
    def __init__(self, config: NetworkConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        
        # Heの初期値（ReLUと相性が良いスケーリング）
        scale1 = np.sqrt(2.0 / config.input_size)
        scale2 = np.sqrt(2.0 / config.hidden_size)
        scale3 = np.sqrt(2.0 / config.hidden_size2)

        self.params: dict[str, np.ndarray] = {
            "W1": (rng.standard_normal((config.input_size, config.hidden_size)) * scale1).astype(np.float32),
            "b1": np.zeros(config.hidden_size, dtype=np.float32),
            "W2": (rng.standard_normal((config.hidden_size, config.hidden_size2)) * scale2).astype(np.float32),
            "b2": np.zeros(config.hidden_size2, dtype=np.float32),
            "W3": (rng.standard_normal((config.hidden_size2, config.output_size)) * scale3).astype(np.float32),
            "b3": np.zeros(config.output_size, dtype=np.float32),
        }

        # Optimizerの初期化（Configから設定を読み込む）
        optimizer_name = config.optimizer.lower()
        if optimizer_name == "adam":
            self.optimizer = Adam(lr=config.learning_rate, weight_decay=config.weight_decay)
        elif optimizer_name == "sgd":
            self.optimizer = SGD(lr=config.learning_rate)
        elif optimizer_name == "momentum":
            self.optimizer = Momentum(lr=config.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z1 = _relu(np.dot(x, self.params["W1"]) + self.params["b1"])
        z2 = _relu(np.dot(z1, self.params["W2"]) + self.params["b2"])
        logits = np.dot(z2, self.params["W3"]) + self.params["b3"]
        return _softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def evaluate_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        correct = 0
        total = x.shape[0]
        batch_size = self.config.batch_size
        for i in range(0, total, batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            pred = self.predict(x_batch)
            correct += int(np.sum(pred == y_batch))
        return float(correct) / float(total)

    def train_epoch(self, x: np.ndarray, y: np.ndarray, epoch: int) -> float:
        rng = np.random.default_rng(self.config.seed + epoch)
        indices = rng.permutation(x.shape[0])
        total_loss = 0.0
        steps = 0
        batch_size = self.config.batch_size

        for start in range(0, x.shape[0], batch_size):
            batch_idx = indices[start : start + batch_size]
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]

            # 順伝播 (Forward)
            z1_linear = np.dot(x_batch, self.params["W1"]) + self.params["b1"]
            z1 = _relu(z1_linear)
            z2_linear = np.dot(z1, self.params["W2"]) + self.params["b2"]
            z2 = _relu(z2_linear)
            logits = np.dot(z2, self.params["W3"]) + self.params["b3"]
            probs = _softmax(logits)

            # 損失計算 (Loss)
            y_one_hot = _one_hot(y_batch, self.config.output_size)
            loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-8), axis=1))
            total_loss += float(loss)
            steps += 1

            # 逆伝播 (Backward)
            d_logits = (probs - y_one_hot) / x_batch.shape[0]
            dW3 = np.dot(z2.T, d_logits)
            db3 = np.sum(d_logits, axis=0)

            d_z2 = np.dot(d_logits, self.params["W3"].T)
            d_z2_linear = d_z2 * _relu_grad(z2_linear)
            dW2 = np.dot(z1.T, d_z2_linear)
            db2 = np.sum(d_z2_linear, axis=0)

            d_z1 = np.dot(d_z2_linear, self.params["W2"].T)
            d_z1_linear = d_z1 * _relu_grad(z1_linear)
            dW1 = np.dot(x_batch.T, d_z1_linear)
            db1 = np.sum(d_z1_linear, axis=0)

            # 勾配を辞書にまとめる
            grads = {
                "W1": dW1.astype(np.float32),
                "b1": db1.astype(np.float32),
                "W2": dW2.astype(np.float32),
                "b2": db2.astype(np.float32),
                "W3": dW3.astype(np.float32),
                "b3": db3.astype(np.float32),
            }
            
            # オプティマイザによるパラメータ更新
            self.optimizer.update(self.params, grads)

        return total_loss / max(steps, 1)

    def to_state(self) -> dict[str, object]:
        """モデルの状態（設定と重み）を保存用の辞書にする"""
        return {
            "model_type": "SimpleMLP",
            "config": {
                "input_size": self.config.input_size,
                "hidden_size": self.config.hidden_size,
                "hidden_size2": self.config.hidden_size2,
                "output_size": self.config.output_size,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "seed": self.config.seed,
                "optimizer": self.config.optimizer,          # 追加：オプティマイザの種類
                "weight_decay": self.config.weight_decay,    # 追加：Weight Decayの強さ
            },
            "params": self.params,
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "SimpleMLP":
        """保存された辞書からモデルを復元する"""
        config_obj = state.get("config")
        if not isinstance(config_obj, dict):
            raise ValueError("Invalid state: 'config' must be a dict")
        config_dict: dict[str, Any] = config_obj

        config = NetworkConfig(
            input_size=int(config_dict["input_size"]),
            hidden_size=int(config_dict["hidden_size"]),
            hidden_size2=int(config_dict.get("hidden_size2", 512)),
            output_size=int(config_dict["output_size"]),
            learning_rate=float(config_dict.get("learning_rate", 0.001)),
            batch_size=int(config_dict.get("batch_size", 128)),
            seed=int(config_dict.get("seed", 42)),
            optimizer=str(config_dict.get("optimizer", "Adam")),               # 復元時に追加
            weight_decay=float(config_dict.get("weight_decay", 0.0001)),       # 復元時に追加
        )

        params_obj = state.get("params")
        if not isinstance(params_obj, dict):
            raise ValueError("Invalid state: 'params' must be a dict")
        params: dict[str, np.ndarray] = {}
        for key, value in params_obj.items():
            if not isinstance(key, str) or not isinstance(value, np.ndarray):
                raise ValueError("Invalid state: params must be dict[str, np.ndarray]")
            params[key] = value

        model = cls(config)
        model.params = params
        return model