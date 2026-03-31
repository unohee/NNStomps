# Created: 2026-03-25
# Purpose: 조건부 GRU 디스토션 모델 — RTNeural 호환
# Dependencies: torch
# Test Status: 미완료

import torch
import torch.nn as nn


class NNStompGRU(nn.Module):
    """조건부 GRU 새추레이션/디스토션 모델

    RTNeural 호환 아키텍처:
        Input(1 + cond_dim) → GRU(hidden) → Dense(1) → Tanh

    조건 벡터(cond)로 플러그인 세팅을 제어합니다.
    Sample-by-sample 추론이 가능하며, 학습은 시퀀스 단위로 수행합니다.

    Args:
        cond_dim: 조건 벡터 차원 (플러그인 파라미터 수)
        hidden_size: GRU hidden state 크기 (RTNeural 실시간 처리: 20~80 권장)
    """

    def __init__(self, cond_dim: int, hidden_size: int = 40):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=1 + cond_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dense = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """순전파

        Args:
            x: (batch, seq_len, 1) 오디오 시퀀스
            cond: (batch, cond_dim) 조건 벡터 (정규화된 파라미터 [0,1])
            hidden: (1, batch, hidden_size) 이전 hidden state (TBPTT용)

        Returns: (output, hidden)
            output: (batch, seq_len, 1)
            hidden: (1, batch, hidden_size) 마지막 hidden state
        """
        batch, seq_len, _ = x.shape

        # 조건 벡터를 매 타임스텝에 반복 → (batch, seq_len, cond_dim)
        cond_expanded = cond.unsqueeze(1).expand(-1, seq_len, -1)

        # 입력: [오디오 샘플, 조건 벡터] → (batch, seq_len, 1 + cond_dim)
        inp = torch.cat([x, cond_expanded], dim=-1)

        h, hidden_out = self.gru(inp, hidden)
        out = self.tanh(self.dense(h))

        return out, hidden_out

    def process_sample(
        self,
        x: float,
        cond: list[float],
        hidden: torch.Tensor | None = None,
    ) -> tuple[float, torch.Tensor]:
        """단일 샘플 추론 (RTNeural과 동일한 인터페이스)

        Args:
            x: 입력 오디오 샘플
            cond: 조건 벡터
            hidden: GRU hidden state

        Returns: (출력 샘플, 새 hidden state)
        """
        x_t = torch.tensor([[[x]]], dtype=torch.float32)
        cond_t = torch.tensor([cond], dtype=torch.float32)

        with torch.no_grad():
            out, hidden_out = self.forward(x_t, cond_t, hidden)

        return float(out[0, 0, 0]), hidden_out


class NNStompGRU2(nn.Module):
    """2단 GRU — 복잡한 비선형 특성용 (고품질)

    Input(1 + cond_dim) → GRU(hidden1) → GRU(hidden2) → Dense(1) → Tanh

    RTNeural 호환. CPU 부담이 NNStompGRU보다 높음.
    """

    def __init__(self, cond_dim: int, hidden1: int = 40, hidden2: int = 20):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        self.gru1 = nn.GRU(
            input_size=1 + cond_dim,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
        )
        self.gru2 = nn.GRU(
            input_size=hidden1,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
        )
        self.dense = nn.Linear(hidden2, 1)
        self.tanh = nn.Tanh()

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        hidden: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None),
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """순전파

        Args:
            x: (batch, seq_len, 1)
            cond: (batch, cond_dim)
            hidden: (h1, h2) 각 GRU의 hidden state

        Returns: (output, (h1, h2))
        """
        batch, seq_len, _ = x.shape
        h1_in, h2_in = hidden

        cond_expanded = cond.unsqueeze(1).expand(-1, seq_len, -1)
        inp = torch.cat([x, cond_expanded], dim=-1)

        out1, h1_out = self.gru1(inp, h1_in)
        out2, h2_out = self.gru2(out1, h2_in)
        out = self.tanh(self.dense(out2))

        return out, (h1_out, h2_out)
