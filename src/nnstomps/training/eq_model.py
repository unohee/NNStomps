# Created: 2026-04-01
# Purpose: Neural EQ 모델 — MLP로 파라미터 → FIR 계수 예측

import torch
import torch.nn as nn


class NNStompEQ(nn.Module):
    """조건부 MLP EQ 모델 — 파라미터 벡터로 FIR 필터 계수 예측

    프로파일된 아날로그 EQ의 주파수 응답을 학습하여
    임의의 파라미터 조합에 대한 FIR 필터를 실시간 생성.

    Args:
        cond_dim: 입력 파라미터 차원 (EQ 노브 수)
        fir_len: 출력 FIR 필터 길이 (256 or 1024)
        hidden_dims: 은닉층 크기 리스트
    """

    def __init__(
        self,
        cond_dim: int,
        fir_len: int = 256,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.fir_len = fir_len

        if hidden_dims is None:
            hidden_dims = [128, 256, 256]

        layers = []
        in_dim = cond_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, fir_len))
        self.mlp = nn.Sequential(*layers)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """파라미터 벡터 → FIR 계수

        Args:
            params: (batch, cond_dim) 정규화된 EQ 파라미터

        Returns:
            fir: (batch, fir_len) FIR 필터 계수
        """
        return self.mlp(params)

    def predict_frequency_response(
        self, params: torch.Tensor, n_fft: int = 32768
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """파라미터 → 주파수 응답 (magnitude dB + phase)

        Args:
            params: (batch, cond_dim)
            n_fft: FFT 크기

        Returns:
            mag_db: (batch, n_fft//2+1) 크기 응답 (dB)
            phase: (batch, n_fft//2+1) 위상 응답 (radians)
        """
        fir = self.forward(params)
        # zero-pad to n_fft
        padded = torch.zeros(fir.shape[0], n_fft, device=fir.device)
        padded[:, :self.fir_len] = fir
        H = torch.fft.rfft(padded)
        mag_db = 20 * torch.log10(torch.abs(H) + 1e-10)
        phase = torch.angle(H)
        return mag_db, phase
