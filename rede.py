import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Ramo Curto (K=3): alta frequência (ruído e impactos rápidos)
        self.ramo_curto = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2)
        )
        # Ramo Longo Dilatado (K=7, d=3): baixa frequência (movimento sustentado)
        self.ramo_longo = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=7, padding=9, dilation=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2)
        )
        # Atenção de Canal (Squeeze-and-Excitation)
        self.atencao_canal = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(32, 8), nn.ReLU(),
            nn.Linear(8, 32), nn.Sigmoid()
        )
        # Convolução Final + Atenção Temporal
        self.conv_final = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU()
        )
        self.atencao_tempo = nn.Conv1d(64, 1, kernel_size=1)
        self.classificador = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.cat([self.ramo_curto(x), self.ramo_longo(x)], dim=1)
        x = x * self.atencao_canal(x).unsqueeze(-1)
        x = self.conv_final(x)
        pesos_tempo = F.softmax(self.atencao_tempo(x), dim=-1)
        x = torch.sum(x * pesos_tempo, dim=-1)
        return self.classificador(x)