"""
model.py
--------
CNN architecture matching Table IV in the paper exactly.

Input  : (batch, 1, 5, 150)  float32   — one channel, 5 range rows x 150 Doppler bins
Output : (batch, Nc)          float32   — raw logits (softmax applied at inference)

Architecture (Table IV):
  Conv  F=10  K=5×8  stride=(1,1)  → (batch,10,1,143)  ReLU
  Conv  F=10  K=1×8  stride=(1,2)  → (batch,10,1, 68)  ReLU
  Conv  F=10  K=1×8  stride=(1,3)  → (batch,10,1, 21)  ReLU
  Flatten                           → (batch, 210)
  FC    210→32                      ReLU
  FC     32→Nc                      (logits)

Total parameters: 8782 + Nc×33  (matches paper)
"""

import torch
import torch.nn as nn


class FMCWClassifier(nn.Module):
    def __init__(self, n_classes: int = 9):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: kernel (5,8), stride (1,1)
            nn.Conv2d(1, 10, kernel_size=(5, 8), stride=(1, 1)),
            nn.ReLU(),
            # Conv2: kernel (1,8), stride (1,2)
            nn.Conv2d(10, 10, kernel_size=(1, 8), stride=(1, 2)),
            nn.ReLU(),
            # Conv3: kernel (1,8), stride (1,3)
            nn.Conv2d(10, 10, kernel_size=(1, 8), stride=(1, 3)),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),           # 10 * 1 * 21 = 210
            nn.Linear(210, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

        # He (Kaiming) initialisation — matches paper
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, 1, 5, 150)
        returns logits : (batch, n_classes)
        """
        x = self.features(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities — use at inference time."""
        return torch.softmax(self(x), dim=-1)


# ── quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = FMCWClassifier(n_classes=9)
    print(model)

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters : {total}")
    print(f"Expected (paper) : {8782 + 9*33} = {8782 + 9*33}")

    dummy = torch.randn(4, 1, 5, 150)
    out   = model(dummy)
    print(f"\nInput  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
