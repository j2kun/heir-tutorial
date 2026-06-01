import torch
import torch_mlir
from torch_mlir.fx import OutputType

import model

checkpoint = torch.load("model.pth", map_location="cpu")
model = model.MLP()

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

sample_input = torch.randn(1, 28 * 28)
mlir = torch_mlir.fx.export_and_import(
    model, sample_input, output_type=OutputType.LINALG_ON_TENSORS
)

print(mlir)
