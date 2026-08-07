import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PLUGIN_DIR = Path(__file__).resolve().parents[1]
COMFYUI_DIR = PLUGIN_DIR.parents[1]
sys.path.insert(0, str(COMFYUI_DIR))

spec = importlib.util.spec_from_file_location("anima_ip_adapter_nodes", PLUGIN_DIR / "nodes.py")
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


class ManualCastLinear(nn.Linear):
    """Minimal stand-in for ComfyUI's runtime weight casting."""

    def forward(self, x):
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        bias = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


class LoRALinearTest(unittest.TestCase):
    def test_uses_activation_dtype_for_mixed_precision_lora(self):
        base = ManualCastLinear(4, 3, dtype=torch.bfloat16)
        lora_A = torch.randn(2, 4, dtype=torch.bfloat16)
        lora_B = torch.randn(3, 2, dtype=torch.bfloat16)
        layer = nodes._LoRALinear(base, lora_A, lora_B, scale=0.5)
        x = torch.randn(2, 4, dtype=torch.float16)

        output = layer(x)

        expected = base(x) + (x @ lora_A.to(torch.float16).T @ lora_B.to(torch.float16).T) * 0.5
        self.assertEqual(output.dtype, torch.float16)
        torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    unittest.main()
