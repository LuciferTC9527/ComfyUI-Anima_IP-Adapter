# Anima IP-Adapter for ComfyUI

IP-Adapter for [Anima](https://huggingface.co/circlestone-labs/Anima), supporting reference-image-driven style transfer in ComfyUI.

## Installation

Clone into `ComfyUI/custom_nodes/`:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/LuciferTC9527/ComfyUI-Anima_IP-Adapter.git
```

## Required Models

| Model | Location | Notes |
|-------|----------|-------|
| SigLIP2 Encoder | `ComfyUI/models/siglip2/siglip2-base-patch16-512/` | Enable `auto_download` in loader, or download manually (see below) |
| IP-Adapter Weights | `ComfyUI/models/ipadapter/` | Download `ip_adapter.safetensors` from [LuciferTC/Anima-IP-Adapter](https://huggingface.co/LuciferTC/Anima-IP-Adapter) |

## Usage

1. Load Anima model in ComfyUI
2. Add **Anima IP-Adapter Loader** → select IP-Adapter file, toggle `auto_download` for SigLIP2
3. Add **Anima IP-Adapter Apply** → connect model, ip_adapter, and reference image
4. Connect to sampler

### Manual SigLIP2 Installation

```bash
cd ComfyUI/models/siglip2/
git clone https://huggingface.co/google/siglip2-base-patch16-512
```

## License

Code: **Apache 2.0**

Weights: [CircleStone Labs Non-Commercial License v1.0](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md)
