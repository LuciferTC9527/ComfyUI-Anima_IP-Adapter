"""
Anima IP-Adapter ComfyUI nodes (SigLIP2 version, self-contained).

Architecture:
  - SigLIP2 vision encoder (frozen) → [B, 1024, 768] tokens
  - Per-block ip_k_proj(768, inner_dim) / ip_v_proj(768, inner_dim) injected via transformer_options
  - Shared projection: shared MLP + per-block adaln_ip (detected automatically)

AnimaIPAdapterLoader — loads SigLIP2 encoder + IP-Adapter K/V weights.
AnimaIPAdapterApply  — encodes ref image → injects IP tokens via transformer_options.

External pip deps: torch, transformers, safetensors, Pillow, numpy, torchvision
"""

import math
import re
import inspect
import os
import folder_paths
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image as PILImage

# ═══════════════════════════════════════════════════════════════════════════════
# SigLIP2 constants
# ═══════════════════════════════════════════════════════════════════════════════

SIGLIP_HIDDEN_SIZE = 768
SIGLIP_NUM_TOKENS = 1024
SIGLIP_IMAGE_SIZE = 512


# ═══════════════════════════════════════════════════════════════════════════════
# Shared projection helpers (for ip_shared_projection = true training)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_shared_proj(dim, inner_dim, device, dtype):
    """Shared projection module: Linear → GELU → Linear."""
    class _Proj(nn.Module):
        def __init__(self):
            super().__init__()
            self.expand = nn.Linear(dim, inner_dim)
            self.act = nn.GELU()
            self.project = nn.Linear(inner_dim, inner_dim)
        def forward(self, x):
            x = self.expand(x)
            x = self.act(x)
            x = self.project(x)
            return x
    return _Proj().to(device=device, dtype=dtype)


def _make_shared_proj_ref(shared_module):
    """Callable reference to shared module, auto-moves to input device."""
    class _Ref:
        def __call__(self, x):
            shared_module.to(x.device)
            return shared_module(x)
    return _Ref()


class _LoRALinear(nn.Module):
    """Minimal LoRA wrapper for inference."""
    def __init__(self, base: nn.Linear, lora_A: torch.Tensor, lora_B: torch.Tensor, scale: float):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(lora_A.to(device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(lora_B.to(device=base.weight.device, dtype=base.weight.dtype))
        self.scale = scale

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


# ═══════════════════════════════════════════════════════════════════════════════
# SigLIP Token Compressor (learned query-based compression)
# Mirrors models/siglip_compressor.py for standalone ComfyUI use.
# ═══════════════════════════════════════════════════════════════════════════════

class _SigLIPCompressorLayer(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm_self = nn.LayerNorm(dim)
        self.self_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.self_out = nn.Linear(dim, dim, bias=False)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, q, kv):
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        Q = self._heads(self.q_proj(q_norm))
        K = self._heads(self.k_proj(kv_norm))
        V = self._heads(self.v_proj(kv_norm))
        out = F.scaled_dot_product_attention(Q, K, V)
        q = q + self.out_proj(self._unheads(out))
        qn = self.norm_self(q)
        qkv = self.self_qkv(qn).chunk(3, dim=-1)
        Q2, K2, V2 = [self._heads(t) for t in qkv]
        s_out = F.scaled_dot_product_attention(Q2, K2, V2)
        q = q + self.self_out(self._unheads(s_out))
        q = q + self.ffn(self.norm_ffn(q))
        return q

    def _heads(self, x):
        B, S, D = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def _unheads(self, x):
        B, H, S, D = x.shape
        return x.transpose(1, 2).reshape(B, S, H * D)


class SigLIPCompressor(nn.Module):
    def __init__(self, dim=768, num_queries=64, num_layers=2):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList([
            _SigLIPCompressorLayer(dim) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, x)
        return self.final_norm(q)


class IPSelfAttn(nn.Module):
    """1-layer self-attention with SDPA (Flash Attention) for O(N) memory."""
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        B, N, D = x.shape
        xn = self.norm1(x)
        qkv = self.qkv(xn)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        x = x + self.out_proj(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# ComfyUI Nodes
# ═══════════════════════════════════════════════════════════════════════════════

class AnimaIPAdapterLoader:
    """Loads SigLIP2 encoder and IP-Adapter K/V weights."""

    CATEGORY = "Anima/IP-Adapter"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ip_adapter_name": (folder_paths.get_filename_list("ipadapter"), ),
                "auto_download": ("BOOLEAN", {"default": False,
                    "tooltip": "Auto-download SigLIP2 encoder to models/siglip2/"}),
            },
        }

    RETURN_TYPES = ("ANIMA_IP_ADAPTER",)
    RETURN_NAMES = ("ip_adapter",)
    FUNCTION = "load"

    def load(self, ip_adapter_name, auto_download):
        import safetensors
        import safetensors.torch
        from transformers import SiglipVisionModel

        from . import SIGLIP2_DIR

        # ── Load SigLIP2 encoder ──
        siglip_id = "google/siglip2-base-patch16-512"
        if auto_download:
            os.makedirs(SIGLIP2_DIR, exist_ok=True)
            if not os.path.isfile(os.path.join(SIGLIP2_DIR, "config.json")):
                print(f"[AnimaIPAdapter] Downloading {siglip_id} ...")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=siglip_id, local_dir=SIGLIP2_DIR)
                print(f"[AnimaIPAdapter] SigLIP2 downloaded to {SIGLIP2_DIR}")

        if os.path.isdir(SIGLIP2_DIR):
            siglip_encoder = SiglipVisionModel.from_pretrained(SIGLIP2_DIR)
        else:
            siglip_encoder = SiglipVisionModel.from_pretrained(siglip_id)
        siglip_encoder.requires_grad_(False)
        siglip_encoder.eval()

        # ── Load IP-Adapter K/V weights + metadata ──
        ip_adapter_path = folder_paths.get_full_path_or_raise("ipadapter", ip_adapter_name)
        ip_sd = {}
        metadata = {}
        with safetensors.safe_open(ip_adapter_path, framework="pt") as f:
            metadata = f.metadata() or {}
            for k in f.keys():
                ip_sd[k] = f.get_tensor(k)
        ip_norm_keys = metadata.get("ip_norm_keys", "False").lower() == "true"
        ip_inject_before_mlp = metadata.get("ip_inject_before_mlp", "False").lower() == "true"
        print(f"[AnimaIPAdapter] Loaded {len(ip_sd)} IP-Adapter keys from {ip_adapter_path}"
              f"{' (ip_norm_keys)' if ip_norm_keys else ''}"
              f"{' (ip_inject_before_mlp)' if ip_inject_before_mlp else ''}")

        # Detect architecture
        use_shared = any(k.startswith("shared_ip_k_proj") for k in ip_sd.keys())

        # Parse per-block indices
        block_indices = set()
        for k in ip_sd.keys():
            m = re.match(r"blocks\.(\d+)\.(ip_k_proj|ip_v_proj|adaln_ip)", k)
            if m:
                block_indices.add(int(m.group(1)))
        num_blocks = max(block_indices) + 1 if block_indices else 28
        print(f"[AnimaIPAdapter] {num_blocks} blocks detected"
              f"{' (shared projection)' if use_shared else ''}")

        # Infer ip_embed_dim and inner_dim
        if use_shared:
            k0 = ip_sd.get("shared_ip_k_proj.expand.weight")
            if k0 is None:
                k0 = ip_sd.get("shared_ip_k_proj.0.weight")
            inner_dim, ip_embed_dim = k0.shape if k0 is not None else (SIGLIP_HIDDEN_SIZE, SIGLIP_HIDDEN_SIZE)
        else:
            sample_weight = None
            for probe in [f"blocks.{i}.ip_k_proj.weight" for i in range(num_blocks)]:
                if probe in ip_sd:
                    sample_weight = ip_sd[probe]
                    break
            if sample_weight is not None:
                inner_dim, ip_embed_dim = sample_weight.shape
            else:
                ip_embed_dim = SIGLIP_HIDDEN_SIZE
                inner_dim = ip_embed_dim
        print(f"[AnimaIPAdapter] ip_embed_dim={ip_embed_dim}, inner_dim={inner_dim}")

        # Parse LoRA weights (lora.* prefix from peft_state_dict)
        lora_weights = {}
        lora_rank = 0
        for k, v in ip_sd.items():
            if k.startswith("lora."):
                inner = k[len("lora."):]
                m = re.match(
                    r"base_model\.model\.blocks\.(\d+)\.cross_attn\.(\w+)\.lora_([AB])(?:\.default)?\.weight",
                    inner)
                if m:
                    bi = int(m.group(1))
                    ln = m.group(2)
                    ab = m.group(3)
                    lora_weights.setdefault(bi, {}).setdefault(ln, {})[ab] = v
                    if ab == "A":
                        lora_rank = max(lora_rank, v.shape[0])
        # If lora.* keys exist but none matched, print first 3 for debugging
        _lora_keys = [k for k in ip_sd if k.startswith("lora.")]
        if _lora_keys and not lora_weights:
            print(f"[AnimaIPAdapter] {len(_lora_keys)} lora keys unmatched, examples:")
            for k in _lora_keys[:3]:
                inner = k[len("lora."):]
                print(f"  {inner[:100]}")

        if lora_weights:
            print(f"[AnimaIPAdapter] Found LoRA: rank={lora_rank}, "
                  f"blocks={sorted(lora_weights.keys())}")

        # Parse IP self-attn weights (ip_self_attn.* prefix)
        self_attn_sd = {}
        for k, v in ip_sd.items():
            if k.startswith("ip_self_attn."):
                self_attn_sd[k[len("ip_self_attn."):]] = v
        ip_self_attn = None
        if self_attn_sd:
            module = IPSelfAttn(dim=ip_embed_dim)
            module.load_state_dict(self_attn_sd, strict=False)
            ip_self_attn = module
            print(f"[AnimaIPAdapter] Found IP-SelfAttn: {len(self_attn_sd)} keys")

        # Parse siglip_norm weights (LayerNorm for intermediate-layer feature normalization)
        siglip_norm_sd = {}
        for k, v in ip_sd.items():
            if k.startswith("siglip_norm."):
                siglip_norm_sd[k[len("siglip_norm."):]] = v
        siglip_norm = None
        if siglip_norm_sd:
            siglip_norm = nn.LayerNorm(ip_embed_dim, elementwise_affine=True)
            siglip_norm.load_state_dict(siglip_norm_sd)
            print(f"[AnimaIPAdapter] Found SigLIP LayerNorm (siglip_norm): {len(siglip_norm_sd)} keys")

        # Parse SigLIP compressor weights (learned query-based token compression)
        compressor_sd = {}
        for k, v in ip_sd.items():
            if k.startswith("siglip_compressor."):
                compressor_sd[k[len("siglip_compressor."):]] = v
        siglip_compressor = None
        if compressor_sd:
            # Detect num_queries from the learned queries parameter shape
            queries_shape = compressor_sd.get("queries")
            if queries_shape is not None:
                num_queries = queries_shape.shape[0]
            else:
                num_queries = 64  # fallback
            # Detect num_layers by counting layer indices in keys
            num_comp_layers = max(
                (int(k.split(".")[1]) + 1 for k in compressor_sd
                 if k.startswith("layers.")),
                default=2,
            )
            siglip_compressor = SigLIPCompressor(
                dim=ip_embed_dim, num_queries=num_queries,
                num_layers=num_comp_layers,
            )
            siglip_compressor.load_state_dict(compressor_sd, strict=False)
            print(f"[AnimaIPAdapter] Found SigLIPCompressor: "
                  f"{num_queries} tokens, {len(compressor_sd)} keys")

        # Extract learned null tokens (for CFG: represents "no IP signal")
        null_tokens = ip_sd.pop("null_tokens", None)
        if null_tokens is not None:
            print(f"[AnimaIPAdapter] Found learned null_tokens: shape={list(null_tokens.shape)}")

        return ({
            "ip_weights": ip_sd,
            "siglip_encoder": siglip_encoder,
            "num_blocks": num_blocks,
            "ip_embed_dim": ip_embed_dim,
            "inner_dim": inner_dim,
            "use_shared_projection": use_shared,
            "lora_weights": lora_weights,
            "lora_rank": lora_rank,
            "ip_self_attn": ip_self_attn,
            "siglip_norm": siglip_norm,
            "siglip_compressor": siglip_compressor,
            "ip_norm_keys": ip_norm_keys,
            "ip_inject_before_mlp": ip_inject_before_mlp,
            "null_tokens": null_tokens,
        },)


class AnimaIPAdapterApply:
    """Encodes ref image through SigLIP2, injects IP tokens via transformer_options.
    Creates ip_k_proj / ip_v_proj on each block at apply time if not present.

    ip_cfg_scale: 1.0 = IP CFG disabled (IP cancels in CFG diff).
      > 1.0 = IP CFG enabled (same as old ip_cfg=True).
      Use the "strength" parameter to control IP contribution magnitude.
    """

    CATEGORY = "Anima/IP-Adapter"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ip_adapter": ("ANIMA_IP_ADAPTER",),
                "ref_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0,
                                       "max": 2.0, "step": 0.05}),
                "ref_image_size": ("INT", {"default": 512, "min": 224, "max": 2048, "step": 16,
                                           "tooltip": "Must be multiple of 16 (patch_size)"}),
                "siglip_layer": ("INT", {"default": -1, "min": -1, "max": 24,
                                         "tooltip": "SigLIP2 hidden_states index. -1=last (default). Must match training config."}),
                "ip_cfg_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 10.0, "step": 0.05,
                                           "tooltip": "IP-Adapter CFG scale. 1.0 = disabled. > 1.0 = enabled."}),
                "ip_cfg_separate": ("BOOLEAN", {"default": False,
                                                "tooltip": "OFF: IP CFG binds to text CFG (old behavior, 1-pass). ON: independent IP CFG scale (2-pass, extra compute)."}),
                "gray_null": ("BOOLEAN", {"default": False,
                                          "tooltip": "ON: encode gray image as null token. OFF: use checkpoint null_tokens or zeros."}),
                "use_lora": ("BOOLEAN", {"default": True,
                                         "tooltip": "ON: apply LoRA weights to cross-attention layers. OFF: skip LoRA."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"

    def apply(self, model, ip_adapter, ref_image, strength, ref_image_size=512, siglip_layer=-1, ip_cfg_scale=4.0, ip_cfg_separate=False, gray_null=False, use_lora=True):
        patched_model = model.clone()
        siglip_encoder = ip_adapter["siglip_encoder"]
        ip_weights = ip_adapter["ip_weights"]
        num_blocks = ip_adapter["num_blocks"]
        ip_embed_dim = ip_adapter["ip_embed_dim"]
        inner_dim = ip_adapter["inner_dim"]
        device = torch.device("cuda")

        model_dtype = next(patched_model.model.diffusion_model.parameters()).dtype
        use_shared = ip_adapter.get("use_shared_projection", False)
        ip_norm_keys = ip_adapter.get("ip_norm_keys", False)
        ip_inject_before_mlp = ip_adapter.get("ip_inject_before_mlp", False)

        # ── Shared projection path ──
        if use_shared:
            dit = patched_model.model.diffusion_model
            if not hasattr(dit, 'shared_ip_k_proj'):
                dit.shared_ip_k_proj = _make_shared_proj(ip_embed_dim, inner_dim, device, model_dtype)
                dit.shared_ip_v_proj = _make_shared_proj(ip_embed_dim, inner_dim, device, model_dtype)

            for proj_name in ["shared_ip_k_proj", "shared_ip_v_proj"]:
                mlp = getattr(dit, proj_name)
                for pname, p in mlp.named_parameters():
                    key = f"{proj_name}.{pname}"
                    if key in ip_weights:
                        p.data.copy_(ip_weights[key].to(dtype=p.dtype, device=p.device))

            # Independent IP Q projection (shared across blocks)
            has_shared_q = any(k.startswith("shared_ip_q_proj") for k in ip_weights.keys())
            if has_shared_q and not hasattr(dit, 'shared_ip_q_proj'):
                x_dim = next(dit.parameters()).shape[0] if len(list(dit.parameters())) > 0 else inner_dim
                # Infer x_dim from adaln_ip.1.weight shape: [inner_dim, x_dim]
                try:
                    x_dim = ip_weights["blocks.0.adaln_ip.1.weight"].shape[1]
                except (KeyError, IndexError):
                    x_dim = inner_dim
                dit.shared_ip_q_proj = nn.Linear(x_dim, inner_dim, bias=False).to(
                    device=device, dtype=model_dtype)
            if has_shared_q:
                mlp = dit.shared_ip_q_proj
                for pname, p in mlp.named_parameters():
                    key = f"shared_ip_q_proj.{pname}"
                    if key in ip_weights:
                        p.data.copy_(ip_weights[key].to(dtype=p.dtype, device=p.device))

            for i, block in enumerate(dit.blocks):
                if i >= num_blocks:
                    break
                block_device = next(block.parameters()).device
                if not hasattr(block, 'ip_k_proj'):
                    block.ip_k_proj = _make_shared_proj_ref(dit.shared_ip_k_proj)
                    block.ip_v_proj = _make_shared_proj_ref(dit.shared_ip_v_proj)
                    block.adaln_ip = nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(inner_dim, inner_dim, bias=True, dtype=model_dtype, device=block_device),
                    )
                    nn.init.zeros_(block.adaln_ip[1].weight)
                    nn.init.constant_(block.adaln_ip[1].bias, 0.1)
                    block.use_ip_adapter = True
                    # V2 architecture flags
                    if not hasattr(block, 'ip_norm_keys'):
                        block.ip_norm_keys = ip_norm_keys
                        block.ip_inject_before_mlp = ip_inject_before_mlp
                # IP K normalization (InstantCharacter-style)
                if ip_norm_keys and not hasattr(block, 'ip_k_norm'):
                    h_d = block.cross_attn.head_dim
                    block.ip_k_norm = nn.LayerNorm(h_d, elementwise_affine=False, eps=1e-6).to(
                        device=block_device, dtype=model_dtype)
                # Independent IP Q
                if has_shared_q and not hasattr(block, 'ip_q_proj'):
                    block.ip_q_proj = _make_shared_proj_ref(dit.shared_ip_q_proj)
                    h_d = block.cross_attn.head_dim
                    block.ip_q_norm = nn.LayerNorm(h_d, elementwise_affine=False, eps=1e-6).to(
                        device=block_device, dtype=model_dtype)
                bp = dict(block.named_parameters())
                for pname in ["adaln_ip.1.weight", "adaln_ip.1.bias"]:
                    key = f"blocks.{i}.{pname}"
                    if key in ip_weights and pname in bp:
                        bp[pname].data.copy_(ip_weights[key].to(
                            dtype=bp[pname].dtype, device=bp[pname].device))
        else:
            # ── Per-block path (original, untouched) ──
            for i, block in enumerate(patched_model.model.diffusion_model.blocks):
                if i >= num_blocks:
                    break
                block_device = next(block.parameters()).device
                if not hasattr(block, 'ip_k_proj'):
                    block.ip_k_proj = nn.Linear(ip_embed_dim, inner_dim, bias=True,
                                                dtype=model_dtype, device=block_device)
                    block.ip_v_proj = nn.Linear(ip_embed_dim, inner_dim, bias=True,
                                                dtype=model_dtype, device=block_device)
                    nn.init.normal_(block.ip_k_proj.weight, std=1.0 / math.sqrt(ip_embed_dim))
                    nn.init.zeros_(block.ip_k_proj.bias)
                    nn.init.normal_(block.ip_v_proj.weight, std=1.0 / math.sqrt(ip_embed_dim))
                    nn.init.zeros_(block.ip_v_proj.bias)
                    block.adaln_ip = nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(inner_dim, inner_dim, bias=True, dtype=model_dtype, device=block_device),
                    )
                    nn.init.zeros_(block.adaln_ip[1].weight)
                    nn.init.constant_(block.adaln_ip[1].bias, 0.1)
                    block.use_ip_adapter = True
                    # V2 architecture flags
                    if not hasattr(block, 'ip_norm_keys'):
                        block.ip_norm_keys = ip_norm_keys
                        block.ip_inject_before_mlp = ip_inject_before_mlp
                # IP K normalization (InstantCharacter-style)
                if ip_norm_keys and not hasattr(block, 'ip_k_norm'):
                    h_d = block.cross_attn.head_dim
                    block.ip_k_norm = nn.LayerNorm(h_d, elementwise_affine=False, eps=1e-6).to(
                        device=block_device, dtype=model_dtype)
                block_params = dict(block.named_parameters())
                for pname in ["ip_k_proj.weight", "ip_k_proj.bias",
                              "ip_v_proj.weight", "ip_v_proj.bias",
                              "adaln_ip.1.weight", "adaln_ip.1.bias"]:
                    key = f"blocks.{i}.{pname}"
                    if key in ip_weights and pname in block_params:
                        block_params[pname].data.copy_(ip_weights[key].to(
                            dtype=block_params[pname].dtype,
                            device=block_params[pname].device,
                        ))

        # ── Install / restore LoRA wrappers on cross-attention layers ──
        lora_weights = ip_adapter.get("lora_weights", {})
        lora_rank = ip_adapter.get("lora_rank", 0)
        dit = patched_model.model.diffusion_model
        _lora_attrs = ("q_proj", "k_proj", "v_proj", "output_proj")
        if use_lora and lora_weights and lora_rank:
            alpha = lora_rank
            scale = alpha / lora_rank
            for bi, layers in lora_weights.items():
                if bi >= len(dit.blocks):
                    continue
                block = dit.blocks[bi]
                for ln, ab_dict in layers.items():
                    if "A" not in ab_dict or "B" not in ab_dict:
                        continue
                    original = getattr(block.cross_attn, ln, None)
                    if original is None or isinstance(original, _LoRALinear):
                        continue
                    wrapper = _LoRALinear(original, ab_dict["A"], ab_dict["B"], scale)
                    setattr(block.cross_attn, ln, wrapper)
            print(f"[AnimaIPAdapter] LoRA installed on {len(lora_weights)} blocks")
        else:
            # Unwrap any previously installed LoRA → restore original layers
            _unwrapped = 0
            for _blk in dit.blocks:
                for _ln in _lora_attrs:
                    _mod = getattr(_blk.cross_attn, _ln, None)
                    if isinstance(_mod, _LoRALinear):
                        setattr(_blk.cross_attn, _ln, _mod.base)
                        _unwrapped += 1
            if _unwrapped:
                print(f"[AnimaIPAdapter] LoRA unwrapped on {_unwrapped} layers")

        # ── Monkey-patch block forward for IP cross-attention ──
        # Each DiT block's forward is replaced so that after the original
        # forward, IP cross-attention is applied using the IP tokens from
        # transformer_options["anima_ip_tokens"].  This avoids modifying
        # ComfyUI's predict2.py core file.
        #
        # Uses persistent flags on blocks (_ip_fwd_patched, _ip_hook_installed)
        # to prevent double-patching when apply() is called multiple times
        # (model.clone() shares block objects).
        _patched_count = 0
        for _i, _blk in enumerate(patched_model.model.diffusion_model.blocks):
            if _i >= num_blocks:
                break
            if not getattr(_blk, 'use_ip_adapter', False):
                continue

            # Hook: capture cross_attn flat query input (once per block)
            if not getattr(_blk, '_ip_hook_installed', False):
                def _make_capture(_ref):
                    def _hook(_mod, _args):
                        _ref._x_cross_flat = _args[0].detach()
                    return _hook
                _blk.cross_attn.register_forward_pre_hook(_make_capture(_blk))
                _blk._ip_hook_installed = True

            # Replace forward (once per block)
            if getattr(_blk, '_ip_fwd_patched', False):
                continue  # already patched

            _orig_fwd = _blk.forward
            if 'ip_hidden_states' in inspect.signature(_orig_fwd).parameters:
                continue  # predict2.py already handles IP, skip monkey-patch

            def _make_fwd(_blk_ref, _orig):
                def _patched_forward(x_B_T_H_W_D, emb_B_T_D, crossattn_emb,
                                     rope_emb_L_1_1_D=None,
                                     adaln_lora_B_T_3D=None,
                                     extra_per_block_pos_emb=None,
                                     transformer_options=None,
                                     **__kwargs):
                    result = _orig(
                        x_B_T_H_W_D, emb_B_T_D, crossattn_emb,
                        rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                        adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                        extra_per_block_pos_emb=extra_per_block_pos_emb,
                        transformer_options=transformer_options,
                        **__kwargs,
                    )
                    if not getattr(_blk_ref, 'use_ip_adapter', False):
                        return result
                    if transformer_options is None:
                        return result
                    ip_tok = transformer_options.get("anima_ip_tokens", None)
                    if ip_tok is None:
                        return result
                    if not hasattr(_blk_ref, '_x_cross_flat'):
                        return result

                    x_q = _blk_ref._x_cross_flat
                    ip_tok = ip_tok.to(dtype=result.dtype)
                    B = x_B_T_H_W_D.shape[0]
                    T, H, W = x_B_T_H_W_D.shape[1], x_B_T_H_W_D.shape[2], x_B_T_H_W_D.shape[3]
                    n_h = _blk_ref.cross_attn.n_heads
                    h_d = _blk_ref.cross_attn.head_dim

                    gate_ip = _blk_ref.adaln_ip(emb_B_T_D)

                    # adapter_scale: 0 if all-null → clean CFG bypass
                    scale_mask = (ip_tok.abs().sum(dim=[1, 2]) > 1e-6).to(
                        dtype=result.dtype).reshape(B, 1, 1)

                    ip_q = _blk_ref.cross_attn.q_proj(x_q).reshape(B, -1, n_h, h_d).permute(0, 2, 1, 3)
                    ip_q = _blk_ref.cross_attn.q_norm(ip_q)
                    ip_k = _blk_ref.ip_k_proj(ip_tok).reshape(B, -1, n_h, h_d).permute(0, 2, 1, 3)
                    ip_v = _blk_ref.ip_v_proj(ip_tok).reshape(B, -1, n_h, h_d).permute(0, 2, 1, 3)
                    ip_kn = getattr(_blk_ref, 'ip_k_norm', None)
                    if ip_kn is not None:
                        ip_k = ip_kn(ip_k)

                    ip_attn = F.scaled_dot_product_attention(ip_q, ip_k, ip_v)
                    ip_out = ip_attn.permute(0, 2, 1, 3).reshape(B, T * H * W, n_h * h_d)

                    gate = gate_ip * scale_mask
                    result = result + gate.reshape(B, T, 1, 1, -1) * ip_out.reshape(B, T, H, W, -1)
                    return result
                return _patched_forward

            _blk.forward = _make_fwd(_blk, _orig_fwd)
            _blk._ip_fwd_patched = True
            _patched_count += 1

        if _patched_count:
            print(f"[AnimaIPAdapter] Patched {_patched_count} block forwards for IP cross-attention")

        # ── Encode reference image with SigLIP2 ──
        from torchvision import transforms as T

        norm = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        def resize_pad_tensor(pil_img, target=512):
            w, h = pil_img.size
            ratio = target / max(w, h)
            new_w = max(1, round(w * ratio))
            new_h = max(1, round(h * ratio))
            img = pil_img.resize((new_w, new_h), PILImage.BILINEAR)
            square = PILImage.new("RGB", (target, target), (0, 0, 0))
            px = (target - new_w) // 2
            py = (target - new_h) // 2
            square.paste(img, (px, py))
            return norm(T.ToTensor()(square))

        pil_images = []
        for b in range(ref_image.shape[0]):
            arr = (ref_image[b].cpu().numpy() * 255).astype(np.uint8)
            pil_images.append(PILImage.fromarray(arr, mode="RGB"))

        tensors = []
        for img in pil_images:
            tensors.append(resize_pad_tensor(img, ref_image_size))
        img_tensor = torch.stack(tensors).to(device=device, dtype=torch.float32)

        siglip_norm = ip_adapter.get("siglip_norm", None)
        siglip_encoder.to(device)
        with torch.no_grad():
            if siglip_layer == -1:
                ip_tokens = siglip_encoder(
                    img_tensor, interpolate_pos_encoding=True,
                ).last_hidden_state
            else:
                outputs = siglip_encoder(
                    img_tensor, interpolate_pos_encoding=True,
                    output_hidden_states=True,
                )
                ip_tokens = outputs.hidden_states[siglip_layer]
                if siglip_norm is not None:
                    siglip_norm.to(device)
                    ip_tokens = siglip_norm(ip_tokens)
            ip_tokens = ip_tokens * strength
        siglip_encoder.to("cpu")

        # SigLIPCompressor: learned query-based compression (must match training config)
        siglip_compressor = ip_adapter.get("siglip_compressor", None)
        if siglip_compressor is not None:
            siglip_compressor.to(device)
            ip_tokens = siglip_compressor(ip_tokens)

        # IP self-attention: token-to-token context exchange
        ip_self_attn = ip_adapter.get("ip_self_attn", None)
        if ip_self_attn is not None:
            ip_self_attn.to(device)
            ip_tokens = ip_self_attn(ip_tokens)

        print(f"[AnimaIPAdapter] ip_tokens shape={list(ip_tokens.shape)}, "
              f"norm={ip_tokens.norm().item():.4f}, strength={strength}")

        # Null tokens: gray-image encoding or checkpoint/zeros
        ip_tokens_stored = ip_tokens.detach()
        null_tokens_stored = ip_adapter.get("null_tokens", None)
        if gray_null:
            # Encode a gray image → natural "no signal" embedding in SigLIP2 space
            gray_img = PILImage.new("RGB", (ref_image_size, ref_image_size),
                                    color=(128, 128, 128))
            gray_tensor = resize_pad_tensor(gray_img, ref_image_size).unsqueeze(0).to(
                device=device, dtype=torch.float32)
            siglip_encoder.to(device)
            with torch.no_grad():
                if siglip_layer == -1:
                    null_tokens_stored = siglip_encoder(
                        gray_tensor, interpolate_pos_encoding=True,
                    ).last_hidden_state
                else:
                    outputs = siglip_encoder(
                        gray_tensor, interpolate_pos_encoding=True,
                        output_hidden_states=True,
                    )
                    null_tokens_stored = outputs.hidden_states[siglip_layer]
                    if siglip_norm is not None:
                        siglip_norm.to(device)
                        null_tokens_stored = siglip_norm(null_tokens_stored)
            siglip_encoder.to("cpu")
            if siglip_compressor is not None:
                siglip_compressor.to(device)
                null_tokens_stored = siglip_compressor(null_tokens_stored)
            if ip_self_attn is not None:
                ip_self_attn.to(device)
                null_tokens_stored = ip_self_attn(null_tokens_stored)
            null_tokens_stored = null_tokens_stored.detach().to(dtype=model_dtype)
        elif null_tokens_stored is not None:
            null_tokens_stored = null_tokens_stored.to(dtype=model_dtype, device=device)
            if null_tokens_stored.shape != ip_tokens_stored.shape:
                null_tokens_stored = null_tokens_stored.expand_as(ip_tokens_stored)
        else:
            null_tokens_stored = torch.zeros_like(ip_tokens_stored)
        print(f"[AnimaIPAdapter] null_tokens norm={null_tokens_stored.norm().item():.4f}" +
              (" (gray)" if gray_null else " (checkpoint)" if ip_adapter.get("null_tokens") is not None else " (zeros)"))

        class IPAdapterHandler:
            """IP token injection with optional independent IP CFG scaling.

            ip_cfg_separate = False: bind to text CFG (old behavior, 1-pass).
              ip_cfg_scale > 1.0 → cond=real, uncond=null → IP scaled by text CFG.
              ip_cfg_scale = 1.0 → all same → IP cancels → disabled.

            ip_cfg_separate = True: independent IP CFG (2-pass).
              Runs model twice (with/without IP), blends via post_cfg_function:
                uncond + text_cfg*(cond_wo_ip-uncond) + ip_eff*(cond_w_ip-cond_wo_ip)
              ip_eff = ip_cfg_scale - 1.0
            """

            def __init__(self, ip_tokens, null_tokens, ip_cfg_scale, ip_cfg_separate):
                self.ip_tokens = ip_tokens
                self.null_tokens = null_tokens
                self.ip_cfg_scale = ip_cfg_scale
                self.ip_cfg_separate = ip_cfg_separate
                self._printed = False
                self._cond_wo_ip = None      # prediction without IP (for post_cfg)
                # When bound to text CFG, always enabled (old behavior).
                # When independent, enabled only if ip_cfg_scale > 1.0.
                self._enabled = (not ip_cfg_separate) or (ip_cfg_scale > 1.0)

            def __call__(self, apply_model_fn, args_dict):
                """Model unet function wrapper."""
                model_input = args_dict["input"]
                timestep = args_dict["timestep"]
                c = dict(args_dict["c"])
                cond_or_uncond = args_dict.get("cond_or_uncond", None)

                all_uncond = (cond_or_uncond is not None
                              and all(u == 1 for u in cond_or_uncond))

                if not all_uncond:
                    ip_B = model_input.shape[0]
                    target_dtype = model_input.dtype
                    target_device = model_input.device

                    ip_tok = self.ip_tokens.to(dtype=target_dtype,
                                                device=target_device)
                    ip_tok_batch = ip_tok.expand(ip_B, -1, -1).clone()

                    if self._enabled and cond_or_uncond is not None:
                        # cond → real, uncond → null (common to both modes)
                        null_tok = self.null_tokens.to(dtype=target_dtype, device=target_device)
                        for idx, is_uncond in enumerate(cond_or_uncond):
                            if idx < ip_B and is_uncond:
                                ip_tok_batch[idx] = null_tok

                        c["transformer_options"] = dict(c.get("transformer_options", {}))
                        c["transformer_options"]["anima_ip_tokens"] = ip_tok_batch

                        if self.ip_cfg_separate:
                            # ── 2-pass: independent IP CFG ──
                            out = apply_model_fn(model_input, timestep, **c)

                            # Second call: ALL null → cond_wo_ip
                            null_batch = null_tok.expand(ip_B, -1, -1)
                            c_null = dict(c)
                            to_null = dict(c["transformer_options"])
                            to_null["anima_ip_tokens"] = null_batch
                            c_null["transformer_options"] = to_null
                            out_null = apply_model_fn(model_input, timestep, **c_null)

                            batch_chunks = len(cond_or_uncond)
                            chunks_wo = out_null.chunk(batch_chunks)
                            for idx, is_uncond in enumerate(cond_or_uncond):
                                if not is_uncond and idx < len(chunks_wo):
                                    self._cond_wo_ip = chunks_wo[idx]
                                    break

                            if not self._printed:
                                print(f"[AnimaIPAdapter] Independent IP CFG (2-pass), "
                                      f"batch={ip_B}, ip_cfg_scale={self.ip_cfg_scale}")
                                self._printed = True
                            return out
                        else:
                            # ── 1-pass: bind to text CFG (old behavior) ──
                            if not self._printed:
                                print(f"[AnimaIPAdapter] IP CFG bound to text CFG (1-pass), "
                                      f"batch={ip_B}, ip_cfg_scale={self.ip_cfg_scale}")
                                self._printed = True
                            return apply_model_fn(model_input, timestep, **c)

                    else:
                        # Disabled (ip_cfg_scale == 1.0): all same → IP cancels
                        if not self._printed:
                            print(f"[AnimaIPAdapter] IP CFG disabled (batch={ip_B})")
                            self._printed = True

                    c["transformer_options"] = dict(c.get("transformer_options", {}))
                    c["transformer_options"]["anima_ip_tokens"] = ip_tok_batch

                return apply_model_fn(model_input, timestep, **c)

            def post_cfg(self, args):
                """Post-CFG correction for independent IP CFG (only when ip_cfg_separate)."""
                if not self.ip_cfg_separate or not self._enabled or self._cond_wo_ip is None:
                    return args["denoised"]

                cond_w_ip = args["cond_denoised"]
                uncond = args["uncond_denoised"]
                cond_scale = args["cond_scale"]
                cond_wo_ip = self._cond_wo_ip.to(dtype=cond_w_ip.dtype,
                                                  device=cond_w_ip.device)

                ip_eff = self.ip_cfg_scale - 1.0
                # Standard: uncond + text_cfg * (cond_w_ip - uncond)
                # Desired:  uncond + text_cfg * (cond_wo_ip - uncond)
                #                 + ip_eff * (cond_w_ip - cond_wo_ip)
                # Correction = (ip_eff - text_cfg) * (cond_w_ip - cond_wo_ip)
                correction = (ip_eff - cond_scale) * (cond_w_ip - cond_wo_ip)
                return args["denoised"] + correction

            def to(self, *args, **kwargs):
                return self

        handler = IPAdapterHandler(ip_tokens_stored, null_tokens_stored, ip_cfg_scale, ip_cfg_separate)
        patched_model.set_model_unet_function_wrapper(handler)
        patched_model.set_model_sampler_post_cfg_function(handler.post_cfg)
        return (patched_model,)


class AnimaIPAdapterVisualize:
    """Visualize where the IP-Adapter is 'looking' in the reference image."""

    CATEGORY = "Anima/IP-Adapter"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ip_adapter": ("ANIMA_IP_ADAPTER",),
                "ref_image": ("IMAGE",),
                "mode": (["key_norm", "token_norm", "key_unique", "combined"], {
                    "default": "combined"}),
                "opacity": ("FLOAT", {"default": 0.6, "min": 0.1,
                                      "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("heatmap",)
    FUNCTION = "visualize"

    def visualize(self, ip_adapter, ref_image, mode, opacity):
        import torch.nn.functional as F
        from torchvision import transforms as T

        siglip_encoder = ip_adapter["siglip_encoder"]
        ip_weights = ip_adapter["ip_weights"]
        num_blocks = ip_adapter["num_blocks"]
        device = torch.device("cuda")

        norm_tf = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        def resize_pad_tensor(pil_img, target=512):
            w, h = pil_img.size
            ratio = target / max(w, h)
            new_w = max(1, round(w * ratio))
            new_h = max(1, round(h * ratio))
            img = pil_img.resize((new_w, new_h), PILImage.BILINEAR)
            square = PILImage.new("RGB", (target, target), (0, 0, 0))
            px = (target - new_w) // 2
            py = (target - new_h) // 2
            square.paste(img, (px, py))
            return square

        arr = (ref_image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = PILImage.fromarray(arr, mode="RGB")

        square = resize_pad_tensor(pil_img)
        img_tensor = norm_tf(T.ToTensor()(square)).unsqueeze(0).to(
            device=device, dtype=torch.float32)

        siglip_encoder.to(device)
        with torch.no_grad():
            ip_tokens = siglip_encoder(
                img_tensor, interpolate_pos_encoding=True).last_hidden_state
        siglip_encoder.to("cpu")

        tokens = ip_tokens[0].float()
        token_norm_map = tokens.norm(dim=1)

        key_norm_map = torch.zeros(1024, device=device)
        key_unique_map = torch.zeros(1024, device=device)
        blocks_with_weights = 0

        for i in range(num_blocks):
            w_key = f"blocks.{i}.ip_k_proj.weight"
            if w_key not in ip_weights:
                continue
            w = ip_weights[w_key].float().to(device)
            keys = tokens @ w.T
            key_norm_map += keys.norm(dim=1)

            k_norm = F.normalize(keys, dim=1)
            sim = k_norm @ k_norm.T
            avg_cos = (sim.sum(dim=1) - 1.0) / 1023.0
            key_unique_map += (1.0 - avg_cos)
            blocks_with_weights += 1

        if blocks_with_weights > 0:
            key_norm_map /= blocks_with_weights
            key_unique_map /= blocks_with_weights

        if mode == "token_norm":
            score_map = token_norm_map
        elif mode == "key_norm":
            score_map = key_norm_map
        elif mode == "key_unique":
            score_map = key_unique_map
        else:
            def minmax(x):
                return (x - x.min()) / (x.max() - x.min() + 1e-8)
            score_map = (minmax(token_norm_map) + minmax(key_norm_map) + minmax(key_unique_map)) / 3.0

        score_2d = score_map.cpu().numpy().reshape(32, 32)
        score_tensor = torch.from_numpy(score_2d).float().unsqueeze(0).unsqueeze(0)
        score_up = F.interpolate(score_tensor, size=(512, 512), mode='bilinear',
                                 align_corners=False)
        score_up = score_up.squeeze().numpy()

        smin, smax = score_up.min().item(), score_up.max().item()
        if smax - smin > 1e-8:
            score_up = (score_up - smin) / (smax - smin)

        heatmap_rgb = np.zeros((512, 512, 3), dtype=np.float32)
        for c in range(3):
            lo = [0.0, 0.0, 0.8][c]
            mid = [0.0, 1.0, 0.0][c]
            hi = [1.0, 0.0, 0.0][c]
            heatmap_rgb[:, :, c] = np.where(
                score_up < 0.5,
                lo + (mid - lo) * (score_up / 0.5),
                mid + (hi - mid) * ((score_up - 0.5) / 0.5)
            )

        ref_rgb = T.ToTensor()(square).permute(1, 2, 0).cpu().numpy()
        overlay = heatmap_rgb * opacity + ref_rgb * (1.0 - opacity)
        overlay = np.clip(overlay, 0.0, 1.0)

        result = torch.from_numpy(overlay).float().unsqueeze(0)

        print(f"[AnimaIPAdapter] Heatmap: mode={mode}, "
              f"token_norm={token_norm_map.mean().item():.4f}, "
              f"key_norm={key_norm_map.mean().item():.4f}, "
              f"key_unique={key_unique_map.mean().item():.4f}")

        return (result,)


NODE_CLASS_MAPPINGS = {
    "AnimaIPAdapterLoader": AnimaIPAdapterLoader,
    "AnimaIPAdapterApply": AnimaIPAdapterApply,
    "AnimaIPAdapterVisualize": AnimaIPAdapterVisualize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaIPAdapterLoader": "Anima IP-Adapter Loader (SigLIP2)",
    "AnimaIPAdapterApply": "Anima IP-Adapter Apply (SigLIP2)",
    "AnimaIPAdapterVisualize": "Anima IP Attn Heatmap",
}
