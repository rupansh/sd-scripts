import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# Anima の Attention クラス名 (library/anima_models.py の Attention)
TARGET_ATTENTION_CLASS = "Attention"

# LLM Adapter 配下は学習対象外
LLM_ADAPTER_NAME = "llm_adapter"


class LLLiteModuleDiT(nn.Module):
    """単一の Attention Linear (q_proj/k_proj/v_proj) に対し LLLite の補正 x + cx を注入する."""

    def __init__(
        self,
        name: str,
        org_module: nn.Linear,
        cond_emb_dim: int,
        mlp_dim: int,
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
    ):
        super().__init__()
        self.lllite_name = name
        # list 包みで nn.Module 登録を回避し、state_dict に元 Linear の重みが入らないようにする
        self.org_module = [org_module]
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.multiplier = multiplier

        in_dim = org_module.in_features

        self.down = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(inplace=True),
        )
        self.mid = nn.Sequential(
            nn.Linear(mlp_dim + cond_emb_dim, mlp_dim),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Linear(mlp_dim, in_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        # 親 ControlNetLLLiteDiT が set_cond_image で注入する
        self.cond_emb: Optional[torch.Tensor] = None

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D) — Anima attention 入力は b (t h w) d で flatten 済み
        if self.multiplier == 0.0 or self.cond_emb is None:
            return self.org_forward(x)

        cx = self.cond_emb  # (B, H*W, cond_emb_dim)

        # CFG 推論用 (学習時は通らない想定)
        if x.shape[0] // 2 == cx.shape[0]:
            cx = cx.repeat(2, 1, 1)

        # T=1 固定前提なので S == H*W のはず
        assert x.shape[1] == cx.shape[1], (
            f"LLLite seq mismatch ({self.lllite_name}): x={x.shape[1]} vs cond_emb={cx.shape[1]}"
        )

        cx = torch.cat([cx, self.down(x)], dim=-1)  # (B, S, mlp+cond)
        cx = self.mid(cx)
        if self.dropout is not None and self.training:
            cx = F.dropout(cx, p=self.dropout)
        cx = self.up(cx) * self.multiplier

        return self.org_forward(x + cx)


class ControlNetLLLiteDiT(nn.Module):
    """Anima DiT 用の ControlNet-LLLite 本体. conditioning1 を共有保持し、各 Attention Linear に LLLite を貼る."""

    TARGET_LAYERS_CHOICES = ("self_attn_q", "self_attn_qkv", "self_attn_qkv_cross_q")

    def __init__(
        self,
        dit: nn.Module,
        cond_emb_dim: int = 32,
        mlp_dim: int = 64,
        target_layers: str = "self_attn_q",
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
    ):
        super().__init__()
        if target_layers not in self.TARGET_LAYERS_CHOICES:
            raise ValueError(
                f"Unknown target_layers: {target_layers}. choices={self.TARGET_LAYERS_CHOICES}"
            )

        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.target_layers = target_layers
        self.dropout = dropout
        self.multiplier = multiplier

        # cond image (B,3,H*16,W*16) -> (B, cond_emb_dim, H, W) (stride 16)
        self.conditioning1 = nn.Sequential(
            nn.Conv2d(3, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=4, stride=4, padding=0),
        )

        modules = self._create_modules(dit, cond_emb_dim, mlp_dim, target_layers, dropout, multiplier)
        self.lllite_modules = nn.ModuleList(modules)
        logger.info(
            f"ControlNet-LLLite (Anima): created {len(self.lllite_modules)} modules for target={target_layers}"
        )

    @staticmethod
    def _should_apply(is_self_attn: bool, child_name: str, target_layers: str) -> bool:
        # 常時スキップ
        if "output_proj" in child_name:
            return False
        if (not is_self_attn) and (child_name in ("k_proj", "v_proj")):
            return False  # cross_attn の K,V は text 側で shape 不一致

        if target_layers == "self_attn_q":
            return is_self_attn and child_name == "q_proj"
        elif target_layers == "self_attn_qkv":
            return is_self_attn and child_name in ("q_proj", "k_proj", "v_proj")
        elif target_layers == "self_attn_qkv_cross_q":
            if is_self_attn and child_name in ("q_proj", "k_proj", "v_proj"):
                return True
            if (not is_self_attn) and child_name == "q_proj":
                return True
            return False
        else:
            raise ValueError(f"Unknown target_layers: {target_layers}")

    def _create_modules(
        self,
        dit: nn.Module,
        cond_emb_dim: int,
        mlp_dim: int,
        target_layers: str,
        dropout: Optional[float],
        multiplier: float,
    ) -> List[LLLiteModuleDiT]:
        modules: List[LLLiteModuleDiT] = []
        for name, module in dit.named_modules():
            if module.__class__.__name__ != TARGET_ATTENTION_CLASS:
                continue
            # LLM Adapter 配下は除外 (クラス名でほぼ落ちるが name でも明示防御)
            if LLM_ADAPTER_NAME in name:
                continue
            if not hasattr(module, "is_selfattn"):
                continue
            is_self_attn = bool(module.is_selfattn)

            for child_name, child in module.named_children():
                if not isinstance(child, nn.Linear):
                    continue
                if not self._should_apply(is_self_attn, child_name, target_layers):
                    continue
                full_name = f"lllite_dit.{name}.{child_name}".replace(".", "_")
                modules.append(
                    LLLiteModuleDiT(
                        full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier
                    )
                )
        return modules

    def set_cond_image(self, cond_image: Optional[torch.Tensor]):
        """cond_image: (B, 3, H*16, W*16). None で解除."""
        if cond_image is None:
            for m in self.lllite_modules:
                m.cond_emb = None
            return
        cx = self.conditioning1(cond_image)  # (B, C, H, W)
        b, c, h, w = cx.shape
        cx = cx.view(b, c, h * w).permute(0, 2, 1).contiguous()  # (B, H*W, C)
        for m in self.lllite_modules:
            m.cond_emb = cx

    def clear_cond_image(self):
        self.set_cond_image(None)

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier
        for m in self.lllite_modules:
            m.multiplier = multiplier

    def apply_to(self):
        for m in self.lllite_modules:
            m.apply_to()


class AnimaControlNetLLLiteWrapper(nn.Module):
    """accelerator.prepare に渡す最上位 nn.Module.
    forward 内で lllite.set_cond_image を呼んで cond の計算を accumulate/autocast/DDP スコープに入れる."""

    def __init__(self, dit: nn.Module, lllite: ControlNetLLLiteDiT):
        super().__init__()
        self.dit = dit
        self.lllite = lllite

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        cond_image: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # T=1 固定
        assert x.shape[2] == 1, f"Anima LLLite supports T=1 only, got T={x.shape[2]}"
        if cond_image is not None:
            # 解像度整合チェック: x は VAE latent (/8)、cond_image は元画像 (/1)。
            # patchify (/2) は DiT 内部 (prepare_embedded_sequence) で実施されるため、
            # ここでは latent HW * 8 == cond_image HW を期待する。
            # conditioning1 (stride 16) は cond_image を /16 = latent/2 = token 空間に揃える。
            expected_h = x.shape[-2] * 8
            expected_w = x.shape[-1] * 8
            assert cond_image.shape[-2] == expected_h and cond_image.shape[-1] == expected_w, (
                f"cond_image HW mismatch: latent={x.shape[-2]}x{x.shape[-1]} -> expected "
                f"{expected_h}x{expected_w}, got {cond_image.shape[-2]}x{cond_image.shape[-1]}"
            )
            self.lllite.set_cond_image(cond_image)
        return self.dit(x, timesteps, context, **kwargs)


# ---------------------------------------------------------------------------
# save / load helpers
# ---------------------------------------------------------------------------

def save_lllite_model(
    file: str,
    lllite: ControlNetLLLiteDiT,
    dtype: Optional[torch.dtype] = None,
    metadata: Optional[dict] = None,
):
    state_dict = lllite.state_dict()
    if dtype is not None:
        for k in list(state_dict.keys()):
            state_dict[k] = state_dict[k].detach().clone().to("cpu").to(dtype)
    else:
        for k in list(state_dict.keys()):
            state_dict[k] = state_dict[k].detach().clone().to("cpu")

    if metadata is not None and len(metadata) == 0:
        metadata = None

    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file

        save_file(state_dict, file, metadata)
    else:
        torch.save(state_dict, file)


def load_lllite_weights(lllite: ControlNetLLLiteDiT, file: str, strict: bool = False):
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file

        weights_sd = load_file(file)
    else:
        weights_sd = torch.load(file, map_location="cpu")
    info = lllite.load_state_dict(weights_sd, strict=strict)
    logger.info(f"loaded LLLite weights from {file}: {info}")
    return info


# ---------------------------------------------------------------------------
# Phase A 動作確認用ダミー実行
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ダミー Attention/DiT を組み立て、構築・apply_to・state_dict・forward を一通り検査する
    class _DummyAttention(nn.Module):
        def __init__(self, dim: int, ctx_dim: Optional[int]):
            super().__init__()
            self.is_selfattn = ctx_dim is None
            qd = dim
            kd = dim if ctx_dim is None else ctx_dim
            self.q_proj = nn.Linear(qd, dim, bias=False)
            self.k_proj = nn.Linear(kd, dim, bias=False)
            self.v_proj = nn.Linear(kd, dim, bias=False)
            self.output_proj = nn.Linear(dim, dim, bias=False)

        # 名前が "Attention" であることが重要 (TARGET_ATTENTION_CLASS と一致させる)

    # 実 Attention クラスを TARGET_ATTENTION_CLASS と同名にするためエイリアス
    Attention = _DummyAttention
    Attention.__name__ = "Attention"

    class _DummyBlock(nn.Module):
        def __init__(self, dim: int, ctx_dim: int):
            super().__init__()
            self.self_attn = Attention(dim, None)
            self.cross_attn = Attention(dim, ctx_dim)

    class _DummyDiT(nn.Module):
        def __init__(self, num_blocks: int = 4, dim: int = 64, ctx_dim: int = 128):
            super().__init__()
            self.blocks = nn.ModuleList([_DummyBlock(dim, ctx_dim) for _ in range(num_blocks)])

        def forward(self, x, t, ctx, **kwargs):
            # x: (B, C=dim, T, H, W) 想定だがダミーなので形のみ通す
            return x

    logger.info("Phase A: dummy build / apply_to / state_dict")
    dit = _DummyDiT(num_blocks=4, dim=64, ctx_dim=128)

    for tl in ControlNetLLLiteDiT.TARGET_LAYERS_CHOICES:
        lllite = ControlNetLLLiteDiT(dit, cond_emb_dim=32, mlp_dim=64, target_layers=tl)
        expected = {"self_attn_q": 4, "self_attn_qkv": 12, "self_attn_qkv_cross_q": 16}[tl]
        assert len(lllite.lllite_modules) == expected, (
            f"target={tl}: expected {expected} modules, got {len(lllite.lllite_modules)}"
        )
        keys = list(lllite.state_dict().keys())
        assert any(k.startswith("conditioning1.") for k in keys)
        assert any(k.startswith("lllite_modules.0.down.") for k in keys)
        assert all("org_module" not in k for k in keys)
        logger.info(f"  target_layers={tl}: {len(lllite.lllite_modules)} modules OK")

    # apply_to + dummy forward
    dit2 = _DummyDiT(num_blocks=2, dim=64, ctx_dim=128)
    lllite2 = ControlNetLLLiteDiT(dit2, cond_emb_dim=32, mlp_dim=64, target_layers="self_attn_qkv_cross_q")
    lllite2.apply_to()
    wrapper = AnimaControlNetLLLiteWrapper(dit2, lllite2)

    B, H, W = 1, 8, 8
    x = torch.randn(B, 16, 1, H, W)
    cond_image = torch.randn(B, 3, H * 16, W * 16)
    # ダミー Attention は forward を持たないので、wrapper.forward は dit のダミー forward に到達するだけだが、
    # set_cond_image だけ確認する
    wrapper.lllite.set_cond_image(cond_image)
    cx = wrapper.lllite.lllite_modules[0].cond_emb
    assert cx is not None and cx.shape == (B, H * W, 32), f"unexpected cond_emb shape: {cx.shape}"
    logger.info(f"  set_cond_image OK: cond_emb={tuple(cx.shape)}")

    # LLLite forward 単体
    mod = wrapper.lllite.lllite_modules[0]
    seq = H * W
    x_seq = torch.randn(B, seq, mod.org_module[0].in_features)
    y = mod(x_seq)  # zero-init なので org_forward(x) と一致するはず
    assert y.shape == x_seq.shape
    # zero-init 確認: up.weight=0 なので cx=0、結果は org_forward(x) と等しい
    y_ref = mod.org_forward(x_seq)
    assert torch.allclose(y, y_ref), "zero-init forward mismatch"
    logger.info("  LLLiteModuleDiT zero-init forward OK")

    logger.info("Phase A dummy check PASSED")
