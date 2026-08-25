"""Frame ingestion hardening — the first thing that breaks in production.

In the lab a frame is always ``(256, 256, 3)`` uint8 straight out of
``env.render()``. Against a real game it arrives from a screen-capture stack,
and every one of these has been observed in the wild:

    * channel-last vs channel-first, RGB vs BGR (OpenCV hands you BGR)
    * an alpha channel from a compositor, or a single grayscale plane
    * uint8, uint16 (10-bit capture), float in ``[0, 1]``, float in ``[0, 255]``
    * NaN/Inf pixels from a half-precision or HDR path
    * a resolution that isn't the model's, and isn't even the same aspect ratio
    * an all-black frame while the game is loading or the window is occluded
    * the *same* frame forever, because capture silently stalled

A policy fed any of these produces confident nonsense, which is worse than
producing nothing: the agent keeps holding the stick while the game state has
moved on. :class:`FrameSanitizer` converts what it can, flags what it can't,
and reports *why* — so the runtime above it can decide to fall back instead of
acting on garbage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FrameHealth:
    """Verdict on one ingested frame.

    ``ok`` means the pixels are safe to hand to the policy. ``black`` and
    ``frozen`` are *advisory*: the frame is structurally valid, but the runtime
    may still prefer a neutral action over acting on a loading screen or a
    stalled capture.
    """

    ok: bool
    reason: str = ""
    black: bool = False
    frozen: bool = False
    saturated: bool = False
    repaired: Tuple[str, ...] = ()

    @property
    def degraded(self) -> bool:
        """True when the frame is usable but not trustworthy."""
        return self.black or self.frozen or bool(self.repaired)


def resize_nearest(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Dependency-free nearest-neighbour resize of an ``(H, W, C)`` array.

    Nearest-neighbour rather than bilinear on purpose: the vision encoder does
    its own bilinear resize to the model resolution anyway, so a second smooth
    filter here would only blur the frame twice for no gain.
    """
    h, w = img.shape[:2]
    if (h, w) == (height, width):
        return img
    ys = (np.arange(height) * (h / height)).astype(np.int64).clip(0, h - 1)
    xs = (np.arange(width) * (w / width)).astype(np.int64).clip(0, w - 1)
    return img[ys[:, None], xs[None, :]]


def letterbox(img: np.ndarray, height: int, width: int, pad: float = 0.0) -> np.ndarray:
    """Resize preserving aspect ratio, padding the remainder with ``pad``.

    Ultrawide (21:9) and vertical mobile captures squashed straight to a square
    change every on-screen distance the policy learned from, which is exactly
    the cue the toy tasks are built on. Letterboxing keeps geometry honest at
    the cost of some wasted pixels.
    """
    h, w = img.shape[:2]
    scale = min(height / h, width / w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = resize_nearest(img, new_h, new_w)
    out = np.full((height, width, img.shape[2]), pad, dtype=resized.dtype)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized
    return out


class FrameSanitizer:
    """Normalize a capture-pipeline frame into ``(H, W, 3)`` float32 in ``[0, 1]``.

    Args:
        target_size: ``(height, width)`` to resize to, or ``None`` to keep the
            incoming resolution (the encoder resizes anyway; set this when you
            want a fixed cost per frame regardless of capture resolution).
        keep_aspect: letterbox instead of stretching when resizing.
        bgr: swap the first and third channels (OpenCV / DirectX capture).
        black_threshold: mean intensity below which a frame counts as black.
        saturation_threshold: fraction of pixels at 1.0 that counts as blown out.
        freeze_patience: how many *consecutive* identical frames before the
            capture is called frozen. 1 would fire on any legitimately static
            scene (a paused menu, a cutscene letterbox); the default of 3 is a
            deliberate trade of detection latency against false positives.
        freeze_tolerance: max mean absolute difference still considered
            "identical". Non-zero tolerates encoder noise in a lossy capture.
    """

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        keep_aspect: bool = False,
        bgr: bool = False,
        black_threshold: float = 0.02,
        saturation_threshold: float = 0.98,
        freeze_patience: int = 3,
        freeze_tolerance: float = 0.0,
    ):
        if freeze_patience < 1:
            raise ValueError("freeze_patience must be >= 1")
        self.target_size = target_size
        self.keep_aspect = keep_aspect
        self.bgr = bgr
        self.black_threshold = black_threshold
        self.saturation_threshold = saturation_threshold
        self.freeze_patience = int(freeze_patience)
        self.freeze_tolerance = float(freeze_tolerance)
        self.reset()

    def reset(self) -> None:
        """Forget freeze history — call between episodes or after a reconnect."""
        self._signature: Optional[np.ndarray] = None
        self._repeats = 0
        self.last_good: Optional[np.ndarray] = None

    # -- shape / dtype coercion -----------------------------------------
    @staticmethod
    def _to_hwc(arr: np.ndarray, repaired: List[str]) -> Optional[np.ndarray]:
        if arr.ndim == 2:                                   # grayscale plane
            repaired.append("gray_to_rgb")
            return np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim != 3:
            return None
        # Channel-first only if the leading axis looks like channels *and* the
        # trailing axis doesn't — a (3, 4, 3) array is genuinely ambiguous, so
        # prefer the channel-last reading rather than guessing.
        if arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
            repaired.append("chw_to_hwc")
            arr = np.transpose(arr, (1, 2, 0))
        c = arr.shape[2]
        if c == 1:
            repaired.append("gray_to_rgb")
            return np.repeat(arr, 3, axis=2)
        if c == 4:
            repaired.append("dropped_alpha")
            return arr[:, :, :3]
        if c == 3:
            return arr
        return None

    @staticmethod
    def _to_unit_float(arr: np.ndarray, repaired: List[str]) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr.astype(np.float32) / 255.0
        if arr.dtype == np.uint16:
            repaired.append("uint16_scaled")
            return arr.astype(np.float32) / 65535.0
        if arr.dtype == bool:
            return arr.astype(np.float32)
        out = arr.astype(np.float32)
        finite = out[np.isfinite(out)]
        # A float frame in [0, 255] is the single most common capture bug: the
        # producer converted dtype but forgot to divide.
        if finite.size and float(finite.max()) > 1.5:
            repaired.append("rescaled_from_255")
            out = out / 255.0
        return out

    # -- main entry point ------------------------------------------------
    def __call__(self, frame) -> Tuple[Optional[np.ndarray], FrameHealth]:
        """Return ``(pixels, health)``; ``pixels`` is ``None`` when unusable."""
        repaired: List[str] = []
        if frame is None:
            return None, FrameHealth(ok=False, reason="frame_is_none")
        try:
            arr = np.asarray(frame)
        except Exception:  # ragged lists, exotic objects
            return None, FrameHealth(ok=False, reason="not_array_like")
        if arr.dtype.kind not in "uifb":
            return None, FrameHealth(ok=False, reason=f"bad_dtype:{arr.dtype}")
        if arr.size == 0:
            return None, FrameHealth(ok=False, reason="empty_frame")

        hwc = self._to_hwc(arr, repaired)
        if hwc is None:
            return None, FrameHealth(ok=False, reason=f"bad_shape:{tuple(arr.shape)}")
        if hwc.shape[0] < 2 or hwc.shape[1] < 2:
            return None, FrameHealth(ok=False, reason=f"too_small:{hwc.shape[:2]}")

        pixels = self._to_unit_float(hwc, repaired)
        if self.bgr:
            pixels = pixels[:, :, ::-1]
            repaired.append("bgr_to_rgb")

        bad = ~np.isfinite(pixels)
        if bad.any():
            if bad.all():
                return None, FrameHealth(ok=False, reason="all_nonfinite")
            repaired.append(f"nonfinite:{int(bad.sum())}")
            pixels = np.where(bad, 0.0, pixels)
        if pixels.min() < 0.0 or pixels.max() > 1.0:
            repaired.append("clipped")
            pixels = np.clip(pixels, 0.0, 1.0)

        if self.target_size is not None:
            th, tw = self.target_size
            pixels = (letterbox(pixels, th, tw) if self.keep_aspect
                      else resize_nearest(pixels, th, tw))

        pixels = np.ascontiguousarray(pixels, dtype=np.float32)
        mean = float(pixels.mean())
        black = mean < self.black_threshold
        saturated = float((pixels >= 0.999).mean()) > self.saturation_threshold
        frozen = self._update_freeze(pixels)

        self.last_good = pixels
        return pixels, FrameHealth(
            ok=True, black=black, frozen=frozen, saturated=saturated,
            repaired=tuple(repaired),
        )

    # -- freeze detection -------------------------------------------------
    def _update_freeze(self, pixels: np.ndarray) -> bool:
        """Compare a cheap downsampled signature against the previous frame.

        A 16x16 gray signature is ~1000x cheaper than comparing full frames and
        is still sensitive enough: any real gameplay motion moves it well above
        a zero tolerance.
        """
        sig = resize_nearest(pixels, 16, 16).mean(axis=2)
        if self._signature is not None:
            delta = float(np.abs(sig - self._signature).mean())
            self._repeats = self._repeats + 1 if delta <= self.freeze_tolerance else 0
        self._signature = sig
        return self._repeats >= self.freeze_patience

    @property
    def repeats(self) -> int:
        """Consecutive identical frames seen so far (diagnostics)."""
        return self._repeats
