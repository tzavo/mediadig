"""Lazy RapidOCR (PP-OCRv6 medium) wrapper running on CUDA."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image

from mediadig.sizing import fit_down

MAX_SIDE = 1600
MIN_SCORE = 0.5


class OcrInitError(RuntimeError):
    """The OCR engine could not be built or is not running on CUDA (fatal)."""


class _Lang(str, Enum):
    # RapidOCR's LangDet/LangRec enums lack the PP-OCRv6 multilingual codes; any Enum passes
    # its param check and normalize_lang() routes "cs" to the multi_PP-OCRv6_* models.
    CS = "cs"


class Ocr:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._engine = None

    def _build(self):
        try:
            import onnxruntime
            from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR

            onnxruntime.preload_dlls()
            self.models_dir.mkdir(parents=True, exist_ok=True)
            engine = RapidOCR(params={
                "Global.use_cls": False, "Global.model_root_dir": str(self.models_dir),
                "Global.log_level": "error",
                "Det.engine_type": EngineType.ONNXRUNTIME, "Det.ocr_version": OCRVersion.PPOCRV6,
                "Det.lang_type": _Lang.CS, "Det.model_type": ModelType.MEDIUM,
                "Rec.engine_type": EngineType.ONNXRUNTIME, "Rec.ocr_version": OCRVersion.PPOCRV6,
                "Rec.lang_type": _Lang.CS, "Rec.model_type": ModelType.MEDIUM,
                "EngineConfig.onnxruntime.use_cuda": True, "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": 0,
            })
            # RapidOCR only logs a warning when CUDA is missing; insist on it here.
            for model in (engine._load_det_model(), engine._load_rec_model()):
                providers = model.session.session.get_providers()
                if not providers or providers[0] != "CUDAExecutionProvider":
                    raise OcrInitError(f"OCR not running on CUDA (providers={providers})")
        except OcrInitError:
            raise
        except Exception as e:
            raise OcrInitError(f"OCR engine failed to build: {e}") from e
        return engine

    def text(self, img: Image.Image) -> str:
        if self._engine is None:
            self._engine = self._build()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = fit_down(img, (MAX_SIDE, MAX_SIDE), Image.Resampling.BICUBIC)
        arr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
        out = self._engine(arr)
        if out.txts is None:
            return ""
        return "\n".join(t for t, s in zip(out.txts, out.scores) if s >= MIN_SCORE)
