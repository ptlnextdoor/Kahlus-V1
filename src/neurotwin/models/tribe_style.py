from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


VALID_EVENT_SUFFIXES = {
    "text_path": {".txt"},
    "audio_path": {".wav", ".mp3", ".flac", ".ogg"},
    "video_path": {".mp4", ".avi", ".mkv", ".mov", ".webm"},
}


class TribeStyleStimulusEncoder(nn.Module):
    """Toy NeuroTwin-native stimulus-to-fMRI encoder for task plumbing."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("TribeStyleStimulusEncoder expects [batch, time, features]")
        encoded, _ = self.temporal(self.project(x.float()))
        return self.head(encoded)


@dataclass(frozen=True)
class TribeStyleSegment:
    event_type: str
    start: float
    duration: float
    timeline: str
    subject: str

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.event_type,
            "start": self.start,
            "duration": self.duration,
            "timeline": self.timeline,
            "subject": self.subject,
        }


class TribeStyleModel:
    """Small TRIBE-compatible facade implemented entirely with NeuroTwin code.

    This is a local toy clean-room approximation for stimulus-to-fMRI
    benchmark plumbing. It does not load TRIBE v2 code, configs, pretrained
    weights, or real video/audio/text encoders.
    """

    model_id = "tribe_style"
    implementation_status = "clean_room_approximation"

    def __init__(
        self,
        stimulus_dim: int = 12,
        output_dim: int = 5,
        hidden_dim: int = 32,
        seed: int = 0,
        device: str = "cpu",
        cache_folder: str | Path | None = None,
    ) -> None:
        self.stimulus_dim = int(stimulus_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        self.cache_folder = str(cache_folder) if cache_folder is not None else None
        if self.cache_folder is not None:
            Path(self.cache_folder).mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        self._model = TribeStyleStimulusEncoder(self.stimulus_dim, self.output_dim, self.hidden_dim).to(self.device)
        self._model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path | None = None,
        checkpoint_name: str = "tribe_style_config.json",
        cache_folder: str | Path | None = None,
        device: str = "auto",
        config_update: dict[str, Any] | None = None,
    ) -> "TribeStyleModel":
        """Load a local NeuroTwin TRIBE-style config or seeded defaults.

        No pretrained TRIBE v2 weights are loaded and no external downloads are
        attempted. The model is a small local baseline for task plumbing until
        real prepared stimulus feature arrays are supplied by the data pipeline.
        """

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        config = {
            "stimulus_dim": 12,
            "output_dim": 5,
            "hidden_dim": 32,
            "seed": 0,
        }
        if checkpoint_dir not in (None, "", "local", "neurotwin/tribe-style"):
            config.update(_load_local_config(checkpoint_dir, checkpoint_name))
        if config_update is not None:
            config.update(config_update)
        return cls(
            stimulus_dim=int(config.get("stimulus_dim", 12)),
            output_dim=int(config.get("output_dim", 5)),
            hidden_dim=int(config.get("hidden_dim", 32)),
            seed=int(config.get("seed", 0)),
            device=str(device),
            cache_folder=cache_folder,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path | None = None,
        checkpoint_name: str = "tribe_style_config.json",
        cache_folder: str | Path | None = None,
        device: str = "auto",
        config_update: dict[str, Any] | None = None,
    ) -> "TribeStyleModel":
        """Compatibility shim for TRIBE-style callers.

        Prefer :meth:`from_checkpoint`. This method never loads TRIBE v2
        pretrained weights and never downloads from HuggingFace or other
        external services.
        """

        return cls.from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            cache_folder=cache_folder,
            device=device,
            config_update=config_update,
        )

    def build_events(
        self,
        text_path: str | None = None,
        audio_path: str | None = None,
        video_path: str | None = None,
    ) -> list[dict[str, object]]:
        """Build minimal local event rows for smoke/pipeline tests.

        Text events are tokenized by whitespace. Audio/video events only record
        file metadata. Downstream predictions from these rows use deterministic
        hash-derived embeddings; they are not real video/audio/text features.
        Real stimulus-fMRI evaluation should use prepared ``stimulus_embedding``
        arrays instead.
        """

        return _build_events(text_path=text_path, audio_path=audio_path, video_path=video_path)

    def get_events_dataframe(
        self,
        text_path: str | None = None,
        audio_path: str | None = None,
        video_path: str | None = None,
    ) -> list[dict[str, object]]:
        """Compatibility shim returning local event rows, not a pandas DataFrame.

        Prefer :meth:`build_events`. This method preserves the broad TRIBE-style
        API shape for smoke tests while staying dependency-light.
        """

        return self.build_events(text_path=text_path, audio_path=audio_path, video_path=video_path)

    def predict(self, events: Iterable[dict[str, Any]], verbose: bool = True) -> tuple[np.ndarray, list[dict[str, object]]]:
        """Predict toy fMRI responses from event rows.

        Event rows are converted to deterministic hash-derived embeddings unless
        real features were already prepared elsewhere. These predictions are for
        local plumbing tests only.
        """

        del verbose
        event_rows = [dict(event) for event in events]
        if not event_rows:
            raise ValueError("events must contain at least one row")
        stimulus = np.asarray([_event_embedding(event, self.stimulus_dim) for event in event_rows], dtype=np.float32)
        x = torch.as_tensor(stimulus[None, :, :], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            prediction = self._model(x).detach().cpu().numpy()[0]
        segments = [
            TribeStyleSegment(
                event_type=str(event.get("type", "Event")),
                start=float(event.get("start", idx)),
                duration=float(event.get("duration", 1.0)),
                timeline=str(event.get("timeline", "default")),
                subject=str(event.get("subject", "average")),
            ).to_dict()
            for idx, event in enumerate(event_rows)
        ]
        return prediction.astype(np.float32), segments


def _build_events(
    text_path: str | None = None,
    audio_path: str | None = None,
    video_path: str | None = None,
) -> list[dict[str, object]]:
    provided = {
        name: value
        for name, value in (
            ("text_path", text_path),
            ("audio_path", audio_path),
            ("video_path", video_path),
        )
        if value is not None
    }
    if len(provided) != 1:
        raise ValueError("Exactly one of text_path, audio_path, video_path must be provided")

    name, value = next(iter(provided.items()))
    path = Path(value)
    if path.suffix.lower() not in VALID_EVENT_SUFFIXES[name]:
        raise ValueError(f"{name} must end with one of {sorted(VALID_EVENT_SUFFIXES[name])}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")

    if name == "text_path":
        text = path.read_text(encoding="utf-8")
        tokens = [token for token in text.replace("\n", " ").split(" ") if token]
        if not tokens:
            raise ValueError(f"Text file is empty: {path}")
        return [
            {
                "type": "Word",
                "text": token,
                "start": float(idx),
                "duration": 1.0,
                "timeline": "default",
                "subject": "average",
            }
            for idx, token in enumerate(tokens)
        ]

    event_type = "Audio" if name == "audio_path" else "Video"
    return [
        {
            "type": event_type,
            "filepath": str(path),
            "start": 0.0,
            "duration": 1.0,
            "timeline": "default",
            "subject": "average",
        }
    ]


def _load_local_config(checkpoint_dir: str | Path, checkpoint_name: str) -> dict[str, Any]:
    root = Path(checkpoint_dir)
    if root.is_file():
        config_path = root
    elif root.is_dir():
        candidates = (root / checkpoint_name, root / "config.json", root / "tribe_style_config.json")
        config_path = next((path for path in candidates if path.is_file()), None)
        if config_path is None:
            return {}
    else:
        raise FileNotFoundError(
            "TribeStyleModel only loads local NeuroTwin configs; "
            f"no external download is attempted for {checkpoint_dir!r}"
        )
    return dict(json.loads(config_path.read_text(encoding="utf-8")))


def _event_embedding(event: dict[str, Any], dim: int) -> np.ndarray:
    token = "|".join(
        str(event.get(key, ""))
        for key in ("type", "text", "filepath", "start", "duration", "timeline", "subject")
    )
    seed = int.from_bytes(sha256(token.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    return rng.normal(size=dim).astype(np.float32)
