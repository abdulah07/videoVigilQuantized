"""
system_utils.py
────────────────────────────────────────────────────────────────────────────────
Detect PC specs and recommend optimal models based on system capabilities.
────────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def get_available_trackers() -> List[str]:
    """
    Get list of ALL available trackers.
    Includes both Ultralytics native (ByteTrack, BoTSort) and custom implementations (OCSort, StrongSort, TrackFormer).
    """
    # All trackers: 2 native + 3 custom implementations
    ALL_TRACKERS = ["bytetrack", "botsort", "ocsort", "strongsort", "trackformer"]
    
    available = []
    try:
        # Try importing TRACKER_DIR from ultralytics
        try:
            from ultralytics.cfg import TRACKER_DIR
            tracker_dir = Path(TRACKER_DIR)
        except (ImportError, AttributeError):
            # Fall back to finding the trackers directory manually
            import ultralytics
            uv_path = Path(ultralytics.__file__).parent
            tracker_dir = uv_path / "cfg" / "trackers"
        
        if tracker_dir.exists():
            yaml_files = sorted(list(tracker_dir.glob("*.yaml")))
            native_trackers = [f.stem for f in yaml_files if f.stem in ALL_TRACKERS]
            available = native_trackers[:]
        
        # Always add custom trackers (they're always available)
        for custom in ["ocsort", "strongsort", "trackformer"]:
            if custom not in available:
                available.append(custom)
        
        logger.info("✓ Available trackers: %s", sorted(available))
    except Exception as e:
        logger.warning("Could not detect trackers: %s, using all trackers", str(e))
        available = ALL_TRACKERS
    
    return sorted(available) if available else ALL_TRACKERS


class SystemSpec:
    """Detect and store system hardware specifications."""

    def __init__(self):
        self.os_name = platform.system()
        self.cpu_cores = os.cpu_count() or 4
        self.cpu_freq = self._get_cpu_freq()
        self.total_ram_gb = self._get_total_ram()
        self.available_ram_gb = self._get_available_ram()
        self.has_gpu = self._detect_gpu()
        self.gpu_vram_gb = self._get_gpu_vram()
        self.gpu_name = self._get_gpu_name()

    @staticmethod
    def _get_cpu_freq() -> float:
        """Get CPU frequency in GHz."""
        try:
            import psutil
            freq = psutil.cpu_freq()
            return freq.max / 1000.0 if freq else 2.5
        except (ImportError, Exception):
            return 2.5

    @staticmethod
    def _get_total_ram() -> float:
        """Get total RAM in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except (ImportError, Exception):
            return 8.0

    @staticmethod
    def _get_available_ram() -> float:
        """Get available RAM in GB."""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 ** 3)
        except (ImportError, Exception):
            return 4.0

    @staticmethod
    def _detect_gpu() -> bool:
        """Detect if NVIDIA GPU is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and result.stdout.strip() != ""
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return False

    @staticmethod
    def _get_gpu_vram() -> float:
        """Get GPU VRAM in GB."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                vram_mb = int(result.stdout.strip().split("\n")[0])
                return vram_mb / 1024.0
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired, Exception):
            pass
        return 0.0

    @staticmethod
    def _get_gpu_name() -> str:
        """Get GPU name."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
        return "N/A"

    def summary(self) -> str:
        """Return human-readable spec summary."""
        lines = [
            f"OS: {self.os_name}",
            f"CPU: {self.cpu_cores} cores @ {self.cpu_freq:.1f} GHz",
            f"RAM: {self.total_ram_gb:.1f} GB total ({self.available_ram_gb:.1f} GB available)",
        ]
        if self.has_gpu:
            lines.append(f"GPU: {self.gpu_name} ({self.gpu_vram_gb:.1f} GB VRAM)")
        else:
            lines.append("GPU: None")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return {
            "os_name": self.os_name,
            "cpu_cores": self.cpu_cores,
            "cpu_freq_ghz": self.cpu_freq,
            "total_ram_gb": self.total_ram_gb,
            "available_ram_gb": self.available_ram_gb,
            "has_gpu": self.has_gpu,
            "gpu_vram_gb": self.gpu_vram_gb,
            "gpu_name": self.gpu_name,
        }


class ModelRecommender:
    """Recommend optimal models based on system specs."""

    # Model profiles: (model_name, min_ram_gb, min_vram_gb, min_cpu_cores, requires_gpu, speed_rank, accuracy_rank)
    REID_PROFILES = {
        "osnet_x0_25": (0.5, 0, 2, False, 1, 1),  # Fastest, lightest
        "osnet_x1_0": (1.0, 0.5, 2, False, 2, 2),
        "osnet_ain_x1_0": (1.5, 1.0, 2, False, 3, 3),  # Balanced
        "resnet50": (2.0, 1.5, 4, False, 4, 4),
        "resnet101": (3.0, 2.0, 4, False, 5, 5),
        "transreid": (2.5, 2.0, 4, True, 3, 5),  # Better accuracy, needs GPU
        "vitreid": (3.5, 3.0, 4, True, 4, 6),  # Best accuracy, GPU recommended
    }

    TRACKER_PROFILES = {
        "bytetrack": (0.2, 0, 2, False, 1, 4),  # Fastest
        "botsort": (0.3, 0, 2, False, 2, 5),
        "ocsort": (0.25, 0, 2, False, 2, 5),
        "strongsort": (0.4, 0, 4, False, 3, 6),  # Better accuracy
        "trackformer": (1.0, 1.0, 4, True, 4, 7),  # Best accuracy, GPU needed
    }

    @staticmethod
    def recommend(spec: SystemSpec, prioritize: str = "balanced") -> Dict[str, List[str]]:
        """
        Recommend models based on specs.

        Args:
            spec: SystemSpec object
            prioritize: 'speed' | 'accuracy' | 'balanced' (default)

        Returns:
            {'reid': [models], 'tracker': [models]}
        """
        reid_candidates = ModelRecommender._filter_models(
            ModelRecommender.REID_PROFILES, spec, prioritize
        )
        
        # Only use available trackers
        available_trackers = get_available_trackers()
        filtered_tracker_profiles = {
            k: v for k, v in ModelRecommender.TRACKER_PROFILES.items()
            if k in available_trackers
        }
        tracker_candidates = ModelRecommender._filter_models(
            filtered_tracker_profiles, spec, prioritize
        )

        return {
            "reid": reid_candidates,
            "tracker": tracker_candidates,
        }

    @staticmethod
    def _filter_models(
        profiles: Dict[str, Tuple],
        spec: SystemSpec,
        prioritize: str = "balanced",
    ) -> List[str]:
        """Filter available models based on system specs."""
        valid = []

        for model_name, (min_ram, min_vram, min_cores, requires_gpu, speed, accuracy) in profiles.items():
            # Check constraints
            if spec.total_ram_gb < min_ram:
                continue
            if requires_gpu and not spec.has_gpu:
                continue
            if spec.has_gpu and spec.gpu_vram_gb < min_vram:
                continue
            if spec.cpu_cores < min_cores:
                continue

            # Calculate score based on prioritization
            if prioritize == "speed":
                score = -speed  # Negative so lower is better
            elif prioritize == "accuracy":
                score = -accuracy
            else:  # balanced
                score = -(speed * 0.5 + accuracy * 0.5)

            valid.append((model_name, score))

        # Sort by score (best first) and return names
        valid.sort(key=lambda x: x[1])
        return [name for name, _ in valid]

    @staticmethod
    def recommendation_summary(spec: SystemSpec, recommendations: Dict) -> str:
        """Generate human-readable recommendation summary."""
        lines = [
            "═" * 60,
            "SYSTEM SPECIFICATIONS & RECOMMENDATIONS",
            "═" * 60,
            "",
            spec.summary(),
            "",
            "─" * 60,
            "RECOMMENDED MODELS (Best → Good):",
            "─" * 60,
        ]

        if recommendations["reid"]:
            lines.append(f"Re-ID: {' → '.join(recommendations['reid'][:3])}")
        else:
            lines.append("Re-ID: (no compatible models)")

        if recommendations["tracker"]:
            lines.append(f"Tracker: {' → '.join(recommendations['tracker'][:3])}")
        else:
            lines.append("Tracker: (no compatible models)")

        lines.append("═" * 60)
        return "\n".join(lines)


def get_system_spec() -> SystemSpec:
    """Get system specifications (cached on first call)."""
    global _CACHED_SPEC
    if "_CACHED_SPEC" not in globals():
        _CACHED_SPEC = SystemSpec()
    return _CACHED_SPEC


def recommend_models(
    prioritize: str = "balanced",
) -> Tuple[SystemSpec, Dict[str, List[str]]]:
    """
    Detect system specs and recommend optimal models.

    Args:
        prioritize: 'speed' | 'accuracy' | 'balanced'

    Returns:
        (SystemSpec, recommendations_dict)
    """
    spec = get_system_spec()
    recommendations = ModelRecommender.recommend(spec, prioritize)
    logger.info("System specs detected: %s", spec.to_dict())
    logger.info("Model recommendations: %s", recommendations)
    return spec, recommendations
