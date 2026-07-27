"""Stratifiziertes, ereignisgetriebenes Sampling (Spec §4).

N=18 Samples: 6 uniform+Jitter, 6 Onset-/RMS-Peaks, 3 Quiet-Fenster,
3 Quote-Frames. Seed aus dem Audio-Content-Hash — deterministisch
reproduzierbar, aber nicht beat-phasenverriegelt (1-Frame/s würde
gegen das Beat-Grid aliasen, z.B. 120 BPM = 2 Hz).
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np

DEFAULT_N = 18
VERIFY_EXTRAS = 6


@dataclass
class SamplePlan:
    """Sample-Zeitpunkte inkl. Stratifikation und Seed (Provenance)."""

    timestamps: list[float]
    seed: str
    categories: dict[str, list[float]] = field(default_factory=dict)
    n: int = 0


def _seed_from_features(features_dict: dict) -> str:
    """Deterministischer Seed aus dem Audio-Content."""
    h = hashlib.sha256()
    h.update(np.asarray(features_dict["rms"]).tobytes())
    h.update(str(features_dict["duration"]).encode())
    return h.hexdigest()


def _rng(seed: str, salt: str = "") -> np.random.Generator:
    # hash() wäre pro Prozess randomisiert (PYTHONHASHSEED) — sha256 ist stabil
    salt_int = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], "big")
    return np.random.default_rng(int(seed[:16], 16) ^ salt_int)


def _uniform_jitter(duration: float, k: int, rng: np.random.Generator) -> list[float]:
    """k gleichverteilte Punkte mit Jitter ±0.5·Intervall (Spec §4)."""
    interval = duration / k
    points = []
    for i in range(k):
        center = (i + 0.5) * interval
        jitter = float(rng.uniform(-0.5, 0.5)) * interval
        points.append(min(max(center + jitter, 0.0), duration))
    return points


def _top_k_with_separation(values: np.ndarray, fps: float, k: int,
                           min_dist_s: float = 0.5,
                           exclude: list[float] | None = None) -> list[float]:
    """Top-k Maxima mit Mindestabstand (gegen Peak-Clustering).

    `exclude`: bereits vergebene Zeitpunkte anderer Strata — Kandidaten
    müssen auch zu diesen min_dist_s Abstand halten, damit die Kategorien
    disjunkt bleiben und n nicht durch Dedup schrumpft.
    """
    exclude = exclude or []
    order = np.argsort(values)[::-1]
    picked: list[float] = []
    for idx in order:
        t = float(idx) / fps
        if all(abs(t - p) >= min_dist_s for p in picked + exclude):
            picked.append(t)
            if len(picked) == k:
                break
    return picked


def _quiet_windows(rms: np.ndarray, fps: float, k: int,
                   window_s: float = 1.0) -> list[float]:
    """k Fenster minimaler RMS-Energie (Worst-Case Vitalität/Blackframe)."""
    win = max(1, int(window_s * fps))
    if len(rms) < win:
        return [0.0]
    energy = np.convolve(rms, np.ones(win) / win, mode="valid")
    return _top_k_with_separation(-energy, fps, k)


def build_sample_plan(
    features_dict: dict,
    n: int = DEFAULT_N,
    quote_times: list[tuple[float, float]] | None = None,
) -> SamplePlan:
    """Erzeugt den stratifizierten Sample-Plan (Spec §4).

    Bei Audio kürzer als der Sample-Bedarf wird die Anzahl adaptiv
    reduziert (Spec §14) — das tatsächliche n steht im Ergebnis.
    """
    duration = float(features_dict["duration"])
    fps = float(features_dict["fps"])
    rms = np.asarray(features_dict["rms"])
    onset = np.asarray(features_dict["onset"])
    seed = _seed_from_features(features_dict)
    rng = _rng(seed)

    # Stratifikation; bei Kurz-Audio anteilig reduzieren (mind. 1 je Kategorie
    # nur wenn die Dauer es hergibt)
    scale = min(1.0, duration / 18.0)
    k_uniform = max(1, round(6 * scale))
    k_peaks = max(1, round(6 * scale))
    k_quiet = max(1, round(3 * scale))
    k_quotes = max(1, round(3 * scale))

    categories: dict[str, list[float]] = {}
    categories["uniform"] = _uniform_jitter(duration, k_uniform, rng)
    categories["peaks"] = _top_k_with_separation(onset, fps, k_peaks)
    categories["quiet"] = _quiet_windows(rms, fps, k_quiet)

    if quote_times:
        quotes = []
        for start, end in quote_times[: k_quotes]:
            quotes.append(float(rng.uniform(start, max(start, end))))
        # Bei weniger Quote-Slots als k_quotes: mit Peaks auffüllen
        while len(quotes) < k_quotes:
            quotes.append(float(rng.uniform(0.0, duration)))
        categories["quotes"] = quotes
    else:
        # Ohne Quotes: auf Peaks auffüllen (Spec §4) — disjunkt zu den
        # bereits gewählten Onset-Peaks, sonst kollidieren beide Strata
        # bei rms == onset Peaks und n schrumpft durch Dedup.
        categories["quotes"] = _top_k_with_separation(
            rms, fps, k_quotes, min_dist_s=1.0,
            exclude=categories["peaks"],
        )

    timestamps = sorted({round(t, 3) for cat in categories.values() for t in cat})
    return SamplePlan(timestamps=timestamps, seed=seed,
                      categories=categories, n=len(timestamps))


def verification_extras(plan: SamplePlan, duration: float,
                        k: int = VERIFY_EXTRAS) -> list[float]:
    """k zusätzliche Kontrollpunkte für die Verify-Phase (Spec §4).

    Disjunkt zu den Probe-Punkten — Overfitting-Kontrolle: der Solver
    darf nicht nur die Zeitpunkte fixen, die er sieht.
    """
    rng = _rng(plan.seed, salt="verify")
    extras: list[float] = []
    while len(extras) < k:
        t = round(float(rng.uniform(0.0, duration)), 3)
        if t not in plan.timestamps and t not in extras:
            extras.append(t)
    return sorted(extras)
