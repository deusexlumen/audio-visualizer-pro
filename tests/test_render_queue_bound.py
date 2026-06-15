"""
Test that the render producer blocks on a full frame queue and never
grows unbounded, even when the encoder is artificially slowed down.
"""
import queue
import subprocess
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.gpu_renderer import GPUBatchRenderer


class TestBoundedFrameQueue:
    """Regression test for unbounded frame queue memory spike."""

    def test_render_queue_never_exceeds_maxsize(self, tmp_path):
        """A slow encoder must not let the frame queue grow beyond maxsize."""
        observed_max = [0]
        original_put = queue.Queue.put

        def tracking_put(self, item, block=True, timeout=None):
            # Track actual max size seen during the render
            observed_max[0] = max(observed_max[0], self.qsize())
            return original_put(self, item, block=block, timeout=timeout)

        # Fake FFmpeg process that writes very slowly
        fake_proc = Mock()
        fake_proc.poll.return_value = None
        fake_proc.returncode = 0
        fake_proc.stdin = Mock()

        write_delay = [0.05]

        def slow_write(data):
            time.sleep(write_delay[0])

        fake_proc.stdin.write.side_effect = slow_write

        # Tiny audio features to keep the test fast
        from src.types import AudioFeatures
        frame_count = 120
        features = AudioFeatures(
            duration=frame_count / 30.0,
            sample_rate=44100,
            fps=30,
            frame_count=frame_count,
            rms=np.random.rand(frame_count).astype(np.float32),
            onset=np.random.rand(frame_count).astype(np.float32),
            spectral_centroid=np.random.rand(frame_count).astype(np.float32),
            spectral_rolloff=np.random.rand(frame_count).astype(np.float32),
            zero_crossing_rate=np.random.rand(frame_count).astype(np.float32),
            transient=np.random.rand(frame_count).astype(np.float32),
            voice_clarity=np.random.rand(frame_count).astype(np.float32),
            voice_band=np.random.rand(frame_count).astype(np.float32),
            chroma=np.random.rand(12, frame_count).astype(np.float32),
            mfcc=np.random.rand(13, frame_count).astype(np.float32),
            tempogram=np.random.rand(384, frame_count).astype(np.float32),
            tempo=120.0,
            key="C major",
            mode="music",
            beat_frames=np.array([], dtype=np.int32),
        )

        output_path = str(tmp_path / "bounded_queue.mp4")

        observed_max = [0]

        def make_tracking_put(original_put):
            def tracking_put(self, item, block=True, timeout=None):
                observed_max[0] = max(observed_max[0], self.qsize())
                return original_put(self, item, block=block, timeout=timeout)
            return tracking_put

        fake_mux_result = Mock()
        fake_mux_result.returncode = 0

        with patch.object(queue.Queue, "put", make_tracking_put(queue.Queue.put)):
            renderer = GPUBatchRenderer(width=320, height=180, fps=30)
            with patch("subprocess.Popen", return_value=fake_proc):
                with patch("subprocess.run", return_value=fake_mux_result):
                    renderer.render(
                        audio_path="dummy.mp3",
                        visualizer_type="voice_flow",
                        output_path=output_path,
                        features=features,
                        preview_mode=True,
                        preview_duration=frame_count / 30.0,
                        quality="low",
                    )

        assert observed_max[0] <= 3, (
            f"Queue grew to {observed_max[0]} items, expected <= 3"
        )
