"""Deadline-aware raw frame decoding and refined Vision encoding."""

from __future__ import annotations

import os
import selectors
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import closing
from itertools import zip_longest
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock
from typing import BinaryIO

import numpy as np

from yap_studio import ffmpeg
from yap_studio.errors import ProcessingError
from yap_studio.faces.model import VideoIdentity
from yap_studio.matte import vision
from yap_studio.matte.model import Cleanup


def gray_decode_command(video: Path, *, alpha: bool = False, webm: bool = False) -> list[str]:
    """Keep the shared gray decode and the fallback's alpha decoder selection."""
    command = ["ffmpeg", "-v", "error"]
    if webm:
        # The default VP9 decoder silently discards the auxiliary alpha stream.
        command += ["-c:v", "libvpx-vp9"]
    return [*command, "-i", str(video), "-map", "0:v:0", "-an", "-vf",
            "alphaextract" if alpha else "format=gray", "-fps_mode", "passthrough",
            "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"]


def read_frames(pipe: BinaryIO, width: int, height: int, deadline: float, *,
                planes: int = 1, check: Callable[[], None] = lambda: None
                ) -> Iterator[np.ndarray]:
    """Read complete planar frames, checking child health while a pipe is idle."""
    with selectors.DefaultSelector() as selector:
        selector.register(pipe, selectors.EVENT_READ)
        while True:
            frame = np.empty((planes, height, width), dtype=np.uint8)
            target = memoryview(frame).cast("B")
            used = 0
            while used < len(target):
                check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired("matte frames", 0)
                if not selector.select(min(remaining, 0.05)):
                    continue
                chunk = os.read(pipe.fileno(), min(len(target) - used, 256 * 1024))
                if not chunk:
                    if used:
                        raise ProcessingError("ffmpeg", 0, "partial decoded matte frame",
                                              code="matte_frame_count")
                    return
                target[used:used + len(chunk)] = chunk
                used += len(chunk)
            yield frame[0] if planes == 1 else np.moveaxis(frame, 0, -1)


def decode_frames(video: Path, width: int, height: int, deadline: float, *,
                  first: int = 0, count: int | None = None, planes: int = 1,
                  commands: list[tuple[str, ...]] | None = None,
                  check: Callable[[], None] = lambda: None) -> Iterator[np.ndarray]:
    """Decode an exact window, preserving decode order and full-resolution planes."""
    from yap_studio.matte.make import _stderr

    trim = f"setpts=N,trim=start_frame={first}"
    if count is not None:
        trim += f":end_frame={first + count}"
    command = gray_decode_command(video)
    filter_index = command.index("-vf")
    command[filter_index:filter_index + 2] = ["-filter:v", trim + ",format="
                                           + ("gray" if planes == 1 else "yuv444p")]
    command[command.index("-pix_fmt") + 1] = "gray" if planes == 1 else "yuv444p"
    command = ffmpeg._safe_command(command)
    if commands is not None:
        commands.append(tuple(command))
    with tempfile.TemporaryFile() as stderr:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=stderr, bufsize=0)
        try:
            assert child.stdout is not None
            yield from read_frames(child.stdout, width, height, deadline,
                                   planes=planes, check=check)
            child.wait(timeout=max(0, deadline - time.monotonic()))
            if child.returncode:
                raise ProcessingError(" ".join(command), child.returncode, _stderr(stderr))
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()
            child.stdout.close()


def encode_vision(stream: vision.HelperStream, source: Path, temporary: Path,
                  video: VideoIdentity, cleanup: Cleanup, deadline: float,
                  joins: Sequence[int]) -> None:
    """Overlap Vision with bounded refinement and reap every child on every exit."""
    from yap_studio.matte.make import _encoder_command, _stderr
    from yap_studio.matte.refine import refine_pairs

    command = _encoder_command(temporary, video, cleanup, refined=True)
    helper = stream.process
    assert helper.stdout is not None
    with tempfile.TemporaryFile() as stderr:
        try:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=stderr, bufsize=0)
        except OSError as exc:
            raise ProcessingError(" ".join(command), -1, str(exc)) from exc
        stopped, diagnostics = Event(), Lock()
        incoming, outgoing = Queue(maxsize=1), Queue(maxsize=1)
        tasks = []

        def check() -> None:
            if stopped.is_set():
                raise CancelledError
            if helper.poll() not in (None, 0):
                with diagnostics:
                    raise ProcessingError(" ".join(stream.command), helper.returncode,
                        _stderr(stream.stderr), code=f"person_matte_exit_{helper.returncode}")
            if encoder.poll() not in (None, 0):
                with diagnostics:
                    raise ProcessingError(" ".join(command), encoder.returncode, _stderr(stderr))

        def main_check() -> None:
            check()
            for task in tasks:
                if task.done():
                    task.result()

        def remaining() -> float:
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise subprocess.TimeoutExpired(command, 0)
            return min(seconds, 0.05)

        def put(queue, value, monitor) -> None:
            while True:
                monitor()
                try:
                    queue.put(value, timeout=remaining())
                    return
                except Full:
                    continue

        def take(queue, monitor):
            while True:
                monitor()
                try:
                    return queue.get(timeout=remaining())
                except Empty:
                    continue

        def finish(child: subprocess.Popen[bytes]) -> None:
            while child.poll() is None:
                check()
                timeout = remaining()
                try:
                    child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    continue
            check()

        def mask_frames() -> Iterator[np.ndarray]:
            try:
                yield from read_frames(helper.stdout, video.width, video.height,
                                       deadline, check=check)
            except ProcessingError:
                finish(helper)
                raise
            # EOF may arrive before the helper has written its final diagnostic.
            finish(helper)

        assert encoder.stdin is not None
        def read() -> None:
            guide_frames = decode_frames(source, video.width, video.height, deadline, check=check)
            with closing(guide_frames) as guides, closing(mask_frames()) as masks:
                for alpha, guide in zip_longest(masks, guides):
                    if alpha is None or guide is None:
                        raise ProcessingError("ffmpeg", 0, "matte and guide frame counts differ",
                                              code="matte_frame_count")
                    put(incoming, (alpha, guide), check)
            put(incoming, None, check)

        def pairs() -> Iterator[tuple[np.ndarray, np.ndarray]]:
            while True:
                pair = take(incoming, main_check)
                if pair is None:
                    return
                yield pair

        def write() -> None:
            os.set_blocking(encoder.stdin.fileno(), False)
            with selectors.DefaultSelector() as ready:
                ready.register(encoder.stdin, selectors.EVENT_WRITE)
                while True:
                    alpha = take(outgoing, check)
                    if alpha is None:
                        break
                    raw = memoryview(alpha).cast("B")
                    while raw:
                        check()
                        if ready.select(remaining()):
                            try:
                                size = os.write(encoder.stdin.fileno(), raw)
                            except BlockingIOError:
                                continue
                            raw = raw[size:]
            encoder.stdin.close()

        pool = ThreadPoolExecutor(max_workers=2)
        try:
            # Each queue holds one frame. Read, refine and write overlap without
            # growing the seven-frame raw window or changing output order.
            tasks.extend([pool.submit(read), pool.submit(write)])
            count = 0
            for alpha in refine_pairs(pairs(), cleanup=cleanup, joins=joins):
                put(outgoing, alpha, main_check)
                count += 1
            if count != video.frames:
                raise ProcessingError("ffmpeg", 0,
                    f"expected {video.frames} matte frames, received {count}",
                    code="matte_frame_count")
            put(outgoing, None, main_check)
            for task in tasks:
                task.result(timeout=max(0, deadline - time.monotonic()))
            finish(helper)
            finish(encoder)
        except BrokenPipeError:
            encoder.wait(timeout=max(0, deadline - time.monotonic()))
            check()
            raise ProcessingError(" ".join(command), encoder.returncode, _stderr(stderr)) from None
        except (subprocess.TimeoutExpired, TimeoutError):
            raise ProcessingError(" ".join(stream.command), -1, "person matte pipeline timed out",
                                  code="person_matte_timeout") from None
        finally:
            stopped.set()
            for child in (helper, encoder):
                if child.poll() is None:
                    child.kill()
            for child in (helper, encoder):
                child.wait()
            pool.shutdown(wait=True, cancel_futures=True)
            encoder.stdin.close()
