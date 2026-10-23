"""Timeout output is retained only for paid picture recovery."""

import subprocess

import pytest

from yap_studio import process
from yap_studio.errors import ProcessingError


@pytest.mark.parametrize("tool", ["wavespeed-pp-cli", "git"])
@pytest.mark.parametrize("stderr", [b"prediction known123 was submitted\n\xff", "submitted", None])
def test_timeout_stderr_is_retained_only_for_wavespeed(monkeypatch, tool, stderr):
    def expired(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=b"{}", stderr=stderr)

    monkeypatch.setattr(process.subprocess, "run", expired)
    with pytest.raises(ProcessingError) as caught:
        process.spawn(["synthetic"], timeout_s=1, tool=tool)
    error = caught.value
    assert error.code == f"{tool}_timeout" and error.returncode == -1
    expected = "timed out after 1 s"
    if tool == "wavespeed-pp-cli":
        if stderr:
            detail = (stderr.decode("utf-8", errors="replace")
                      if isinstance(stderr, bytes) else stderr)
            expected += "\n" + detail
        assert error.stdout == b"{}"
    else:
        assert error.stdout is None
    assert error.full_stderr == expected
