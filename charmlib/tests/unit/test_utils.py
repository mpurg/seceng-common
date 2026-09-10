# Copyright 2026 Canonical Ltd.
#
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for the sanitised subprocess environment helpers."""

from __future__ import annotations

import subprocess
import typing

import pytest

from charmlibs.seceng import utils


def test_clean_env_inherits_proxies_and_drops_python_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HTTP_PROXY', 'http://proxy:3128')
    monkeypatch.setenv('http_proxy', 'http://proxy:3128')
    monkeypatch.setenv('VIRTUAL_ENV', '/home/user/venv')
    monkeypatch.setenv('PYTHONPATH', '/home/user/lib')

    env = utils.clean_env({'VIRTUAL_ENV': '/attacker/venv'})

    assert env['HTTP_PROXY'] == 'http://proxy:3128'
    assert env['http_proxy'] == 'http://proxy:3128'
    assert env['PATH'] == '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
    assert env['HOME'] == '/root'
    assert 'VIRTUAL_ENV' not in env
    assert 'PYTHONPATH' not in env


def test_run_forwards_arguments_to_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, object] = {}

    def fake_subprocess_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        recorded.update(kw)
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(subprocess, 'run', fake_subprocess_run)

    utils.run(['/usr/bin/true'], check=False, capture=True, timeout=123.0)

    assert recorded['check'] is False
    assert recorded['capture_output'] is True
    assert recorded['timeout'] == 123.0
    env = typing.cast(dict[str, str], recorded['env'])
    assert 'PATH' in env
    assert 'VIRTUAL_ENV' not in env
