# Copyright 2026 Canonical Ltd.
#
# SPDX-License-Identifier: LGPL-3.0-only

"""Fixtures shared across the unit suites."""

import collections.abc
import contextlib
import os
import pathlib
import typing

import pytest

from charmlibs.seceng import utils


@pytest.fixture
def writable_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the secure writer create files under a test directory.

    ``utils.open_file_secure`` refuses to traverse a directory owned by a
    non-root user, which every pytest temporary directory is, so the stamp
    writer needs its file primitive replaced even though the stamp logic itself
    is exercised for real.
    """

    @contextlib.contextmanager
    def fake_open_file_secure(
        path: pathlib.Path,
        *,
        user: str | None = None,
        group: str | None = None,
        mode: int = 0o600,
        create_parents: bool = False,
        text: bool = True,
    ) -> collections.abc.Iterator[typing.TextIO | typing.BinaryIO]:
        path.parent.mkdir(parents=True, exist_ok=True)
        file: typing.TextIO | typing.BinaryIO
        if text:
            file = path.open('w', encoding='utf-8')
        else:
            file = path.open('wb')
        with file:
            yield file
        os.chmod(path, mode)

    monkeypatch.setattr(utils, 'open_file_secure', fake_open_file_secure)
