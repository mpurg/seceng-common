# Copyright 2026 Canonical Ltd.
#
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for charmlibs.seceng.workload primitives and workflow."""

from __future__ import annotations

import io
import os
import pathlib
import subprocess
import tarfile
import typing

import pytest

from charmlibs.seceng import utils
from charmlibs.seceng.workload import (
    ArtifactExtractionError,
    WorkloadError,
    WorkloadInstallError,
    flip_symlink,
    get_active_version,
    install_and_activate_wheelhouse,
    is_version_installed,
    prune_versions,
    unpack_archive,
    validate_version,
)


# ============================================================================
# Layer 1: Version Validation
# ============================================================================


def test_validate_version_valid_tags() -> None:
    assert validate_version('1.0.0') == '1.0.0'
    assert validate_version('v1.2.3_rc1-beta') == 'v1.2.3_rc1-beta'
    assert validate_version('0.1.0') == '0.1.0'


@pytest.mark.parametrize(
    'invalid',
    [
        '',
        '-1.0.0',
        '../1.0.0',
        '1.0.0/../../etc',
        '1.0;rm -rf /',
        '1.0?foo=bar',
        '1.0#frag',
        '1.0 2.0',
        '1.0\n2.0',
    ],
)
def test_validate_version_rejects_unsafe(invalid: str) -> None:
    with pytest.raises(WorkloadError):
        validate_version(invalid)


# ============================================================================
# Layer 1: Tarball Extraction Hardening (unpack_archive)
# ============================================================================


def _create_tar(path: pathlib.Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, 'w:gz') as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 1000
            tar.addfile(info, io.BytesIO(data))


def test_unpack_archive_strips_single_top_level_dir(tmp_path: pathlib.Path) -> None:
    archive = tmp_path / 'bundle.tar.gz'
    _create_tar(
        archive,
        {
            'bundle/file1.txt': b'hello',
            'bundle/subdir/file2.txt': b'world',
        },
    )
    dest = tmp_path / 'output'
    unpack_archive(archive, dest)

    assert (dest / 'file1.txt').read_bytes() == b'hello'
    assert (dest / 'subdir' / 'file2.txt').read_bytes() == b'world'
    assert not (dest / 'bundle').exists()


def test_unpack_archive_preserves_archive_without_single_root(tmp_path: pathlib.Path) -> None:
    archive = tmp_path / 'multi.tar.gz'
    _create_tar(
        archive,
        {
            'file1.txt': b'hello',
            'file2.txt': b'world',
        },
    )
    dest = tmp_path / 'output'
    unpack_archive(archive, dest)

    assert (dest / 'file1.txt').read_bytes() == b'hello'
    assert (dest / 'file2.txt').read_bytes() == b'world'


def test_unpack_archive_is_non_destructive_to_input_archive(tmp_path: pathlib.Path) -> None:
    archive = tmp_path / 'keep.tar.gz'
    _create_tar(archive, {'wheelhouse/pkg.whl': b'fake-wheel'})
    initial_bytes = archive.read_bytes()

    dest = tmp_path / 'dest'
    unpack_archive(archive, dest)

    assert archive.exists()
    assert archive.read_bytes() == initial_bytes


def test_unpack_archive_replaces_existing_dest_cleanly(tmp_path: pathlib.Path) -> None:
    dest = tmp_path / 'dest'
    dest.mkdir()
    (dest / 'old.txt').write_text('old content')

    archive = tmp_path / 'new.tar.gz'
    _create_tar(archive, {'root/new.txt': b'new content'})

    unpack_archive(archive, dest)
    assert (dest / 'new.txt').read_bytes() == b'new content'
    assert not (dest / 'old.txt').exists()


def test_unpack_archive_rejects_path_traversal(tmp_path: pathlib.Path) -> None:
    archive = tmp_path / 'evil.tar.gz'
    _create_tar(archive, {'bundle/../../evil.txt': b'bad'})
    dest = tmp_path / 'dest'

    with pytest.raises(ArtifactExtractionError, match='unsafe member path'):
        unpack_archive(archive, dest)


def test_unpack_archive_rejects_symlink_with_filter_data(tmp_path: pathlib.Path) -> None:
    archive = tmp_path / 'symlink.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        info = tarfile.TarInfo(name='bundle/link')
        info.type = tarfile.SYMTYPE
        info.linkname = '/etc/passwd'
        tar.addfile(info)

    dest = tmp_path / 'dest'
    with pytest.raises(ArtifactExtractionError):
        unpack_archive(archive, dest)


def test_unpack_archive_restores_previous_dest_on_swap_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / 'dest'
    dest.mkdir()
    (dest / 'old.txt').write_text('old content')

    archive = tmp_path / 'new.tar.gz'
    _create_tar(archive, {'root/new.txt': b'new content'})

    real_replace = os.replace
    swap_attempts: list[str] = []

    def failing_replace(src: typing.Any, dst: typing.Any, **kw: typing.Any) -> None:
        if pathlib.Path(str(dst)) == dest:
            swap_attempts.append(str(src))
            if len(swap_attempts) == 1:
                raise OSError('simulated swap failure')
        return real_replace(src, dst, **kw)

    monkeypatch.setattr(os, 'replace', failing_replace)

    with pytest.raises(ArtifactExtractionError, match='Failed to atomically swap'):
        unpack_archive(archive, dest)

    assert (dest / 'old.txt').read_text() == 'old content'
    assert not (dest / 'new.txt').exists()
    # Neither the staging directory nor the swap backup leaks.
    assert [entry.name for entry in tmp_path.iterdir() if entry.name.startswith('.')] == []


# ============================================================================
# Layer 1: Symlink and Pruning Helpers
# ============================================================================


def test_flip_symlink_atomic(tmp_path: pathlib.Path) -> None:
    link = tmp_path / 'current'
    target_v1 = tmp_path / 'venvs' / '1.0.0'
    target_v1.mkdir(parents=True)
    target_v2 = tmp_path / 'venvs' / '2.0.0'
    target_v2.mkdir(parents=True)

    flip_symlink(link, target_v1)
    assert link.is_symlink()
    assert os.readlink(link) == 'venvs/1.0.0'

    flip_symlink(link, target_v2)
    assert os.readlink(link) == 'venvs/2.0.0'


def test_flip_symlink_refuses_regular_file(tmp_path: pathlib.Path) -> None:
    link = tmp_path / 'current'
    link.write_text('regular file')
    target = tmp_path / 'venvs' / '1.0.0'

    with pytest.raises(WorkloadError, match='is not a symlink'):
        flip_symlink(link, target)


def test_prune_versions_preserves_active(tmp_path: pathlib.Path) -> None:
    venvs = tmp_path / 'venvs'
    venvs.mkdir()

    v1 = venvs / '1.0.0'
    v2 = venvs / '2.0.0'
    v3 = venvs / '3.0.0'
    v4 = venvs / '4.0.0'

    for v, mtime in [(v1, 100), (v2, 200), (v3, 300), (v4, 400)]:
        v.mkdir()
        os.utime(v, (mtime, mtime))

    # active is v1 (the oldest by mtime)
    removed = prune_versions(venvs, active_version='1.0.0', keep=2)

    # Must preserve v1 (active) and v4 (newest). v2 and v3 removed.
    assert removed == ['2.0.0', '3.0.0']
    assert v1.exists()
    assert v4.exists()
    assert not v2.exists()
    assert not v3.exists()


def test_prune_versions_removes_transient_hidden_directories(tmp_path: pathlib.Path) -> None:
    venvs = tmp_path / 'venvs'
    venvs.mkdir()
    (venvs / '1.0.0').mkdir()
    (venvs / '.2.0.0.staging').mkdir()
    (venvs / '.3.0.0.old').mkdir()

    removed = prune_versions(venvs, active_version='1.0.0', keep=2)

    # Transient directories are not version directories and never count
    # against the retention budget.
    assert removed == []
    assert (venvs / '1.0.0').exists()
    assert not (venvs / '.2.0.0.staging').exists()
    assert not (venvs / '.3.0.0.old').exists()


# ============================================================================
# Query Helpers
# ============================================================================


def test_get_active_version(tmp_path: pathlib.Path) -> None:
    assert get_active_version(tmp_path) is None

    link = tmp_path / 'current'
    os.symlink(pathlib.Path('venvs') / '1.2.3', link)
    assert get_active_version(tmp_path) == '1.2.3'


def test_is_version_installed(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    v_dir = tmp_path / 'venvs' / '1.0.0'
    bin_dir = v_dir / 'bin'
    bin_dir.mkdir(parents=True)
    python_bin = bin_dir / 'python3'
    python_bin.touch()
    req_file = v_dir / '.wheelhouse-requirement'
    req_file.write_text('my-package==1.0.0\n')

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        if 'bad_import' in cmd[-1]:
            return subprocess.CompletedProcess(cmd, returncode=1)
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(utils, 'run', fake_run)

    # Success case
    assert is_version_installed(tmp_path, '1.0.0', 'my-package==1.0.0', 'good_mod') is True
    # Missing version
    assert is_version_installed(tmp_path, '2.0.0', 'my-package==1.0.0', 'good_mod') is False
    # Requirement mismatch (e.g. extras changed)
    assert is_version_installed(tmp_path, '1.0.0', 'my-package[extra]==1.0.0', 'good_mod') is False
    # Self-check failure
    assert is_version_installed(tmp_path, '1.0.0', 'my-package==1.0.0', 'bad_import') is False


# ============================================================================
# Layer 2: Python Wheelhouse Workflow & Transactional Safety
# ============================================================================


def test_install_and_activate_wheelhouse_success(
    tmp_path: pathlib.Path, writable_root: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_root = tmp_path / 'workload'
    wheelhouse_dir = tmp_path / 'wheelhouse'
    wheelhouse_dir.mkdir()

    commands: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(cmd)
        # Mock venv creation creating bin/python3 and bin/pip
        if '-m' in cmd and 'venv' in cmd:
            venv_path = pathlib.Path(cmd[-1])
            (venv_path / 'bin').mkdir(parents=True, exist_ok=True)
            (venv_path / 'bin' / 'python3').touch()
            (venv_path / 'bin' / 'pip').touch()
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(utils, 'run', fake_run)

    install_and_activate_wheelhouse(
        wheelhouse_dir=wheelhouse_dir,
        install_root=install_root,
        version='1.0.0',
        requirement='my-pkg==1.0.0',
        import_name='my_pkg',
    )

    # 1. current symlink points to venvs/1.0.0
    assert get_active_version(install_root) == '1.0.0'
    # 2. installed-version stamp is not written (current symlink is sole physical source of truth)
    assert not (install_root / 'installed-version').exists()
    # 3. requirement stamp written
    stamp = install_root / 'venvs' / '1.0.0' / '.wheelhouse-requirement'
    assert stamp.read_text().strip() == 'my-pkg==1.0.0'


def test_install_and_activate_wheelhouse_pip_failure_transactional_cleanup(
    tmp_path: pathlib.Path, writable_root: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_root = tmp_path / 'workload'
    wheelhouse_dir = tmp_path / 'wheelhouse'
    wheelhouse_dir.mkdir()

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        if '-m' in cmd and 'venv' in cmd:
            venv_path = pathlib.Path(cmd[-1])
            (venv_path / 'bin').mkdir(parents=True, exist_ok=True)
            (venv_path / 'bin' / 'python3').touch()
            (venv_path / 'bin' / 'pip').touch()
            return subprocess.CompletedProcess(cmd, returncode=0)
        if 'install' in cmd:
            raise subprocess.CalledProcessError(1, cmd, stderr=b'pip failed resolution')
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(utils, 'run', fake_run)

    with pytest.raises(WorkloadInstallError, match='pip install failed'):
        install_and_activate_wheelhouse(
            wheelhouse_dir=wheelhouse_dir,
            install_root=install_root,
            version='1.0.0',
            requirement='my-pkg==1.0.0',
            import_name='my_pkg',
        )

    # Transactional Safety: Candidate directory deleted, no current symlink created
    assert not (install_root / 'venvs' / '1.0.0').exists()
    assert not (install_root / 'current').exists()


def test_install_and_activate_wheelhouse_self_check_failure_transactional_cleanup(
    tmp_path: pathlib.Path, writable_root: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_root = tmp_path / 'workload'
    wheelhouse_dir = tmp_path / 'wheelhouse'
    wheelhouse_dir.mkdir()

    # Pre-existing working version 0.9.0
    v09 = install_root / 'venvs' / '0.9.0'
    v09.mkdir(parents=True)
    current = install_root / 'current'
    flip_symlink(current, v09)

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        if '-m' in cmd and 'venv' in cmd:
            venv_path = pathlib.Path(cmd[-1])
            (venv_path / 'bin').mkdir(parents=True, exist_ok=True)
            (venv_path / 'bin' / 'python3').touch()
            (venv_path / 'bin' / 'pip').touch()
            return subprocess.CompletedProcess(cmd, returncode=0)
        if 'import broken_mod' in cmd[-1]:
            return subprocess.CompletedProcess(cmd, returncode=1, stderr=b'ImportError: broken')
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(utils, 'run', fake_run)

    with pytest.raises(WorkloadInstallError, match='failed self-check'):
        install_and_activate_wheelhouse(
            wheelhouse_dir=wheelhouse_dir,
            install_root=install_root,
            version='1.0.0',
            requirement='my-pkg==1.0.0',
            import_name='broken_mod',
        )

    # Failed v1.0.0 cleaned up completely
    assert not (install_root / 'venvs' / '1.0.0').exists()
    # Pre-existing active version undisturbed!
    assert get_active_version(install_root) == '0.9.0'


def test_install_and_activate_wheelhouse_requirement_stamp_oserror_transactional_cleanup(
    tmp_path: pathlib.Path, writable_root: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_root = tmp_path / 'workload'
    wheelhouse_dir = tmp_path / 'wheelhouse'
    wheelhouse_dir.mkdir()

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        if '-m' in cmd and 'venv' in cmd:
            venv_path = pathlib.Path(cmd[-1])
            (venv_path / 'bin').mkdir(parents=True, exist_ok=True)
            (venv_path / 'bin' / 'python3').touch()
            (venv_path / 'bin' / 'pip').touch()
            return subprocess.CompletedProcess(cmd, returncode=0)
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(utils, 'run', fake_run)

    orig_open_file_secure = utils.open_file_secure

    def mock_open_file_secure(path: pathlib.Path, **kwargs: object) -> typing.Any:
        if path.name == '.wheelhouse-requirement':
            raise OSError('Disk full')
        return orig_open_file_secure(path, **kwargs)  # type: ignore[call-overload]  # mock forwards kwargs to overloaded function

    monkeypatch.setattr(utils, 'open_file_secure', mock_open_file_secure)

    with pytest.raises(WorkloadInstallError, match='Failed to write requirement stamp: Disk full'):
        install_and_activate_wheelhouse(
            wheelhouse_dir=wheelhouse_dir,
            install_root=install_root,
            version='1.0.0',
            requirement='my-pkg==1.0.0',
            import_name='my_pkg',
        )

    # Failed v1.0.0 directory cleaned up
    assert not (install_root / 'venvs' / '1.0.0').exists()


def test_install_and_activate_wheelhouse_no_installed_version_stamp(
    tmp_path: pathlib.Path, writable_root: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_root = tmp_path / 'workload'
    wheelhouse_dir = tmp_path / 'wheelhouse'
    wheelhouse_dir.mkdir()

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        if '-m' in cmd and 'venv' in cmd:
            venv_path = pathlib.Path(cmd[-1])
            (venv_path / 'bin').mkdir(parents=True, exist_ok=True)
            (venv_path / 'bin' / 'python3').touch()
            (venv_path / 'bin' / 'pip').touch()
            return subprocess.CompletedProcess(cmd, returncode=0)
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(utils, 'run', fake_run)

    install_and_activate_wheelhouse(
        wheelhouse_dir=wheelhouse_dir,
        install_root=install_root,
        version='1.0.0',
        requirement='my-pkg==1.0.0',
        import_name='my_pkg',
    )

    assert not (install_root / 'installed-version').exists()
    assert get_active_version(install_root) == '1.0.0'


def _fake_venv_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
    cmd = list(cmd)
    if '-m' in cmd and 'venv' in cmd:
        venv_path = pathlib.Path(cmd[-1])
        (venv_path / 'bin').mkdir(parents=True, exist_ok=True)
        (venv_path / 'bin' / 'python3').touch()
        (venv_path / 'bin' / 'pip').touch()
    return subprocess.CompletedProcess(cmd, returncode=0)


def test_install_and_activate_wheelhouse_failure_preserves_active_version(
    tmp_path: pathlib.Path, writable_root: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed reinstall of the active version must not destroy the running venv."""
    install_root = tmp_path / 'workload'
    wheelhouse_dir = tmp_path / 'wheelhouse'
    wheelhouse_dir.mkdir()

    monkeypatch.setattr(utils, 'run', _fake_venv_run)
    install_and_activate_wheelhouse(
        wheelhouse_dir=wheelhouse_dir,
        install_root=install_root,
        version='1.0.0',
        requirement='my-pkg==1.0.0',
        import_name='my_pkg',
    )
    stamp = install_root / 'venvs' / '1.0.0' / '.wheelhouse-requirement'
    assert stamp.read_text().strip() == 'my-pkg==1.0.0'

    # The requirement changes while the version tag stays the same; pip now fails.
    def failing_pip_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        cmd = list(cmd)
        if 'install' in cmd:
            raise subprocess.CalledProcessError(1, cmd, stderr=b'pip failed resolution')
        return _fake_venv_run(cmd, **kw)

    monkeypatch.setattr(utils, 'run', failing_pip_run)

    with pytest.raises(WorkloadInstallError, match='pip install failed'):
        install_and_activate_wheelhouse(
            wheelhouse_dir=wheelhouse_dir,
            install_root=install_root,
            version='1.0.0',
            requirement='my-pkg[extra]==1.0.0',
            import_name='my_pkg',
        )

    # The previously activated deployment is untouched and still resolves.
    assert get_active_version(install_root) == '1.0.0'
    assert (install_root / 'current').resolve() == (install_root / 'venvs' / '1.0.0').resolve()
    assert stamp.read_text().strip() == 'my-pkg==1.0.0'
    # The failed staging candidate was cleaned up.
    assert not (install_root / 'venvs' / '.1.0.0.staging').exists()


def test_install_and_activate_wheelhouse_reinstall_replaces_active_venv(
    tmp_path: pathlib.Path, writable_root: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_root = tmp_path / 'workload'
    wheelhouse_dir = tmp_path / 'wheelhouse'
    wheelhouse_dir.mkdir()

    monkeypatch.setattr(utils, 'run', _fake_venv_run)

    install_and_activate_wheelhouse(
        wheelhouse_dir=wheelhouse_dir,
        install_root=install_root,
        version='1.0.0',
        requirement='my-pkg==1.0.0',
        import_name='my_pkg',
    )
    # Reinstalling the same version with a different requirement replaces the venv.
    install_and_activate_wheelhouse(
        wheelhouse_dir=wheelhouse_dir,
        install_root=install_root,
        version='1.0.0',
        requirement='my-pkg[extra]==1.0.0',
        import_name='my_pkg',
    )

    stamp = install_root / 'venvs' / '1.0.0' / '.wheelhouse-requirement'
    assert stamp.read_text().strip() == 'my-pkg[extra]==1.0.0'
    assert get_active_version(install_root) == '1.0.0'
    # Neither the staging directory nor the swap backup leaks.
    assert list((install_root / 'venvs').glob('.*')) == []


def test_install_and_activate_wheelhouse_rewrites_staging_paths(
    tmp_path: pathlib.Path, writable_root: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_root = tmp_path / 'workload'
    wheelhouse_dir = tmp_path / 'wheelhouse'
    wheelhouse_dir.mkdir()

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        cmd = list(cmd)
        if '-m' in cmd and 'venv' in cmd:
            venv_path = pathlib.Path(cmd[-1])
            (venv_path / 'bin').mkdir(parents=True, exist_ok=True)
            (venv_path / 'bin' / 'python3').touch()
            # Console scripts get the build-time venv path baked into their shebang.
            (venv_path / 'bin' / 'entrypoint').write_text(f'#!{venv_path}/bin/python3\n')
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(utils, 'run', fake_run)

    install_and_activate_wheelhouse(
        wheelhouse_dir=wheelhouse_dir,
        install_root=install_root,
        version='1.0.0',
        requirement='my-pkg==1.0.0',
        import_name='my_pkg',
    )

    entrypoint = install_root / 'venvs' / '1.0.0' / 'bin' / 'entrypoint'
    assert entrypoint.read_text() == f'#!{install_root}/venvs/1.0.0/bin/python3\n'


def test_install_and_activate_wheelhouse_pip_timeout(
    tmp_path: pathlib.Path, writable_root: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_root = tmp_path / 'workload'
    wheelhouse_dir = tmp_path / 'wheelhouse'
    wheelhouse_dir.mkdir()

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        cmd = list(cmd)
        if '-m' in cmd and 'venv' in cmd:
            return _fake_venv_run(cmd, **kw)
        if 'install' in cmd:
            raise subprocess.TimeoutExpired(cmd, 300)
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(utils, 'run', fake_run)

    with pytest.raises(WorkloadInstallError, match='pip install timed out'):
        install_and_activate_wheelhouse(
            wheelhouse_dir=wheelhouse_dir,
            install_root=install_root,
            version='1.0.0',
            requirement='my-pkg==1.0.0',
            import_name='my_pkg',
        )

    assert not (install_root / 'venvs' / '1.0.0').exists()
    assert not (install_root / 'venvs' / '.1.0.0.staging').exists()
