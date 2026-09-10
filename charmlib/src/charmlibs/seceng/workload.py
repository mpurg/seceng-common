# Copyright 2026 Canonical Ltd.
#
# SPDX-License-Identifier: LGPL-3.0-only

"""Stateless workload deployment and lifecycle primitives for SecEng charms.

This module provides pure Python, ops-free primitives and workflows for
unpacking archives safely, building isolated virtual environments, and
atomically activating workloads via symlink flips. Release artifacts are
acquired by charmlibs.seceng.github, and the units that run a deployed workload
are managed by charmlibs.seceng.systemd.
"""

from __future__ import annotations

__all__ = [
    'ArtifactExtractionError',
    'WorkloadError',
    'WorkloadInstallError',
    'flip_symlink',
    'get_active_version',
    'install_and_activate_wheelhouse',
    'is_version_installed',
    'prune_versions',
    'unpack_archive',
    'validate_version',
]

import contextlib
import logging
import os
import pathlib
import re
import shutil
import subprocess
import tarfile

from . import utils

_VERSION_PATTERN = re.compile(r'\A[A-Za-z0-9][A-Za-z0-9._-]*\Z')
_SELF_CHECK_REASON_LIMIT = 200
# Bound for every local subprocess (venv creation, pip, import self-check);
# without it a hung child blocks the hook until Juju kills it.
_SUBPROCESS_TIMEOUT_SECONDS = 300


class WorkloadError(Exception):
    """Base error for workload operations, safe for Juju status messages."""


class ArtifactExtractionError(WorkloadError):
    """Archive extraction failed or contained unsafe paths."""


class WorkloadInstallError(WorkloadError):
    """Virtual environment creation, pip install, or self-check failed."""


def validate_version(version: str) -> str:
    """Validate release tag string format.

    Rejects empty strings, path traversal, leading hyphens, and URL metacharacters.
    Raises WorkloadError on invalid version.
    """
    if not version:
        raise WorkloadError('version must not be empty.')
    if not _VERSION_PATTERN.match(version):
        raise WorkloadError(
            f'version {version!r} is not a valid release tag: use only letters, digits, dot, underscore, '
            'and hyphen, starting with a letter or digit.'
        )
    return version


def _swap_directory(staging_dir: pathlib.Path, dest_dir: pathlib.Path) -> None:
    """Atomically replace dest_dir with staging_dir.

    The previous dest_dir (if any) is renamed aside, the staging directory is
    renamed into place, and the previous copy is deleted. If the final rename
    fails, the previous copy is restored; if restoration also fails, the backup
    is left next to dest_dir (hidden, dot-prefixed) for prune_versions() or a
    later invocation to clean up.
    """
    backup = dest_dir.with_name(f'.{dest_dir.name}.old')
    shutil.rmtree(backup, ignore_errors=True)
    if dest_dir.is_symlink() or dest_dir.exists():
        os.replace(dest_dir, backup)
    try:
        os.replace(staging_dir, dest_dir)
    except OSError:
        if backup.is_symlink() or backup.exists():
            try:
                os.replace(backup, dest_dir)
            except OSError:
                logging.exception(f'Failed to restore previous directory at {dest_dir}; kept {backup}')
        raise
    shutil.rmtree(backup, ignore_errors=True)


def unpack_archive(archive_path: pathlib.Path, dest_dir: pathlib.Path) -> None:
    """Extract a tar archive into dest_dir, which must not yet exist.

    The destination is claimed with mkdir(exist_ok=False), so extraction only
    ever populates a directory this call created and never writes into a tree a
    workload may be reading. dest_dir.parent must already exist: creating it
    would mean inventing the ownership and permissions that belong to the
    caller.

    Device nodes, links whose targets escape the extraction root, and member
    paths that are absolute, empty, or contain '..' are all refused. The
    archive's own layout is preserved. dest_dir is removed again if anything
    fails, so a partial extraction never outlives the failure.

    Raises ArtifactExtractionError.
    """
    if not archive_path.is_file():
        raise ArtifactExtractionError(f'archive file not found: {archive_path}')

    try:
        dest_dir.mkdir(exist_ok=False)
    except OSError as err:
        raise ArtifactExtractionError(f'failed to claim extraction directory {dest_dir}: {err}') from err

    with contextlib.ExitStack() as on_failure:
        on_failure.callback(shutil.rmtree, dest_dir, ignore_errors=True)
        try:
            with tarfile.open(archive_path, mode='r:*') as archive:
                members = archive.getmembers()
                for member in members:
                    member_path = pathlib.PurePosixPath(member.name)
                    if not member_path.parts or member_path.is_absolute() or '..' in member_path.parts:
                        raise ArtifactExtractionError(f'archive contains unsafe member path: {member.name!r}')
                archive.extractall(dest_dir, members=members, filter='data')
        except (OSError, tarfile.TarError, EOFError) as err:
            raise ArtifactExtractionError(f'failed to extract archive {archive_path.name!r}: {err}') from err
        on_failure.pop_all()


def flip_symlink(current_link: pathlib.Path, target_dir: pathlib.Path) -> None:
    """Atomically update current_link to point to target_dir via os.replace.

    A temporary symlink is created in current_link's parent directory and
    atomically renamed over current_link.
    """
    if current_link.exists(follow_symlinks=False) and not current_link.is_symlink():
        raise WorkloadError(f'{current_link} exists and is not a symlink; refusing to replace.')

    current_link.parent.mkdir(parents=True, exist_ok=True)
    temp_link = current_link.with_name(f'.{current_link.name}.tmp')
    if temp_link.is_symlink() or temp_link.exists():
        temp_link.unlink()

    target_to_store: pathlib.Path
    if target_dir.is_absolute():
        try:
            target_to_store = target_dir.relative_to(current_link.parent)
        except ValueError:
            target_to_store = target_dir
    else:
        target_to_store = target_dir

    os.symlink(target_to_store, temp_link)
    try:
        os.replace(temp_link, current_link)
    except OSError as err:
        if temp_link.is_symlink() or temp_link.exists():
            temp_link.unlink(missing_ok=True)
        raise WorkloadError(f'Failed to atomically swap symlink {current_link}: {err}') from err


def prune_versions(versions_dir: pathlib.Path, active_version: str | None, keep: int = 2) -> list[str]:
    """Remove oldest version directories by modification time, retaining keep versions.

    The active_version is always preserved if present. Hidden dot-prefixed
    directories are transient build artifacts (version tags can never start
    with a dot) and are always removed.
    Returns the sorted list of removed version directory names.
    """
    if keep < 1:
        raise ValueError('keep must be at least 1.')
    if not versions_dir.exists():
        return []

    entries: list[pathlib.Path] = []
    for entry in versions_dir.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            continue
        if entry.name.startswith('.'):
            shutil.rmtree(entry, ignore_errors=True)
            continue
        entries.append(entry)
    retained: list[str] = [active_version] if active_version else []
    for entry in sorted(entries, key=lambda e: e.stat().st_mtime, reverse=True):
        if len(retained) >= keep:
            break
        if entry.name not in retained:
            retained.append(entry.name)

    removed: list[str] = []
    for entry in entries:
        if entry.name not in retained:
            shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry.name)
    return sorted(removed)


def get_active_version(install_root: pathlib.Path) -> str | None:
    """Return the version currently targeted by install_root / 'current', if any."""
    current_link = install_root / 'current'
    try:
        target = os.readlink(current_link)
    except OSError:
        return None
    return pathlib.PurePosixPath(target).name


def is_version_installed(
    install_root: pathlib.Path,
    version: str,
    requirement: str,
    import_name: str,
) -> bool:
    """Return whether a version venv exists, matches requirement, and imports cleanly."""
    venv_dir = install_root / 'venvs' / version
    python_bin = venv_dir / 'bin' / 'python3'
    if not python_bin.is_file():
        python_bin = venv_dir / 'bin' / 'python'
        if not python_bin.is_file():
            return False

    requirement_stamp = venv_dir / '.wheelhouse-requirement'
    try:
        content = requirement_stamp.read_text(encoding='utf-8').strip()
    except OSError:
        return False
    if content != requirement.strip():
        return False

    if not all(part.isidentifier() for part in import_name.split('.')):
        return False

    try:
        result = utils.run(
            [str(python_bin), '-c', f'import {import_name}'],
            check=False,
            capture=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _rewrite_venv_paths(staging_dir: pathlib.Path, venv_dir: pathlib.Path) -> None:
    """Rewrite absolute staging paths in venv scripts to their final location.

    Virtualenv creation and pip bake the absolute staging path into script
    shebangs and activation scripts; after the staging directory is renamed
    into place those references would dangle.
    """
    old_path = str(staging_dir).encode('utf-8')
    new_path = str(venv_dir).encode('utf-8')
    for entry in (staging_dir / 'bin').iterdir():
        if entry.is_symlink() or not entry.is_file():
            continue
        data = entry.read_bytes()
        if old_path in data:
            entry.write_bytes(data.replace(old_path, new_path))


def install_and_activate_wheelhouse(
    wheelhouse_dir: pathlib.Path,
    install_root: pathlib.Path,
    version: str,
    requirement: str,
    import_name: str,
) -> None:
    """Build, verify, and activate a Python wheelhouse workload atomically.

    Executes the following sequence:
    1. Validates version tag.
    2. Creates a staging virtualenv at install_root / 'venvs' / ('.' + version + '.staging').
    3. Runs pip install --isolated --no-index --find-links wheelhouse_dir --force-reinstall requirement.
    4. Rewrites staging paths baked into venv scripts to their final location.
    5. Runs module self-check: <venv_python> -c "import <import_name>".
    6. Writes requirement stamp: <staging> / '.wheelhouse-requirement'.
    7. Atomically swaps the staging venv into install_root / 'venvs' / version.
    8. Atomically flips install_root / 'current' -> venvs / version.
    9. Prunes old versions in install_root / 'venvs' (retains active + 1 previous).

    Transactional Safety: the candidate venv is built under a hidden staging
    path and only swapped into place once every check passes, so a failure at
    any step -- including a reinstall of the currently active version -- leaves
    the active deployment and the current symlink fully intact. Failed staging
    builds are deleted immediately. All subprocesses are bounded by a
    300 second timeout.
    Raises WorkloadInstallError on failure.
    """
    try:
        version = validate_version(version)
    except WorkloadError as err:
        raise WorkloadInstallError(str(err)) from err

    if not all(part.isidentifier() for part in import_name.split('.')):
        raise WorkloadInstallError(f'Invalid import name: {import_name!r}')

    venvs_dir = install_root / 'venvs'
    venv_dir = venvs_dir / version
    staging_dir = venvs_dir / f'.{version}.staging'
    try:
        venvs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise WorkloadInstallError(f'Failed to create venvs directory: {err}') from err

    # A previous crashed attempt may have left a staging directory behind.
    shutil.rmtree(staging_dir, ignore_errors=True)

    try:
        try:
            utils.run(
                ['/usr/bin/python3', '-m', 'venv', str(staging_dir)],
                check=True,
                capture=True,
                timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as err:
            raise WorkloadInstallError(f'Failed to create virtualenv for {version}: {err}') from err

        python_bin = staging_dir / 'bin' / 'python3'
        if not python_bin.exists():
            python_bin = staging_dir / 'bin' / 'python'

        pip_bin = staging_dir / 'bin' / 'pip'

        try:
            utils.run(
                [
                    str(pip_bin),
                    'install',
                    '--isolated',
                    '--no-index',
                    '--find-links',
                    str(wheelhouse_dir),
                    '--force-reinstall',
                    requirement,
                ],
                check=True,
                capture=True,
                timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as err:
            raise WorkloadInstallError(f'pip install timed out for {requirement} in {version}') from err
        except subprocess.CalledProcessError as err:
            detail = ''
            if err.stderr:
                lines = [
                    line.strip() for line in err.stderr.decode('utf-8', errors='replace').splitlines() if line.strip()
                ]
                if lines:
                    detail = f': {lines[-1][:_SELF_CHECK_REASON_LIMIT]}'
            raise WorkloadInstallError(f'pip install failed for {requirement} in {version}{detail}') from err

        try:
            _rewrite_venv_paths(staging_dir, venv_dir)
        except OSError as err:
            raise WorkloadInstallError(f'Failed to rewrite staging paths for {version}: {err}') from err

        try:
            result = utils.run(
                [str(python_bin), '-c', f'import {import_name}'],
                check=False,
                capture=True,
                timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as err:
            raise WorkloadInstallError(
                f'Workload {version} failed self-check: import of {import_name} timed out'
            ) from err
        if result.returncode != 0:
            detail = ''
            if result.stderr:
                lines = [
                    line.strip()
                    for line in result.stderr.decode('utf-8', errors='replace').splitlines()
                    if line.strip()
                ]
                if lines:
                    detail = f' ({lines[-1][:_SELF_CHECK_REASON_LIMIT]})'
            raise WorkloadInstallError(
                f'Workload {version} failed self-check: {import_name} could not be imported{detail}'
            )

        requirement_stamp = staging_dir / '.wheelhouse-requirement'
        try:
            with utils.open_file_secure(requirement_stamp, mode=0o644, create_parents=True) as f:
                f.write(f'{requirement}\n')
        except OSError as err:
            raise WorkloadInstallError(f'Failed to write requirement stamp: {err}') from err

        try:
            _swap_directory(staging_dir, venv_dir)
        except OSError as err:
            raise WorkloadInstallError(f'Failed to activate virtualenv for {version}: {err}') from err
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    current_link = install_root / 'current'
    try:
        flip_symlink(current_link, venv_dir)
    except WorkloadError as err:
        raise WorkloadInstallError(str(err)) from err

    try:
        prune_versions(venvs_dir, active_version=version, keep=2)
    except (OSError, WorkloadError) as err:
        raise WorkloadInstallError(f'Failed to prune old versions: {err}') from err
