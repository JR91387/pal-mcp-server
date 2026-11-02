"""
Symlink regression tests for file path validation.

A symlink inside an allowed directory must not give access to a blocked
location: resolve_and_validate_path() resolves symlinks before checking
is_dangerous_path(), and file reads go through it.
"""

from pathlib import Path

import pytest

from utils.file_utils import expand_paths, read_file_content, resolve_and_validate_path


@pytest.fixture
def project(tmp_path):
    (tmp_path / "ok.py").write_text("print('ok')\n")
    (tmp_path / "passwd_link.py").symlink_to("/etc/passwd")
    (tmp_path / "etc_dir_link").symlink_to("/etc")
    return tmp_path


def test_symlink_to_etc_file_refused(project):
    with pytest.raises(PermissionError):
        resolve_and_validate_path(str(project / "passwd_link.py"))


def test_symlinked_dir_to_etc_refused(project):
    with pytest.raises(PermissionError):
        resolve_and_validate_path(str(project / "etc_dir_link" / "passwd"))


def test_expand_paths_skips_direct_symlink_to_etc(project):
    assert expand_paths([str(project / "passwd_link.py")]) == []
    assert expand_paths([str(project / "etc_dir_link")]) == []


def test_symlink_found_by_directory_scan_is_not_read(project):
    for found in expand_paths([str(project)]):
        content, _ = read_file_content(found)
        assert "root:" not in content


def test_regular_file_still_allowed(project):
    assert resolve_and_validate_path(str(project / "ok.py")) == (project / "ok.py").resolve()


@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="no /proc")
def test_symlink_to_proc_environ_refused(tmp_path):
    """Needs /proc in DANGEROUS_SYSTEM_PATHS (fix/block-proc-sys)."""
    link = tmp_path / "env_link.txt"
    link.symlink_to("/proc/self/environ")
    with pytest.raises(PermissionError):
        resolve_and_validate_path(str(link))
