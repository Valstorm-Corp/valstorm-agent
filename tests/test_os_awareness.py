import pytest
from core.context import WorkspaceContextManager
from core.sandbox import translate_cross_os_path, HostSandbox


def test_system_prompt_os_awareness_linux_wsl(tmp_path):
    mgr = WorkspaceContextManager(workdir=str(tmp_path))
    prompt = mgr.build_system_prompt(user_context={"os_target": "linux"})

    assert "Operating System: Linux (WSL Subsystem" in prompt
    assert "Filesystem Root: /" in prompt
    assert "/mnt/c/" in prompt
    assert "Cross-Boundary Spanning" in prompt


def test_system_prompt_os_awareness_windows(tmp_path):
    mgr = WorkspaceContextManager(workdir=str(tmp_path))
    prompt = mgr.build_system_prompt(user_context={"os_target": "windows"})

    assert "Operating System: Windows (Native Host)" in prompt
    assert "Filesystem Root: C:\\" in prompt
    assert "wsl.localhost" in prompt
    assert "Cross-Boundary Spanning" in prompt


def test_translate_cross_os_path_linux_mode():
    # UNC WSL paths -> Linux root paths
    assert (
        translate_cross_os_path(
            r"\\wsl.localhost\Ubuntu\home\llfbh33\valstorm\monorepo",
            platform_override="linux",
        )
        == "/home/llfbh33/valstorm/monorepo"
    )
    assert (
        translate_cross_os_path(r"\\wsl.localhost\Ubuntu", platform_override="linux")
        == "/"
    )
    # Corrupted joined path containing \\wsl.localhost\
    corrupted = r"/home/llfbh33/valstorm/monorepo/apps/valstorm-desktop/src-tauri/\\wsl.localhost\Ubuntu\home\llfbh33\valstorm\monorepo"
    assert (
        translate_cross_os_path(corrupted, platform_override="linux")
        == "/home/llfbh33/valstorm/monorepo"
    )
    # Windows drive path -> /mnt/<drive>/...
    assert (
        translate_cross_os_path(r"C:\Users\Aubrie\Projects", platform_override="linux")
        == "/mnt/c/Users/Aubrie/Projects"
    )


def test_translate_cross_os_path_windows_mode():
    # /mnt/<drive>/... -> Windows drive path
    assert (
        translate_cross_os_path("/mnt/c/Users/Aubrie/Projects", platform_override="win32")
        == r"C:\Users\Aubrie\Projects"
    )
    # Linux path -> \\wsl.localhost\<distro>\...
    translated_root = translate_cross_os_path("/", platform_override="win32")
    assert translated_root.startswith("\\\\wsl.localhost\\")
    translated_home = translate_cross_os_path("/home/aubrie/code", platform_override="win32")
    assert translated_home.endswith(r"\home\aubrie\code")


def test_host_sandbox_stores_os_target_and_resolves_cross_boundary_paths(tmp_path):
    sb = HostSandbox(base_dir=str(tmp_path), os_target="linux")
    assert sb.os_target == "linux"
    assert sb.base_dir == tmp_path.resolve()

