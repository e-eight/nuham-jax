import subprocess
import sys


def _run(env_extra: dict[str, str]) -> str:
    import os

    env = {k: v for k, v in os.environ.items() if k != "XLA_PYTHON_CLIENT_PREALLOCATE"}
    env.update(env_extra)
    code = "import nuham, os; print(os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'))"
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_gpu_memory_is_not_preallocated_by_default() -> None:
    assert _run({}) == "false"


def test_explicit_user_setting_wins() -> None:
    assert _run({"XLA_PYTHON_CLIENT_PREALLOCATE": "true"}) == "true"
