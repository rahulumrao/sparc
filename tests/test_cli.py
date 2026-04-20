import subprocess
import sys

def test_cli_help():
    # only works if entrypoint is installed as sparc
    p = subprocess.run(["sparc", "-h"], capture_output=True, text=True)
    assert p.returncode == 0
    assert "usage" in (p.stdout + p.stderr).lower()
