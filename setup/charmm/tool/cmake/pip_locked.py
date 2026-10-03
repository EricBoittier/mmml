"""
pip_locked.py — Cross-platform serialized pip installer.

Wraps pip with a file-based lock to prevent race conditions when multiple
concurrent CMake builds attempt to install a Python package into the same
conda environment simultaneously.

Usage (from CMake or command line):
    python pip_locked.py install <package>
    python pip_locked.py install -e <path>
    python pip_locked.py uninstall <package>

    All arguments are passed through directly to pip unchanged.

Locking mechanism:
    - Unix/macOS: fcntl.flock()   — kernel-level, blocking, auto-released on crash
    - Windows:    msvcrt.locking() — byte-range lock, polled since no blocking mode

Lock file location:
    - Unix/macOS: /tmp/pip_install.lock
    - Windows:    %TEMP%\\pip_install.lock
"""

import os
import subprocess
import sys
import time


# Lock file path, chosen per platform.
# On Unix, /tmp is world-writable and survives reboots on most distros.
# On Windows, %TEMP% is user-scoped, which is fine since conda environments
# are typically per-user too.
_LOCK_FILE = (
    os.path.join(os.environ.get("TEMP", "C:\\Temp"), "pip_install.lock")
    if os.name == "nt"
    else "/tmp/pip_install.lock"
)

# How long to wait between lock attempts on Windows (seconds).
_WINDOWS_POLL_INTERVAL = 0.1


def _lock_unix(lock_path: str):
    """Acquire an exclusive file lock using fcntl (Unix/macOS only).

    Opens (or creates) the lock file and acquires an exclusive lock via
    fcntl.flock(). The call blocks until the lock is available — no polling
    required. The lock is automatically released by the OS if the process
    crashes, preventing deadlocks.

    Args:
        lock_path: Path to the lock file to create/open.

    Returns:
        The open file object. Must be passed to _unlock_unix() when done.
    """
    import fcntl
    f = open(lock_path, "w")
    fcntl.flock(f, fcntl.LOCK_EX)  # Blocks until all other holders release
    return f


def _unlock_unix(f) -> None:
    """Release a Unix file lock acquired by _lock_unix().

    Explicitly unlocks before closing. The LOCK_UN call is technically
    redundant — closing the file also releases the lock — but makes the
    intent clear and avoids relying on that implicit behaviour.

    Args:
        f: The open file object returned by _lock_unix().
    """
    import fcntl
    fcntl.flock(f, fcntl.LOCK_UN)
    f.close()


def _lock_windows(lock_path: str):
    """Acquire an exclusive file lock using msvcrt (Windows only).

    msvcrt.locking() does not support a blocking mode — it raises OSError
    immediately if the lock is held by another process. This function
    emulates blocking by polling at a fixed interval until the lock is
    acquired.

    Locks a single byte (byte 0) of the lock file, which is the conventional
    minimal unit for advisory locking on Windows.

    Args:
        lock_path: Path to the lock file to create/open.

    Returns:
        The open file object. Must be passed to _unlock_windows() when done.
    """
    import msvcrt
    f = open(lock_path, "w")
    while True:
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)  # Non-blocking attempt
            return f  # Lock acquired
        except OSError:
            time.sleep(_WINDOWS_POLL_INTERVAL)  # Lock held elsewhere — retry


def _unlock_windows(f) -> None:
    """Release a Windows file lock acquired by _lock_windows().

    Unlocks the single byte that was locked by _lock_windows(), then closes
    the file. Unlike Unix, Windows locks are not released implicitly on
    close, so the explicit LK_UNLCK call is required.

    Args:
        f: The open file object returned by _lock_windows().
    """
    import msvcrt
    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    f.close()


def main() -> None:
    """Acquire a cross-platform file lock and run pip with the given arguments.

    Selects the appropriate lock/unlock implementation for the current OS,
    acquires the lock (blocking until available), delegates to pip via
    subprocess, then releases the lock. The lock is always released in a
    finally block, so a pip failure or exception never leaves it held.

    Pip is invoked as `sys.executable -m pip` to guarantee it runs inside
    the same Python environment that CMake resolved — not whatever pip
    happens to be first on PATH.

    Exits with pip's return code so CMake can detect and propagate failures.
    """
    if os.name == "nt":
        lock   = _lock_windows
        unlock = _unlock_windows
    else:
        lock   = _lock_unix
        unlock = _unlock_unix

    f = lock(_LOCK_FILE)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip"] + sys.argv[1:],
            check=False  # We propagate the return code ourselves
        )
        sys.exit(result.returncode)
    finally:
        unlock(f)  # Always release — even if pip crashes or returns non-zero


if __name__ == "__main__":
    main()
