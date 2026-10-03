"""Safeguards system for pycharmm debugging and error prevention.

This module provides:
- Exception classes for CHARMM-related errors
- Optional pre-validation and logging to help users debug CHARMM scripts
- Persistent logging that survives kernel crashes (for Jupyter notebooks)

Design Philosophy:
- INFORMATIVE, NOT RESTRICTIVE: Warnings only, never blocks operations
- OPT-IN: Disabled by default, doesn't change existing behavior
- PERSISTENT: Log file survives kernel crashes for post-mortem debugging
- EXTENSIBLE: Users can register custom validators

Example Usage:
    >>> import pycharmm.safeguards as safe
    >>> safe.enable()  # Enable with default settings
    >>> # Or with custom log file:
    >>> safe.enable(log_file='./my_charmm_debug.log')

    # Now all charmm_script calls are logged and validated
    >>> pycharmm.read.rtf('missing.rtf')
    # [pycharmm] WARNING: File not found: missing.rtf
    # [pycharmm] Tip: Check file path. CHARMM may crash if bomb level >= 0

    # After kernel crash, check the log:
    # $ tail pycharmm_debug.log
"""

import os
import re
import sys
import warnings
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional, Dict, Any
from contextlib import contextmanager


# ============================================================================
# Exception Classes
# ============================================================================

class CharmmError(Exception):
    """Base exception for CHARMM-related errors.

    All pycharmm exceptions that relate to CHARMM operations inherit from
    this class, allowing users to catch all CHARMM errors with a single
    except clause.

    Examples
    --------
    >>> try:
    ...     pycharmm.lingo.charmm_script('invalid', raise_on_error=True)
    ... except CharmmError as e:
    ...     print(f"CHARMM operation failed: {e}")
    """
    pass


class CharmmScriptError(CharmmError):
    """Raised when a CHARMM script command fails.

    This exception is raised when charmm_script() is called with
    raise_on_error=True and the command returns a non-success status.

    Attributes
    ----------
    script : str
        The CHARMM script that failed.
    status : int
        The return status from CHARMM (1 = success, other = failure).

    Examples
    --------
    >>> import pycharmm.lingo as lingo
    >>> try:
    ...     lingo.charmm_script('bad command', raise_on_error=True)
    ... except CharmmScriptError as e:
    ...     print(f"Command failed: {e.script}")
    ...     print(f"Status: {e.status}")
    """

    def __init__(self, script: str, status: int, message: str = None):
        self.script = script
        self.status = status
        if message is None:
            # Truncate long scripts for readability
            preview = script[:100] + "..." if len(script) > 100 else script
            message = f"CHARMM command failed (status {status}): {preview}"
        super().__init__(message)


class CharmmWarning(UserWarning):
    """Warning issued for non-fatal CHARMM issues.

    This warning class is used for CHARMM-related warnings that don't
    necessarily indicate failure but may require user attention.
    """
    pass


# ============================================================================
# Safeguards Configuration
# ============================================================================

@dataclass
class SafeguardConfig:
    """Configuration for the safeguards system."""
    enabled: bool = False
    log_file: Optional[str] = None
    log_level: str = 'INFO'
    validate_files: bool = True
    validate_gpu: bool = True
    validate_state: bool = True
    show_commands: bool = True
    show_warnings: bool = True
    # Note: We never block operations, only warn


# Global configuration
_config = SafeguardConfig()
_logger: Optional[logging.Logger] = None
_file_handler: Optional[logging.FileHandler] = None
_validators: List[Tuple[re.Pattern, Callable]] = []


def _setup_logger():
    """Set up the persistent logger."""
    global _logger, _file_handler

    if _logger is not None:
        return

    _logger = logging.getLogger('pycharmm.safeguards')
    _logger.setLevel(getattr(logging, _config.log_level.upper(), logging.INFO))
    _logger.handlers.clear()

    # Console handler for warnings
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter('[pycharmm] %(message)s'))
    _logger.addHandler(console)

    # File handler - crucial for post-crash debugging
    if _config.log_file:
        try:
            _file_handler = logging.FileHandler(_config.log_file, mode='a')
            _file_handler.setLevel(logging.DEBUG)
            _file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            _logger.addHandler(_file_handler)
            # Flush immediately so logs survive crashes
            _file_handler.flush()
        except (IOError, OSError) as e:
            warnings.warn(f"Could not open log file {_config.log_file}: {e}")


def _flush_log():
    """Ensure log is written to disk immediately."""
    if _file_handler:
        _file_handler.flush()


def enable(log_file: str = './pycharmm_debug.log', **kwargs):
    """Enable the safeguards system.

    Parameters
    ----------
    log_file : str, optional
        Path to the log file for persistent debugging info.
        Default: './pycharmm_debug.log'
    validate_files : bool, optional
        Check if files exist before file operations. Default: True
    validate_gpu : bool, optional
        Check GPU availability before GPU operations. Default: True
    validate_state : bool, optional
        Check CHARMM state (PSF loaded, etc.) before operations. Default: True
    show_commands : bool, optional
        Log all CHARMM commands. Default: True
    show_warnings : bool, optional
        Display warnings to console. Default: True
    log_level : str, optional
        Logging level ('DEBUG', 'INFO', 'WARNING'). Default: 'INFO'

    Example
    -------
    >>> import pycharmm.safeguards as safe
    >>> safe.enable()  # Default settings
    >>> safe.enable(log_file='./session.log', validate_gpu=False)
    """
    global _config, _logger, _file_handler

    _config.enabled = True
    _config.log_file = log_file

    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)

    # Reset logger with new settings
    _logger = None
    _file_handler = None
    _setup_logger()

    # Register built-in validators
    _register_builtin_validators()

    if _logger:
        _logger.info("=" * 60)
        _logger.info(f"pycharmm safeguards enabled - {datetime.now()}")
        _logger.info(f"Log file: {os.path.abspath(log_file)}")
        _flush_log()


def disable():
    """Disable the safeguards system."""
    global _config, _logger, _file_handler

    if _logger:
        _logger.info("pycharmm safeguards disabled")
        _flush_log()

    _config.enabled = False

    if _file_handler:
        _file_handler.close()
        _file_handler = None
    _logger = None


def is_enabled() -> bool:
    """Check if safeguards are currently enabled."""
    return _config.enabled


@contextmanager
def enabled(log_file: str = './pycharmm_debug.log', **kwargs):
    """Context manager to temporarily enable safeguards.

    Example
    -------
    >>> with safeguards.enabled():
    ...     pycharmm.read.rtf('topology.rtf')
    """
    global _config

    was_enabled = _config.enabled
    old_config = SafeguardConfig(
        enabled=_config.enabled,
        log_file=_config.log_file,
        log_level=_config.log_level,
        validate_files=_config.validate_files,
        validate_gpu=_config.validate_gpu,
        validate_state=_config.validate_state,
        show_commands=_config.show_commands,
        show_warnings=_config.show_warnings
    )

    try:
        enable(log_file, **kwargs)
        yield
    finally:
        if was_enabled:
            # Restore old config
            _config = old_config
        else:
            disable()


# ============================================================================
# Validators
# ============================================================================

def _check_cuda_available() -> Tuple[bool, str]:
    """Check if CUDA GPU is available."""
    # Method 1: Check via environment
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible == '-1':
        return False, "CUDA_VISIBLE_DEVICES set to -1 (GPU disabled)"

    # Method 2: Try to detect CUDA library
    try:
        import ctypes
        try:
            ctypes.CDLL('libcuda.so')
            return True, "CUDA library found"
        except OSError:
            pass
        try:
            ctypes.CDLL('libcuda.dylib')  # macOS
            return True, "CUDA library found"
        except OSError:
            pass
    except Exception:
        pass

    # Method 3: Check nvidia-smi
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '-L'],
                                capture_output=True, timeout=5)
        if result.returncode == 0 and b'GPU' in result.stdout:
            return True, "GPU detected via nvidia-smi"
    except Exception:
        pass

    return False, "No CUDA GPU detected"


def _extract_filename(script: str, pattern: str) -> Optional[str]:
    """Extract filename from CHARMM script command."""
    # Common patterns: name filename, unit N name filename, etc.
    match = re.search(r'name\s+(\S+)', script, re.IGNORECASE)
    if match:
        return match.group(1)

    # Stream pattern
    match = re.search(r'stream\s+(\S+)', script, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def _validate_file_exists(script: str) -> List[Tuple[str, str]]:
    """Check if file exists for file read operations."""
    warnings_list = []

    filename = _extract_filename(script, r'name\s+(\S+)')
    if filename and not os.path.isfile(filename):
        warnings_list.append((
            f"File not found: {filename}",
            "Check file path. CHARMM may crash if bomb level >= 0"
        ))

    return warnings_list


def _validate_stream_file(script: str) -> List[Tuple[str, str]]:
    """Check if stream file exists."""
    warnings_list = []

    match = re.search(r'stream\s+(\S+)', script, re.IGNORECASE)
    if match:
        filename = match.group(1)
        if not os.path.isfile(filename):
            warnings_list.append((
                f"Stream file not found: {filename}",
                "Check file path. CHARMM will crash on missing stream file"
            ))

    return warnings_list


def _validate_gpu_operation(script: str) -> List[Tuple[str, str]]:
    """Check GPU availability for GPU operations."""
    warnings_list = []

    available, reason = _check_cuda_available()
    if not available:
        warnings_list.append((
            f"GPU operation requested but: {reason}",
            "BLaDE/DOMDEC GPU requires CUDA. Use CPU alternative or check GPU setup"
        ))

    return warnings_list


def _validate_psf_loaded(script: str) -> List[Tuple[str, str]]:
    """Check if PSF is loaded for coordinate operations."""
    warnings_list = []

    try:
        from pycharmm.lingo import get_energy_value
        natom = get_energy_value('NATOM')
        if natom is None or natom == 0:
            warnings_list.append((
                "No atoms defined (NATOM=0)",
                "Load PSF/structure before coordinate operations"
            ))
    except Exception:
        # Can't check, don't warn
        pass

    return warnings_list


def _register_builtin_validators():
    """Register built-in pattern validators."""
    global _validators
    _validators.clear()

    if _config.validate_files:
        # File read operations
        _validators.append((
            re.compile(r'\b(open\s+read|read\s+(rtf|prm|psf|coor|pdb))\b', re.IGNORECASE),
            _validate_file_exists
        ))
        # Stream operations
        _validators.append((
            re.compile(r'\bstream\b', re.IGNORECASE),
            _validate_stream_file
        ))

    if _config.validate_gpu:
        # GPU operations
        _validators.append((
            re.compile(r'\b(blade\s+on|domdec\s+gpu)\b', re.IGNORECASE),
            _validate_gpu_operation
        ))

    if _config.validate_state:
        # Operations requiring PSF
        _validators.append((
            re.compile(r'\bread\s+coor\b', re.IGNORECASE),
            _validate_psf_loaded
        ))


def register_validator(pattern: str, validator: Callable[[str], List[Tuple[str, str]]]):
    """Register a custom validator.

    Parameters
    ----------
    pattern : str
        Regex pattern to match CHARMM commands
    validator : callable
        Function that takes script string and returns list of (warning, tip) tuples

    Example
    -------
    >>> def check_my_condition(script):
    ...     if 'DANGEROUS' in script:
    ...         return [("Dangerous command detected", "Be careful!")]
    ...     return []
    >>> safeguards.register_validator(r'DANGEROUS', check_my_condition)
    """
    _validators.append((re.compile(pattern, re.IGNORECASE), validator))


# ============================================================================
# Main Hook
# ============================================================================

def _log_state():
    """Log current CHARMM state for debugging."""
    if not _logger:
        return

    try:
        from pycharmm.lingo import get_energy_value
        from pycharmm.settings import set_bomb_level, set_warn_level, set_verbosity

        natom = get_energy_value('NATOM')

        # Get current levels (set and immediately restore)
        bomb = set_bomb_level(0)
        set_bomb_level(bomb)
        warn = set_warn_level(0)
        set_warn_level(warn)
        verb = set_verbosity(5)
        set_verbosity(verb)

        _logger.debug(f"State: NATOM={natom}, BOMBLEV={bomb}, WRNLEV={warn}, PRNLEV={verb}")
    except Exception as e:
        _logger.debug(f"Could not get state: {e}")


def _pre_execute_hook(script: str):
    """Hook called before charmm_script execution.

    This function:
    1. Logs the command (for post-crash debugging)
    2. Runs validators to check prerequisites
    3. Displays warnings (but NEVER blocks execution)

    Parameters
    ----------
    script : str
        The CHARMM script about to be executed
    """
    if not _config.enabled:
        return

    if not _logger:
        _setup_logger()

    # Log command (crucial for post-crash debugging)
    if _config.show_commands:
        # Truncate very long scripts
        display_script = script.strip()
        if len(display_script) > 200:
            display_script = display_script[:200] + "..."
        display_script = display_script.replace('\n', ' | ')
        _logger.info(f"Command: {display_script}")

    # Log state before execution
    _log_state()

    # Run validators
    all_warnings = []
    for pattern, validator in _validators:
        if pattern.search(script):
            try:
                warnings_list = validator(script)
                all_warnings.extend(warnings_list)
            except Exception as e:
                _logger.debug(f"Validator error: {e}")

    # Display warnings (informative only, never blocks)
    for warning_msg, tip in all_warnings:
        _logger.warning(f"WARNING: {warning_msg}")
        if tip:
            _logger.warning(f"Tip: {tip}")

    # Flush to ensure log survives potential crash
    _flush_log()


def _post_execute_hook(script: str, status: int):
    """Hook called after charmm_script execution.

    Parameters
    ----------
    script : str
        The CHARMM script that was executed
    status : int
        Return status from eval_charmm_script
    """
    if not _config.enabled or not _logger:
        return

    if status != 1:
        _logger.warning(f"Command returned status {status} (expected 1)")
    else:
        _logger.debug(f"Command completed successfully")

    _flush_log()


# ============================================================================
# Utility Functions
# ============================================================================

def log_message(message: str, level: str = 'INFO'):
    """Log a custom message.

    Useful for adding context to the log file.

    Parameters
    ----------
    message : str
        Message to log
    level : str
        Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR')

    Example
    -------
    >>> safeguards.log_message("Starting equilibration phase")
    >>> safeguards.log_message("Temperature = 300K", level='DEBUG')
    """
    if not _config.enabled:
        return

    if not _logger:
        _setup_logger()

    log_func = getattr(_logger, level.lower(), _logger.info)
    log_func(message)
    _flush_log()


def log_checkpoint(name: str):
    """Log a checkpoint for easier debugging.

    Parameters
    ----------
    name : str
        Checkpoint name/description

    Example
    -------
    >>> safeguards.log_checkpoint("After minimization")
    >>> # ... run dynamics ...
    >>> safeguards.log_checkpoint("After 1000 steps dynamics")
    """
    if not _config.enabled:
        return

    if not _logger:
        _setup_logger()

    _logger.info("=" * 40)
    _logger.info(f"CHECKPOINT: {name}")
    _logger.info(f"Time: {datetime.now()}")
    _log_state()
    _logger.info("=" * 40)
    _flush_log()


def get_log_file() -> Optional[str]:
    """Get the current log file path."""
    return _config.log_file


def get_config() -> Dict[str, Any]:
    """Get current safeguards configuration."""
    return {
        'enabled': _config.enabled,
        'log_file': _config.log_file,
        'log_level': _config.log_level,
        'validate_files': _config.validate_files,
        'validate_gpu': _config.validate_gpu,
        'validate_state': _config.validate_state,
        'show_commands': _config.show_commands,
        'show_warnings': _config.show_warnings
    }
