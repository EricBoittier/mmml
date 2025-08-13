"""Module for converting TensorBoard logs to Polars DataFrames."""

import struct
from pathlib import Path
from typing import Dict, List, Tuple, Union

import polars as pl
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python.summary.summary_iterator import summary_iterator

# Example usage
"""
log_dir = Path("/path/to/tensorboard/logs")
df = process_tensorboard_logs(log_dir)
print(df.head())
Example usage:
"""


def read_tensor(value) -> Tuple[str, float]:
    """
    Read a tensor value from TensorBoard event and convert it to a float.

    Args:
        value: TensorBoard event value containing tensor data

    Returns:
        Tuple of (tag name, float value)
    """
    binary_content = value.tensor.tensor_content
    try:
        float_value = struct.unpack("f", binary_content)[0]  # 'f' is for 32-bit float
        return value.tag, float_value
    except struct.error:
        return value.tag, 0.0


def tensorboard_to_polars(logdir: Union[str, Path], epoch: int = 0) -> pl.DataFrame:
    """
    Convert a single TensorBoard log file to a Polars DataFrame.

    Args:
        logdir: Path to TensorBoard log fie
        epoch: Epoch number to associate with this log file

    Returns:
        Polars DataFrame containing the TensorBoard metrics
    """
    data: Dict[str, Union[float, int, str]] = {}

    for event in summary_iterator(str(logdir)):
        for value in event.summary.value:
            k, v = read_tensor(value)
            data[k] = v

    data["epoch"] = epoch
    data["log"] = str(logdir)

    if data:
        return pl.DataFrame(data)
    return pl.DataFrame()


def process_tensorboard_logs(log_dir: Union[str, Path]) -> pl.DataFrame:
    """
    Process all TensorBoard log files in a directory and combine them into a single DataFrame.

    Args:
        log_dir: Directory containing TensorBoard log files

    Returns:
        Combined Polars DataFrame with all metrics
    """
    path = Path(log_dir)
    if not path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    files = sorted(path.glob("*"))
    if not files:
        raise ValueError(f"No log files found in directory: {log_dir}")

    dataframes = [
        tensorboard_to_polars(str(file), epoch=i) for i, file in enumerate(files)
    ]

    return pl.concat(dataframes)
