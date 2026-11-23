"""Pre-flight guard for spike S3: abort if the GPU or disk is too full (see S3.md)."""

import shutil
import subprocess
import sys

MIN_FREE_VRAM_MIB = 4096
MIN_FREE_DISK_GIB = 5.0


def check(label: str = "") -> tuple[int, float]:
    free = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
        ).split()[0]
    )
    disk = shutil.disk_usage("/").free / 2**30
    print(f"[guard{(' ' + label) if label else ''}] free VRAM {free} MiB, free disk {disk:.1f} GiB")
    if free < MIN_FREE_VRAM_MIB:
        sys.exit(f"ABORT: free VRAM {free} MiB < {MIN_FREE_VRAM_MIB}; GPU busy (LLM inference?)")
    if disk < MIN_FREE_DISK_GIB:
        sys.exit(f"ABORT: free disk {disk:.1f} GiB < {MIN_FREE_DISK_GIB}")
    return free, disk
