import platform
import subprocess
import sys

import matplotlib
import numpy as np
import PIL
import scipy
import skimage


def sysctl(key):
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def machine_lines():
    ram = sysctl("hw.memsize")
    ram_text = f"{int(ram) / 2**30:.0f} GiB" if ram.isdigit() else "unknown"
    performance = sysctl("hw.perflevel0.physicalcpu")
    efficiency = sysctl("hw.perflevel1.physicalcpu")
    return [
        f"cpu         {sysctl('machdep.cpu.brand_string')}, {sysctl('hw.ncpu')} cores "
        f"({performance} performance + {efficiency} efficiency)",
        f"ram         {ram_text}",
        f"os          {platform.system()} {platform.mac_ver()[0] or platform.release()} "
        f"({platform.machine()})",
        f"python      {sys.version.split()[0]}",
        f"numpy       {np.__version__}",
        f"scipy       {scipy.__version__} (comparisons the handout asks for, sine integral for a plot)",
        f"skimage     {skimage.__version__} (unit-test checks only)",
        f"matplotlib  {matplotlib.__version__}",
        f"pillow      {PIL.__version__} (image io only)",
    ]


def write_environment(path):
    lines = machine_lines()
    path.write_text("\n".join(lines) + "\n")
    return lines
