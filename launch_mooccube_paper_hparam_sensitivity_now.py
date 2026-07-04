from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parent
    script = repo / "run_mooccube_paper_hparam_sensitivity_serial.ps1"
    out_dir = repo / "outputs" / "content_delta_pop5" / "course_hparam_sensitivity_e60_3seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{random.randint(1000, 9999)}_{random.randint(1000, 9999)}"
    stdout_path = out_dir / f"course_hparam_sensitivity_watcher_{run_id}_stdout.log"
    stderr_path = out_dir / f"course_hparam_sensitivity_watcher_{run_id}_stderr.log"
    paths_path = out_dir / "course_hparam_sensitivity_worker_latest_paths.txt"
    paths_path.write_text(f"STDOUT={stdout_path}\nSTDERR={stderr_path}\n", encoding="utf-8")

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

    env = os.environ.copy()
    system_root = env.get("SystemRoot", r"C:\Windows")
    env["Path"] = ";".join(
        [
            rf"{system_root}\System32",
            system_root,
            rf"{system_root}\System32\WindowsPowerShell\v1.0",
            r"D:\Anaconda3\envs\zw",
            r"D:\Anaconda3\envs\zw\Scripts",
            str(repo),
        ]
    )
    env.pop("PATH", None)

    powershell = rf"{system_root}\System32\WindowsPowerShell\v1.0\powershell.exe"
    args = [
        powershell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Repo",
        str(repo),
        "-PythonRunner",
        r"D:\Anaconda3\envs\zw\python.exe",
        "-NoAutoWait",
        "-SkipGpuWait",
        "-PollSeconds",
        "300",
        "-MinFreeGpuMiB",
        "9000",
    ]
    with stdout_path.open("w", encoding="utf-8", newline="") as stdout, stderr_path.open(
        "w", encoding="utf-8", newline=""
    ) as stderr:
        proc = subprocess.Popen(
            args,
            cwd=str(repo),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
            close_fds=False,
        )
    print(f"PID={proc.pid}")
    print(f"PATHS={paths_path}")


if __name__ == "__main__":
    main()
