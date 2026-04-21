#!/usr/bin/env python3
import time
import subprocess
import os
from datetime import datetime

LOG_FILE = "metrics.log"

def get_cpu_temp():
    try:
        temps = []
        for d in os.listdir("/sys/class/thermal/"):
            if d.startswith("thermal_zone"):
                with open(f"/sys/class/thermal/{d}/temp") as f:
                    # typical values are millidegrees C
                    val = float(f.read().strip())
                    if val > 1000: val /= 1000.0
                    temps.append(val)
        return max(temps) if temps else 0.0
    except Exception:
        return 0.0

def get_gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True
        ).strip().split(', ')
        return float(out[0]), float(out[1]), float(out[2])
    except Exception:
        return 0.0, 0.0, 0.0

if __name__ == "__main__":
    print(f"Starting Safe Hardware Telemetry... Logging to {LOG_FILE}\n")
    print("="*70)
    print(f"{'Time':<12} | {'CPU Temp':<10} | {'GPU Temp':<10} | {'VRAM (MB)':<10} | {'Status':<10}")
    print("="*70)
    
    with open(LOG_FILE, "a") as f:
        f.write("Time,CPU_Temp_C,GPU_Temp_C,VRAM_MB,GPU_Util_Pct\n")
        
        try:
            while True:
                cpu_t = get_cpu_temp()
                gpu_t, vram, gpu_util = get_gpu_stats()
                
                status = "SAFE"
                color = "\033[92m" # Green
                
                if cpu_t > 85.0 or gpu_t > 82.0:
                    status = "DANGER! " + ("CPU" if cpu_t > 85 else "GPU")
                    color = "\033[91m" # Red
                elif cpu_t > 75.0 or gpu_t > 75.0:
                    status = "WARNING"
                    color = "\033[93m" # Yellow
                    
                reset = "\033[0m"
                now = datetime.now().strftime("%H:%M:%S")
                
                line = f"{now:<12} | {cpu_t:<8.1f} C | {gpu_t:<8.1f} C | {vram:<10.0f} | {color}{status:<15}{reset}"
                print(f"\r{line}", end="", flush=True)
                
                f.write(f"{now},{cpu_t:.1f},{gpu_t:.1f},{vram:.0f},{gpu_util:.1f}\n")
                f.flush()
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
