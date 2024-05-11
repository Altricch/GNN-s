import psutil
from time import sleep, time
import os

def monitor_resources(interval=1, duration=60):
    """
    Monitor and print system CPU, GPU, and memory utilization.
    Args:
    - interval: Sampling interval in seconds.
    - duration: Total duration to monitor in seconds.
    """
    start_time = time()
    while True:
        # Check the time elapsed
        if (time() - start_time) > duration:
            break

        # CPU utilization
        #process_names = [proc.name() for proc in psutil.process_iter()]

        # Get the current process ID
        pid = os.getpid()
        # Get the psutil Process object using the PID
        current_process = psutil.Process(pid)
        num_cpus = psutil.cpu_count()

        #interval=interval
        cpu_usage = current_process.cpu_percent(interval=interval)/ num_cpus
        cpu_time = current_process.cpu_times()
        memory = psutil.virtual_memory().total

        # Memory utilization
        memory_current = current_process.memory_info()
        memory_percent = current_process.memory_percent("vms")
        memory_guaglione = current_process.memory_full_info().vms  / memory
        #memory_usage = round(memory.used / memory.total * 100,2)
        elapsed_time = round(time() - start_time,2)
        
        # GPU utilization and memory usage
        # gpus = GPUtil.getGPUs()
        # for gpu in gpus:
        #     gpu_usage = gpu.load * 100
        #     gpu_memory_usage = gpu.memoryUsed / gpu.memoryTotal * 100
        #     print(f"GPU ID: {gpu.id}, GPU Usage: {gpu_usage}%, GPU Mem: {gpu_memory_usage}%")

        # Print CPU and Memory usage
        print(f'''\n[MONITOR]
              Process Name: {current_process.name()}
              CPU Usage: {cpu_usage}%
              CPU time: {cpu_time}s
              Memory: {memory_current}
              Memory percent: {memory_percent}%
              Memory guaglione: {memory_guaglione}%
              Elapsed Time {elapsed_time}s\n''')
        sleep(interval)

# Example usage
#monitor_resources(interval=1, duration=300)  # Monitor for 5 minutes