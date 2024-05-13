from time import time
import psutil


def computeStats(conf_start_time, epoch, current_process, num_cpus, requirements):

    cpu_usage = round(current_process.cpu_percent(interval=None) / num_cpus, 2)

    memory = psutil.virtual_memory().total
    memory_allocated = current_process.memory_full_info().uss
    memory_perc = round((memory_allocated / memory) * 100, 2)

    conf_elapsed_time = round(time() - conf_start_time, 2)

    requirements[f"Epoch {epoch}"] = {
        "CPU": cpu_usage,
        "Memory": memory_perc,
        "Time": conf_elapsed_time,
    }
