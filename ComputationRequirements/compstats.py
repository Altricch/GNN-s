from time import time
import psutil
import json

# This file is used to compute the CPU, Memory and Time consumption taken into account different models with different number of convolution layers
# Store the data in a json file

def computeStats(
    conf_start_time,
    conv,
    lr,
    lay,
    current_process,
    num_cpus,
    requirements,
    dump_immediately=False,
    file_name="test",
    counter="",
):

    cpu_usage = round(current_process.cpu_percent(interval=None) / num_cpus, 2)

    memory = psutil.virtual_memory().total
    memory_allocated = current_process.memory_full_info().uss
    memory_perc = round((memory_allocated / memory) * 100, 2)

    conf_elapsed_time = round(time() - conf_start_time, 2)
    # QuickFix
    if conv == -1:
        return

    requirements[f"{counter} Conv {conv}, LR {lr}, Hidden {lay}"] = {
        "CPU": cpu_usage,
        "Memory": memory_perc,
        "Time": conf_elapsed_time,
    }
    if dump_immediately:
        with open(file_name, "w") as json_file:
            json.dump(requirements, json_file, indent=4)


def computeStatsModel(
    conf_start_time,
    conv,
    lr,
    lay,
    current_process,
    num_cpus,
    requirements,
    model,
    dump_immediately=False,
    file_name="test",
    counter="",
):

    cpu_usage = round(current_process.cpu_percent(interval=None) / num_cpus, 2)

    memory = psutil.virtual_memory().total
    memory_allocated = current_process.memory_full_info().uss
    memory_perc = round((memory_allocated / memory) * 100, 2)

    conf_elapsed_time = round(time() - conf_start_time, 2)
    
    if conv == -1:
        return

    requirements[f"{counter} Conv {conv}, LR {lr}, Hidden {lay}, Model {model}"] = {
        "CPU": cpu_usage,
        "Memory": memory_perc,
        "Time": conf_elapsed_time,
    }
    
    if dump_immediately:
        with open(file_name, "w") as json_file:
            json.dump(requirements, json_file, indent=4)
