import matplotlib.pyplot as plt
import json
import matplotlib.ticker as ticker

# Load the data from the JSON files
data = {}

adgn_path = 'Train/comp_per_model/ADGN_Message.py_requirements.json'
gat_path = 'Train/comp_per_model/GAT.py_requirements.json'
gcn_path = 'Train/comp_per_model/GCN.py_requirements.json'
ggnn_path = 'Train/comp_per_model/GGNN.py_requirements.json'

paths = [adgn_path, gat_path, gcn_path, ggnn_path]

for i in paths:
    filename = i 
    model = i.split('/')[-1].split('.')[0]
    with open(filename, "r") as file:
        data[model] = json.load(file)

print(data["GCN"]['Epoch 0'])

# Extract CPU, Memory, and Time data for each model
epochs = list(data["GCN"].keys())
# breakpoint()
cpu_data = {model: [data[model][epoch]["CPU"] for epoch in epochs] for model in data}
memory_data = {model: [data[model][epoch]["Memory"] for epoch in epochs] for model in data}
time_data = {model: [data[model][epoch]["Time"] for epoch in epochs] for model in data}

# Plot and save CPU comparison
plt.figure(figsize=(10, 6))
for model in cpu_data:
    plt.plot(list(epochs), cpu_data[model], label=model)
plt.xlabel("Epoch")
plt.ylabel("CPU Usage")
plt.title("CPU Comparison")
plt.legend()
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5)) 
plt.tight_layout()  # Adjust layout to prevent overlap
plt.grid(True)
plt.savefig("Train/comp_per_model/cpu_comparison.png")
plt.show()

# Plot and save Memory comparison
plt.figure(figsize=(10, 6))
for model in memory_data:
    plt.plot(list(epochs), memory_data[model], label=model)
plt.xlabel("Epoch")
plt.ylabel("Memory Usage")
plt.title("Memory Comparison")
plt.legend()
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5)) 
plt.tight_layout()  # Adjust layout to prevent overlap
plt.grid(True)
plt.savefig("Train/comp_per_model/memory_comparison.png")
plt.show()

# Plot and save Time taken comparison
plt.figure(figsize=(10, 6))
for model in time_data:
    plt.plot(list(epochs), time_data[model], label=model)
plt.xlabel("Epoch")
plt.ylabel("Time Taken")
plt.title("Time Taken Comparison")
plt.legend()
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5)) 
plt.tight_layout()  # Adjust layout to prevent overlap
plt.grid(True)
plt.savefig("Train/comp_per_model/time_comparison.png")
plt.show()