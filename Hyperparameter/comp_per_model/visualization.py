import matplotlib.pyplot as plt
import json

with open("Hyperparameter/comp_per_model/GCN.py_config.json", "r") as file:
    json_data = file.read()


name = "GCN"
path = "Hyperparameter/comp_per_model/"

# Load JSON data into Python dictionary
data = json.loads(json_data)

# Extract keys (number of convolutional layers), accuracies, learning rates, and hidden dimensions
conv_layers = [int(key) for key in data.keys()]
accuracies = [entry["ACC"] for entry in data.values()]
learning_rates = [entry["LR"] for entry in data.values()]
hidden_dims = [entry["Hidden"] for entry in data.values()]

# Plot the data
fig, ax1 = plt.subplots(figsize=(16,9))

# Plot Accuracy
color = "tab:blue"
ax1.set_xlabel("Number of Convolutional Layers")
ax1.set_ylabel("Accuracy", color=color)
ax1.plot(conv_layers, accuracies, marker="o", linestyle="-", color=color)
ax1.tick_params(axis="y", labelcolor=color)

# Create a second y-axis for Hidden Dimension
ax2 = ax1.twinx()

# Plot Hidden Dimension
color = "tab:green"
ax2.set_ylabel("Hidden Dimension", color=color)
ax2.plot(conv_layers, hidden_dims, marker="^", linestyle="-.", color=color)
ax2.tick_params(axis="y", labelcolor=color)

# Create a third y-axis for Learning Rate
ax3 = ax1.twinx()

# Plot Learning Rate
color = "tab:red"
ax3.spines["right"].set_position(("outward", 60))  # Adjust the position of this axis
ax3.set_ylabel("Learning Rate", color=color)
ax3.plot(conv_layers, learning_rates, marker="s", linestyle="--", color=color)
ax3.tick_params(axis="y", labelcolor=color)

fig.tight_layout(
    rect=[0, 0.03, 1, 0.97]
)  # Adjust the layout to accommodate the title on top
plt.title(
    name
    + ": Accuracy, Learning Rate, and Hidden Dimension vs. Number of Convolutional Layers",
    loc="center",
)
# Save the plot as an image
plt.savefig(path + name + ".png", bbox_inches="tight")
plt.show()
