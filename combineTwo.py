import matplotlib.pyplot as plt

def parse_file(file_path):
    rounds = []
    accuracies = []
    losses = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("INFO:root:Round"):
                parts = line.split(", ")
                round_info = parts[0].split(" ")[1]
                accuracy_info = parts[1].split(": ")[1].strip('%')
                loss_info = parts[2].split(": ")[1]

                rounds.append(int(round_info.split('/')[0]))
                accuracies.append(float(accuracy_info))
                losses.append(float(loss_info))
    
    return rounds, accuracies, losses

# File paths
file1 = "Logincr.log"
file2 = "random.log"

# Parse data
rounds1, accuracies1, losses1 = parse_file(file1)
rounds2, accuracies2, losses2 = parse_file(file2)

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot accuracies on the left y-axis
ax1.plot(rounds1, accuracies1, label="Proposed Accuracy", color="blue")
ax1.plot(rounds2, accuracies2, label="Random Accuracy", color="green")
ax1.set_xlabel("Rounds")
ax1.set_ylabel("Accuracy (%)", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")
ax1.grid(True)

# Create a second y-axis for losses
ax2 = ax1.twinx()
ax2.plot(rounds1, losses1, label="Proposed Loss", color="red")
ax2.plot(rounds2, losses2, label="Random Loss", color="orange")
ax2.set_ylabel("Loss", color="red")
ax2.tick_params(axis='y', labelcolor="red")

# Combine legends from both axes and move them outside the plot
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

# Title and save
plt.title("Test Accuracy and Loss Over Rounds")
plt.tight_layout()
plt.savefig('result.png')