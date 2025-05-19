import matplotlib.pyplot as plt
import os

os.makedirs('result_plot', exist_ok=True)

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
colors = [
    "blue", "orange", "green", "red", "purple", 
    "brown", "pink", "gray", "olive", "cyan", 
    "magenta", "yellow", "teal"
]
# File paths
file1 = "Log_proposed.log"
file2 = "Log_random.log"
file3 = "Log_wo_new_class.log"
file4 = "Log_wo_cs.log"
file5 = "Log_wo_data_rank.log"
file6 = "Log_wo_kl.log"
file7 = "Log_wo_new_class_and_data_rank.log"
file8 = "Log_wo_cs_and_kl.log"
file9 = "Log_only_new_class.log"
file10 = "Log_only_cs.log"
file11 = "Log_only_data_rank.log"
file12 = "Log_only_kl.log"
file13 = "Log_no_signal.log"

# Parse data
rounds1, accuracies1, losses1 = parse_file(file1)
rounds2, accuracies2, losses2 = parse_file(file2)
rounds3, accuracies3, losses3 = parse_file(file3)
rounds4, accuracies4, losses4 = parse_file(file4)
rounds5, accuracies5, losses5 = parse_file(file5)
rounds6, accuracies6, losses6 = parse_file(file6)
rounds7, accuracies7, losses7 = parse_file(file7)
rounds8, accuracies8, losses8 = parse_file(file8)
rounds9, accuracies9, losses9 = parse_file(file9)
rounds10, accuracies10, losses10 = parse_file(file10)
rounds11, accuracies11, losses11 = parse_file(file11)
rounds12, accuracies12, losses12 = parse_file(file12)
rounds13, accuracies13, losses13 = parse_file(file13)
# Plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(rounds1, accuracies1, label=file1.replace("Log_", "").replace(".log", ""), color=colors[0])
plt.plot(rounds2, accuracies2, label=file2.replace("Log_", "").replace(".log", ""), color=colors[1])
plt.plot(rounds3, accuracies3, label=file3.replace("Log_", "").replace(".log", ""), color=colors[2])
plt.plot(rounds4, accuracies4, label=file4.replace("Log_", "").replace(".log", ""), color=colors[3])
plt.plot(rounds5, accuracies5, label=file5.replace("Log_", "").replace(".log", ""), color=colors[4])
plt.plot(rounds6, accuracies6, label=file6.replace("Log_", "").replace(".log", ""), color=colors[5])
plt.plot(rounds7, accuracies7, label=file7.replace("Log_", "").replace(".log", ""), color=colors[6])
plt.plot(rounds8, accuracies8, label=file8.replace("Log_", "").replace(".log", ""), color=colors[7])
plt.plot(rounds9, accuracies9, label=file9.replace("Log_", "").replace(".log", ""), color=colors[8])
plt.plot(rounds10, accuracies10, label=file10.replace("Log_", "").replace(".log", ""), color=colors[9])
plt.plot(rounds11, accuracies11, label=file11.replace("Log_", "").replace(".log", ""), color=colors[10])
plt.plot(rounds12, accuracies12, label=file12.replace("Log_", "").replace(".log", ""), color=colors[11])
plt.plot(rounds13, accuracies13, label=file13.replace("Log_", "").replace(".log", ""), color=colors[12])
plt.xlabel("Rounds")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy Over Rounds")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
plt.grid(True)
plt.tight_layout()
plt.savefig('result_plot/accuracy_plot.png')
plt.show()

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(rounds1, losses1, label=file1.replace("Log_", "").replace(".log", ""), color=colors[0])
plt.plot(rounds2, losses2, label=file2.replace("Log_", "").replace(".log", ""), color=colors[1])
plt.plot(rounds3, losses3, label=file3.replace("Log_", "").replace(".log", ""), color=colors[2])
plt.plot(rounds4, losses4, label=file4.replace("Log_", "").replace(".log", ""), color=colors[3])
plt.plot(rounds5, losses5, label=file5.replace("Log_", "").replace(".log", ""), color=colors[4])
plt.plot(rounds6, losses6, label=file6.replace("Log_", "").replace(".log", ""), color=colors[5])
plt.plot(rounds7, losses7, label=file7.replace("Log_", "").replace(".log", ""), color=colors[6])
plt.plot(rounds8, losses8, label=file8.replace("Log_", "").replace(".log", ""), color=colors[7])
plt.plot(rounds9, losses9, label=file9.replace("Log_", "").replace(".log", ""), color=colors[8])
plt.plot(rounds10, losses10, label=file10.replace("Log_", "").replace(".log", ""), color=colors[9])
plt.plot(rounds11, losses11, label=file11.replace("Log_", "").replace(".log", ""), color=colors[10])
plt.plot(rounds12, losses12, label=file12.replace("Log_", "").replace(".log", ""), color=colors[11])
plt.plot(rounds13, losses13, label=file13.replace("Log_", "").replace(".log", ""), color=colors[12])
plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.title("Test Loss Over Rounds")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
plt.grid(True)
plt.tight_layout()
plt.savefig('result_plot/loss_plot.png')
plt.show()
