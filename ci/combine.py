import matplotlib.pyplot as plt

def parse_log_file(file_path):
    epochs = []
    accuracies = []
    with open(file_path, 'r') as f:
        for line in f:
            if "Round" in line and "Test Accuracy" in line:
                parts = line.strip().split(',')
                epoch = int(parts[0].split('Round')[1].split('/')[0].strip())
                accuracy = float(parts[1].split(':')[1].replace('%', '').strip())
                epochs.append(epoch)
                accuracies.append(accuracy)
    return epochs, accuracies

# 讀取數據
epochs_cs, accuracies_cs = parse_log_file('cs.log')
epochs_random, accuracies_random = parse_log_file('random.log')
epochs_topdatasize, accuracies_topdatasize = parse_log_file('topDataSize.log')

plt.figure(figsize=(12, 6))
plt.rcParams.update({'font.size': 18})  # 更新全局字體大小

# 繪製比較曲線
plt.plot(epochs_cs, accuracies_cs, label='Cosine Similarity', color='#ff0000')
plt.plot(epochs_random, accuracies_random, label='Random', color='#101fef')
plt.plot(epochs_topdatasize, accuracies_topdatasize, label='Top Data Size', color='#32931a')

# 標註最後的準確率
def annotate_last_compact(epochs_list, accuracies_list, colors, labels, gap=6):
    last_points = []
    for epochs, accuracies, color, label in zip(epochs_list, accuracies_list, colors, labels):
        if epochs and accuracies:
            last_points.append((accuracies[-1], epochs[-1], color, label))
    last_points.sort(reverse=True, key=lambda x: x[0])
    xlim = plt.gca().get_xlim()
    for idx, (acc, epoch, color, label) in enumerate(last_points):
        x_annotate = min(epoch + 2, xlim[1] - 5)
        y_annotate = last_points[0][0] - idx * (gap + 1)  # 增加間隔以讓 box 之間有更多空間
        plt.annotate(f'{label}: {acc:.2f}%', 
                     xy=(epoch, acc), 
                     xytext=(x_annotate-250, y_annotate-40),
                     textcoords='data',
                     fontsize=16,
                     color=color,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.8),
                     ha='left', va='center')

epochs_list = [epochs_cs, epochs_random, epochs_topdatasize]
accuracies_list = [accuracies_cs, accuracies_random, accuracies_topdatasize]
colors = ["#ff0000", "#101fef", "#32931a"]
labels = ['Cosine Similarity', 'Random', 'Top Data Size']

annotate_last_compact(epochs_list, accuracies_list, colors, labels, gap=3)

plt.title('Preliminary Class-Increment Experiment')
plt.xlabel('Round', fontsize=18)
plt.ylabel('Accuracy (%)', fontsize=18)
plt.legend(loc='lower right', fontsize=16, ncol=3)
plt.grid(True)
plt.savefig('accuracy_comparison.png')