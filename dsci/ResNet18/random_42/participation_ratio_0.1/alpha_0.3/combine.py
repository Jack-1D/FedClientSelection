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
epochs_ocs, accuracies_ocs = parse_log_file('OCS.log')
epochs_poc, accuracies_poc = parse_log_file('PoC(d=6, K=30, m=3).log')
epochs_random, accuracies_random = parse_log_file('random.log')
epochs_topdatasize, accuracies_topdatasize = parse_log_file('topDataSize.log')
epochs_pncs, accuracies_pncs = parse_log_file('PNCS.log')
epochs_keepsignal_0533, accuracies_keepsignal_0533 = parse_log_file('keepSignal(alpha=0.66, beta=0.66, t=0.0533).log')
epochs_proposed_0533, accuracies_proposed_0533 = parse_log_file('proposed(alpha=0.66, beta=0.66, t=0.0533).log')
epochs_keepsignal_new, accuracies_keepsignal_new = parse_log_file('keepSignal(alpha=0.66, beta=0.66, t=0.1333333).log')
epochs_proposed_new, accuracies_proposed_new = parse_log_file('proposed(alpha=0.66, beta=0.66, t=0.1333333).log')

plt.figure(figsize=(12, 6))

plt.plot(epochs_cs, accuracies_cs, label='CS', color='#1f77b4')
# plt.plot(epochs_keepsignal, accuracies_keepsignal, label='KeepSignal (t=0.2)', color='#ff7f0e')
plt.plot(epochs_ocs, accuracies_ocs, label='OCS', color='#2ca02c')
plt.plot(epochs_poc, accuracies_poc, label='PoC (d=6, K=30, m=3)', color='#d62728')
# plt.plot(epochs_proposed, accuracies_proposed, label='Proposed (t=0.08)', color='#9467bd')
plt.plot(epochs_random, accuracies_random, label='Random', color='#8c564b')
plt.plot(epochs_topdatasize, accuracies_topdatasize, label='TopDataSize', color='#e377c2')
plt.plot(epochs_pncs, accuracies_pncs, label='PNCS', color='#17becf')
plt.plot(epochs_keepsignal_0533, accuracies_keepsignal_0533, label='KeepSignal (α=0.66, β=0.66, t=0.0533)', color='#bcbd22')
plt.plot(epochs_proposed_0533, accuracies_proposed_0533, label='Proposed (α=0.66, β=0.66, t=0.0533)', color='#7f7f7f')
plt.plot(epochs_keepsignal_new, accuracies_keepsignal_new, label='KeepSignal (α=0.66, β=0.66, t=0.1333)', color='#ff7f0e')
plt.plot(epochs_proposed_new, accuracies_proposed_new, label='Proposed (α=0.66, β=0.66, t=0.1333)', color='#9467bd')

def annotate_best(epochs, accuracies, label_color, idx):
    if not epochs or not accuracies:
        return
    max_accuracy = max(accuracies)
    max_index = accuracies.index(max_accuracy)
    max_epoch = epochs[max_index]
    # 所有標注初始高度一樣，偶數個往上一點
    base_offset = 67
    extra_offset = 2 if idx % 2 == 0 else 0
    plt.annotate(f'{max_accuracy:.2f}%', 
                 xy=(max_epoch, max_accuracy), 
                 xytext=(idx * 50 + 30, base_offset + extra_offset),
                 textcoords='data',
                 arrowprops=dict(arrowstyle='->', color=label_color, lw=1.5),
                 color=label_color)

# annotate_best(epochs_cs, accuracies_cs, '#1f77b4', 0)
# annotate_best(epochs_keepsignal, accuracies_keepsignal, '#ff7f0e', 1)
# annotate_best(epochs_ocs, accuracies_ocs, '#2ca02c', 2)
# annotate_best(epochs_poc, accuracies_poc, '#d62728', 3)
# annotate_best(epochs_proposed, accuracies_proposed, '#9467bd', 4)
# annotate_best(epochs_random, accuracies_random, '#8c564b', 5)
# annotate_best(epochs_topdatasize, accuracies_topdatasize, '#e377c2', 6)

def annotate_last_compact(epochs_list, accuracies_list, colors, labels, gap=6):
    last_points = []
    for epochs, accuracies, color, label in zip(epochs_list, accuracies_list, colors, labels):
        if epochs and accuracies:
            last_points.append((accuracies[-1], epochs[-1], color, label))
    last_points.sort(reverse=True, key=lambda x: x[0])
    xlim = plt.gca().get_xlim()
    for idx, (acc, epoch, color, label) in enumerate(last_points):
        x_annotate = min(epoch + 2, xlim[1] - 5)
        y_annotate = last_points[0][0] - idx * gap  # 固定間隔
        plt.annotate(f'{label}: {acc:.2f}%', 
                     xy=(epoch, acc), 
                     xytext=(x_annotate-90, y_annotate-35),
                     textcoords='data',
                     fontsize=9,
                     color=color,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.8),
                     ha='left', va='center')

epochs_list = [
    epochs_cs, epochs_ocs,
    epochs_poc, epochs_random, epochs_topdatasize, epochs_pncs,
    epochs_keepsignal_0533, epochs_proposed_0533, epochs_keepsignal_new, epochs_proposed_new
]
accuracies_list = [
    accuracies_cs, accuracies_ocs,
    accuracies_poc, accuracies_random, accuracies_topdatasize, accuracies_pncs,
    accuracies_keepsignal_0533, accuracies_proposed_0533, accuracies_keepsignal_new, accuracies_proposed_new
]
colors = [
    '#1f77b4', '#2ca02c',
    '#d62728', '#8c564b', '#e377c2', '#17becf',
    '#bcbd22', '#7f7f7f', '#ff7f0e', '#9467bd'
]
labels = [
    'CS', 'OCS',
    'PoC (d=6, K=30, m=3)', 'Random', 'TopDataSize', 'PNCS',
    'KeepSignal (α=0.66, β=0.66, t=0.0533)', 'Proposed (α=0.66, β=0.66, t=0.0533)',
    'KeepSignal (α=0.66, β=0.66, t=0.1333)', 'Proposed (α=0.66, β=0.66, t=0.1333)'
]

annotate_last_compact(epochs_list, accuracies_list, colors, labels, gap=3)

def annotate_epoch_accuracy_sorted(epochs_list, accuracies_list, colors, labels, target_epoch, gap=6):
    epoch_points = []
    for epochs, accuracies, color, label in zip(epochs_list, accuracies_list, colors, labels):
        if target_epoch in epochs:
            idx = epochs.index(target_epoch)
            acc = accuracies[idx]
            epoch_points.append((acc, color, label))
    epoch_points.sort(reverse=True, key=lambda x: x[0])
    for i, (acc, color, label) in enumerate(epoch_points):
        y_annotate = epoch_points[0][0] - i * gap  # 固定間隔
        plt.annotate(f'{label} @ {target_epoch}: {acc:.2f}%',
                     xy=(target_epoch, acc),
                     xytext=(target_epoch - 280, y_annotate - 35),
                     textcoords='data',
                     fontsize=9,
                     color=color,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.8),
                     ha='left', va='center')

# 你可以根據需要修改這裡的 target_epoch
target_epoch = 480
annotate_epoch_accuracy_sorted(epochs_list, accuracies_list, colors, labels, target_epoch, gap=3)
plt.axvline(x=target_epoch, color='gray', linestyle='--', linewidth=1, alpha=0.5)
plt.text(target_epoch, plt.ylim()[0] - 3, str(target_epoch), color='gray', fontsize=9, ha='center', va='bottom', alpha=0.7)

plt.title('Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend(loc='lower left', fontsize='small', ncol=3)
plt.grid(True)
plt.savefig('accuracy_comparison.png')
