import matplotlib.pyplot as plt
import os
import glob

def parse_log_file(file_path):
    epochs = []
    accuracies = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if "Round" in line and "Test Accuracy" in line:
                    parts = line.strip().split(',')
                    epoch = int(parts[0].split('Round')[1].split('/')[0].strip())
                    accuracy = float(parts[1].split(':')[1].replace('%', '').strip())
                    epochs.append(epoch)
                    accuracies.append(accuracy)
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"Error reading {file_path}: {e}")
    return epochs, accuracies

def get_clean_label(filename):
    """從檔案名稱生成清晰的標籤"""
    name = filename.replace('.log', '')
    return name

# ====== 檔案管理設定 ======
# 你可以在這裡控制要比較的檔案
current_dir = os.path.dirname(os.path.abspath(__file__))
# 垂直虛線設定
VERTICAL_LINES = [
    11, 12, 35, 50, 51
]
# 包裝 plt.savefig / plt.show，確保在輸出或顯示圖形前畫上垂直線
_orig_savefig = plt.savefig
_orig_show = plt.show

def _draw_vlines():
    ax = plt.gca()
    for x in VERTICAL_LINES:
        ax.axvline(x=x, color='black', linestyle='--', linewidth=1)

def _savefig_override(*args, **kwargs):
    _draw_vlines()
    return _orig_savefig(*args, **kwargs)

def _show_override(*args, **kwargs):
    _draw_vlines()
    return _orig_show(*args, **kwargs)

plt.savefig = _savefig_override
plt.show = _show_override
# 檔案分組設定
AVG_GROUPS = {
    # 'Keep Signal': [
    #     'keepSignal(zeta=0.5)_42.log',
    #     'keepSignal(zeta=0.5)_43.log',
    #     'keepSignal(zeta=0.5)_44.log',
    # ],
    'FedUE': [
        'random_42.log',
        'random_45.log',
        'random_46.log',
    ],
}

SELECTED_FILES = [
    'Data Size.log',
    'Cosine Similarity.log',
    # 'topDataSize.log',
    # 'cs.log',
    # 'random.log',
    # 平均曲線不直接列入，後面處理
]

# 選項2: 排除特定檔案
EXCLUDE_FILES = [
    # 'some_unwanted_file.log',  # 取消註解來排除特定檔案
]

# 選項3: 只包含符合模式的檔案
INCLUDE_PATTERNS = [
    # 'fixed*.log',     # 所有 fixed 開頭的檔案
    # 'dynamic*.log',   # 所有 dynamic 開頭的檔案
    # 'proposed*.log',  # 所有 proposed 開頭的檔案
]

# 選項4: 自動讀取所有檔案（原始行為）
AUTO_READ_ALL = False  # 設為 True 來自動讀取所有 .log 檔案

# ====== 檔案讀取邏輯 ======
if SELECTED_FILES:
    # 使用指定的檔案列表
    log_files = [os.path.join(current_dir, f) for f in SELECTED_FILES if os.path.exists(os.path.join(current_dir, f))]
    print(f"Using selected files: {len(log_files)}/{len(SELECTED_FILES)} files found")
    missing_files = [f for f in SELECTED_FILES if not os.path.exists(os.path.join(current_dir, f))]
    if missing_files:
        print(f"Missing files: {missing_files}")
        
elif INCLUDE_PATTERNS:
    # 使用模式匹配
    log_files = []
    for pattern in INCLUDE_PATTERNS:
        pattern_files = glob.glob(os.path.join(current_dir, pattern))
        log_files.extend(pattern_files)
    log_files = list(set(log_files))  # 去除重複
    log_files.sort()
    print(f"Using pattern matching: found {len(log_files)} files")
    
elif AUTO_READ_ALL:
    # 自動讀取所有檔案
    log_files = glob.glob(os.path.join(current_dir, '*.log'))
    log_files.sort()
    print(f"Auto-reading all .log files: found {len(log_files)} files")
    
else:
    # 預設：讀取所有檔案但排除指定的
    log_files = glob.glob(os.path.join(current_dir, '*.log'))
    if EXCLUDE_FILES:
        log_files = [f for f in log_files if os.path.basename(f) not in EXCLUDE_FILES]
    log_files.sort()
    print(f"Reading all files (excluding {len(EXCLUDE_FILES)} files): found {len(log_files)} files")

print(f"Found {len(log_files)} log files:")
for file in log_files:
    print(f"  - {os.path.basename(file)}")


# 讀取所有數據（單檔案）
all_data = []
for log_file in log_files:
    filename = os.path.basename(log_file)
    epochs, accuracies = parse_log_file(log_file)
    if epochs and accuracies:
        label = get_clean_label(filename)
        # 跳過要做平均的檔案
        if filename not in sum(AVG_GROUPS.values(), []):
            all_data.append({
                'filename': filename,
                'label': label,
                'epochs': epochs,
                'accuracies': accuracies
            })
            print(f"Successfully loaded {filename}: {len(epochs)} data points")
    else:
        print(f"Failed to load data from {filename}")

# 讀取並平均分組檔案
for label, files in AVG_GROUPS.items():
    group_epochs = []
    group_accs = []
    for fname in files:
        fpath = os.path.join(current_dir, fname)
        if os.path.exists(fpath):
            epochs, accs = parse_log_file(fpath)
            if epochs and accs:
                group_epochs.append(epochs)
                group_accs.append(accs)
                print(f"Loaded for average: {fname} ({len(epochs)} points)")
        else:
            print(f"Missing for average: {fname}")
    # 取平均（以最短長度為基準）

    if group_epochs and group_accs:
        min_len = min(len(e) for e in group_epochs)
        avg_epochs = group_epochs[0][:min_len]
        avg_accs = [float(sum(accs[i] for accs in group_accs)) / len(group_accs) for i in range(min_len)]
        # 印出最後一輪的 accuracy
        print(f"Average method [{label}] last round: epoch={avg_epochs[-1]}, accuracy={avg_accs[-1]:.2f}%")
        all_data.append({
            'filename': '+'.join(files),
            'label': label,
            'epochs': avg_epochs,
            'accuracies': avg_accs
        })
        print(f"Added average curve for {label}: {len(avg_epochs)} points")

print(f"\nLoaded {len(all_data)} datasets for comparison.")

plt.figure(figsize=(8, 6))

# 定義顏色調色盤
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7a6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    '#f7b6d2', '#c49c94', '#e377c2',
]

# 繪製所有曲線
for i, data in enumerate(all_data):
    color = colors[i % len(colors)]
    plt.plot(data['epochs'], data['accuracies'], 
             label=data['label'], color=color, linewidth=2)
    plt.tick_params(axis='both', which='major', labelsize=20)

# 提取資料用於標註功能
epochs_list = [data['epochs'] for data in all_data]
accuracies_list = [data['accuracies'] for data in all_data]
plot_colors = [colors[i % len(colors)] for i in range(len(all_data))]
labels = [data['label'] for data in all_data]

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
                     xytext=(x_annotate-60, y_annotate-40),
                     textcoords='data',
                     fontsize=9,
                     color=color,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.8),
                     ha='left', va='center')

# annotate_last_compact(epochs_list, accuracies_list, plot_colors, labels, gap=3)

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
                     xytext=(target_epoch - 280, y_annotate - 40),
                     textcoords='data',
                     fontsize=9,
                     color=color,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.8),
                     ha='left', va='center')

# 你可以根據需要修改這裡的 target_epoch，取消註解來顯示特定 epoch 的比較
# target_epoch = 100
# annotate_epoch_accuracy_sorted(epochs_list, accuracies_list, plot_colors, labels, target_epoch, gap=3)
# plt.axvline(x=target_epoch, color='gray', linestyle='--', linewidth=1, alpha=0.5)
# plt.text(target_epoch, plt.ylim()[0] - 3, str(target_epoch), color='gray', fontsize=9, ha='center', va='bottom', alpha=0.7)

# plt.title(f'Test Accuracy vs. Communication Round test (Dirichlet α=0.3)')
# plt.xlabel('Round', fontsize=18, fontweight='bold')
plt.xlabel('Round', fontsize=22)
plt.ylabel('Accuracy (%)', fontsize=22)
plt.legend(loc='lower right', fontsize=20, ncol=1)
plt.grid(True)
plt.tight_layout()
plt.savefig('CNN_2.0_0.1_data_cs.png')
plt.show()

# ====== 使用說明 ======
"""
檔案管理選項說明：

1. SELECTED_FILES (推薦)：
   - 明確指定要比較的檔案
   - 可以控制比較的順序
   - 自動檢查檔案是否存在

2. EXCLUDE_FILES：
   - 排除特定不想比較的檔案
   - 適合大部分檔案都要比較的情況

3. INCLUDE_PATTERNS：
   - 使用通配符模式選擇檔案
   - 例如：'fixed*.log', 'dynamic*.log'

4. AUTO_READ_ALL：
   - 自動讀取所有 .log 檔案
   - 原始行為，適合快速檢視

使用範例：

# 只比較固定值實驗
SELECTED_FILES = [
    'fixed(zeta=0).log',
    'fixed(zeta=0.33).log', 
    'fixed(zeta=0.66).log',
    'fixed(zeta=1).log'
]

# 只比較動態策略
INCLUDE_PATTERNS = ['dynamic*.log']

# 排除某些檔案
EXCLUDE_FILES = ['old_experiment.log', 'test.log']
"""