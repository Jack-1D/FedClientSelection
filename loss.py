import torch
import torch.nn.functional as F

# 計算KL散度
def compute_kl_divergence(model1, model2, data_loader, device):
    model1.eval()
    model2.eval()
    kl_div = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            batch_size = data.size(0)
            
            # 獲取模型輸出（softmax機率）
            output1 = F.softmax(model1(data), dim=1) + 1e-10  # 避免log(0)
            output2 = F.softmax(model2(data), dim=1) + 1e-10
            
            # 計算KL散度
            kl = torch.sum(output1 * torch.log(output1 / output2), dim=1).mean()
            kl_div += kl.item() * batch_size
            total_samples += batch_size
    
    return kl_div / total_samples if total_samples > 0 else float('inf')