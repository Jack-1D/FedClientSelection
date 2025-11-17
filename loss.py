import torch
import torch.nn.functional as F
import numpy as np
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 計算KL散度
def compute_kl_divergence(model1, model2, data_loader, device):
    model1.to(device)
    model2.to(device)
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

            del data, output1, output2, kl
            torch.cuda.empty_cache()
    
    model1.cpu()
    model2.cpu()
    
    return kl_div / total_samples if total_samples > 0 else float('inf')

def client_count_cs_score(model_type, global_model, local_state_dict, dataLoader, mu=0.5):
    """
    計算客戶端的 CS Score，結合 extractor 和 predictor 的 cosine similarity
    :param model_type: 客戶端模型類型
    :param global_model: 全局模型
    :param local_state_dict: 客戶端模型的狀態字典
    :param dataLoader: 客戶端數據加載器
    :param mu: 合併 extractor 和 predictor 的權重超參數
    :return: CS Score
    """
    local_model = copy.deepcopy(model_type)
    local_model.load_state_dict(local_state_dict)
    local_model.to(device)
    global_model.to(device)

    global_model.eval()
    local_model.eval()
    total_cs = 0.0

    with torch.no_grad():
        for data, _ in dataLoader:
            data = data.to(device)
            
            # Extractor feature maps
            global_feature_map = global_model.extractor(data)
            local_feature_map = local_model.extractor(data)
            global_feature_map_flat = global_feature_map.view(global_feature_map.size(0), -1)
            local_feature_map_flat = local_feature_map.view(local_feature_map.size(0), -1)
            
            # Predictor outputs
            global_predictor_output = global_model.predictor(global_feature_map_flat)
            local_predictor_output = local_model.predictor(local_feature_map_flat)
            
            # Cosine similarity for extractor
            extractor_sim = F.cosine_similarity(global_feature_map_flat, local_feature_map_flat, dim=1)
            
            # Cosine similarity for predictor
            predictor_sim = F.cosine_similarity(global_predictor_output, local_predictor_output, dim=1)
            
            # Combine extractor and predictor similarity using mu
            combined_sim = mu * extractor_sim + (1 - mu) * predictor_sim
            
            total_cs += combined_sim.cpu().sum().item()  # 把 batch 裡所有 sample 的 loss 加總

    avg_cs_score = total_cs / len(dataLoader.dataset)

    del local_model
    del global_feature_map, local_feature_map, global_feature_map_flat, local_feature_map_flat
    del global_predictor_output, local_predictor_output
    torch.cuda.empty_cache()
    global_model.cpu()

    return avg_cs_score if not np.isnan(avg_cs_score) else 1e-8

def normalized_shannon_entropy(class_counts):
    """
    計算類別分佈的正規化香農熵
    :param class_counts: 類別計數的列表或數組
    :return: 正規化香農熵
    """
    class_counts = np.array(class_counts)
    total = class_counts.sum()
    if total == 0:
        return 0.0  # no data

    p = class_counts / total
    p = p[p > 0]  # 避免 log(0)
    entropy = -np.sum(p * np.log(p)) / np.log(len(class_counts))
    return entropy