import torch
import numpy as np

def set_seed(args):
    # 設置隨機種子以確保可重現性
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    # 確保 DataLoader 的隨機性可控
    generator = torch.Generator()
    generator.manual_seed(args.random_seed)

    # 強制使用確定性操作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False