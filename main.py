import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import logging
from model import get_model
from option import args_parser
from set_seed import set_seed
from fedPro import fedPro
from fedProRandom import fedProRandom
from PoC import PoC
from OCS import OCS
from PNCS import PNCS
from fedPromid import fedPromid


args = args_parser()
print("Arguments:")
for arg, value in vars(args).items():
    print(f"{arg}: {value}")

# 根據資料集設定類別數
if args.dataset == "CIFAR10":
    num_classes = 10
elif args.dataset == "CIFAR100":
    num_classes = 100
elif args.dataset == "TinyImageNet":
    num_classes = 200
elif args.dataset == "SVHN":
    num_classes = 10
else:
    raise ValueError(f"Unsupported dataset: {args.dataset}")

model_type = get_model(args.model_type, num_classes=num_classes, random_seed=args.random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(args)

logging.basicConfig(level=logging.INFO, filename=args.log_file, filemode='a')

if __name__ == "__main__":
    if args.method == "proposed":
        fedPro(args, model_type)
    elif args.method == "random":
        fedProRandom(args, model_type)
    elif args.method == "PoC":
        PoC(args, model_type)
    elif args.method == "OCS":
        OCS(args, model_type)
    elif args.method == "PNCS":
        PNCS(args, model_type)
    elif args.method == "fedPromid":
        fedPromid(args, model_type)
    else:
        raise ValueError(f"Method {args.method} is not supported. Please choose 'proposed' or 'random'.")