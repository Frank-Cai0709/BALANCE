import torch
import torch.nn as nn
import argparse
import pickle
from models.Prototype import PrototypeNet, explain
from datasets import split_dataset
from trainer import train_model, test_model
import hydra


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BreakHis 40X', choices=['BreakHis 40X','BreakHis 100X','BreakHis 200X','BreakHis 400X'])
    parser.add_argument('--method', type=str, default='Prototype', choices=['Prototype'])
    parser.add_argument('--explain', action='store_true', help='whether to run prototype explanation')
    args = parser.parse_args()
    
    # Model and training parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with hydra.initialize(config_path="configs", version_base='1.3'):
        cfg = hydra.compose(config_name="global", overrides=[f"method={args.method}", f"dataset={args.dataset}"])
    benign_dir = f'./dataset/{args.dataset}/benign'
    malignant_dir = f'./dataset/{args.dataset}/malignant'

    model_path = f'./checkpoints/{args.dataset}.pth'

    train_loader,test_loader = split_dataset(benign_dir, malignant_dir)

    model = PrototypeNet(num_classes=cfg.dataset.num_classes, num_prototypes_per_class=10).to(device)

    if args.explain:
        print("Running prototype-based explanation...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        with open(f"./proto_meta/{args.dataset}_{args.method}.pkl", "rb") as f:
            model.proto_meta = pickle.load(f)
        save_path = f"./explain_outputs/{args.dataset}"
        for img_batch, label_batch, img_path_batch in test_loader: #one batch
            for i in range(len(img_batch)):
                img_tensor = img_batch[i].unsqueeze(0).to(device)
                img_path = img_path_batch[i]
                label = label_batch[i].item() 
                explain(
                    model,
                    img_tensor,
                    img_path=img_path,
                    label=label,
                    device=device,
                    save_path = save_path,
                    count = i 
                )
            exit()
    
    train_model(model, cfg, train_loader,  model_path, device)
    test_model(model, cfg, test_loader, model_path, device)

