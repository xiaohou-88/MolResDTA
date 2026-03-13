import os
import builtins
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

from model import Classifier, SMILESModel, FASTAModel
from process_data import DTAData, CHARPROTLEN, MACCSLEN
from process_data import collate_fn
from train_and_test import test


def build_dataloaders(dataset: str, batch_size: int, device):
    if dataset == "davis":
        df_test = pd.read_csv("./data/Davis/Davis_test.csv")
        max_smiles_len = 85
        max_fasta_len = 1000
    elif dataset == "kiba":
        df_test = pd.read_csv("./data/KIBA/kiba_test.csv")
        max_smiles_len = 100
        max_fasta_len = 1000
    elif dataset == "Bind":
        df_test = pd.read_csv("./data/BindingDB/BindingDB_test.csv")
        max_smiles_len = 100
        max_fasta_len = 1000
    else:
        raise ValueError(f"Unknown DATASET: {dataset}")

    fasta_test = list(df_test["target_sequence"])
    smiles_test = list(df_test["iso_smiles"])
    label_test = list(df_test["affinity"])

    test_set = DTAData(smiles_test, fasta_test, label_test, device, max_smiles_len, max_fasta_len)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return test_loader


def build_model(device):
    smiles_model = SMILESModel(char_set_len=MACCSLEN)
    fasta_model = FASTAModel(char_set_len=CHARPROTLEN + 1)
    model = Classifier(smiles_model, fasta_model).to(device)
    return model


def load_checkpoint(model, ckpt_path: str, map_location):
    ckpt = torch.load(ckpt_path, map_location=map_location)

   
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="davis", choices=["davis", "kiba", "Bind"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--ckpt", type=str, required=True, help="path to best .pth checkpoint")
    parser.add_argument("--pred_out", type=str, default=None, help="output prediction txt file")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataloader / model
    test_loader = build_dataloaders(args.dataset, args.batch_size, device)
    model = build_model(device)

    # load weights
    ckpt_meta = load_checkpoint(model, args.ckpt, map_location=device)

    # loss
    loss_fn = torch.nn.MSELoss()

    # redirect prediction.txt -> custom file (optional)
    pred_out = args.pred_out
    if pred_out is None:
        os.makedirs("predictions", exist_ok=True)
        base = os.path.splitext(os.path.basename(args.ckpt))[0]
        pred_out = os.path.join("predictions", f"{args.dataset}_{base}_prediction.txt")

    orig_open = builtins.open

    def open_redirect(file, mode="r", *open_args, **open_kwargs):
        if file == "prediction.txt":
            file = pred_out
        return orig_open(file, mode, *open_args, **open_kwargs)

    builtins.open = open_redirect
    try:
        test_loss, test_ci, test_rm2 = test(test_loader, model, loss_fn)
    finally:
        builtins.open = orig_open

    print("===== Test Only Result =====")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.ckpt}")
    if isinstance(ckpt_meta, dict):
        for k in ["epoch", "seed", "test_loss", "test_ci", "test_rm2"]:
            if k in ckpt_meta:
                print(f"Checkpoint[{k}]: {ckpt_meta[k]}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test CI:   {test_ci:.6f}")
    print(f"Test RM2:  {test_rm2:.6f}")
    print(f"Predictions written to: {pred_out}")


if __name__ == "__main__":
    main()
