"""Training adapter for pinned upstream STTN-CP; no sequence crosses a split."""
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .sttn_architecture import STTNCP, combined_loss, count_trainable_parameters


def make_windows(features, labels, row_ids, length):
    if len(features) < length:
        raise ValueError("Split shorter than sequence length")
    windows = np.lib.stride_tricks.sliding_window_view(features, length, axis=0).transpose(0, 2, 1)
    return np.ascontiguousarray(windows, dtype=np.float32), labels[length-1:], row_ids[length-1:]


def score(model, features, batch_size, device):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start+batch_size]).to(device)
            values.append(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(values)


def train_sttn(features, labels, groups, output, seed=42, n_jobs=2, quick=False, device="auto", max_epochs=None):
    output = Path(output)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(n_jobs)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    config = dict(input_dim=features.shape[1], seq_len=4 if quick else 8,
        embed_dim=16 if quick else 128, n_blocks=1 if quick else 3,
        n_heads=4, mlp_dim=32 if quick else 256, proj_dim=8 if quick else 64)
    batch_size, epochs, patience = (32, 2, 2) if quick else (128, 50, 10)
    if max_epochs is not None:
        if max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        epochs = max_epochs
    scaler = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                       ("scale", MinMaxScaler())])
    scaler.fit(features[groups["train"]])
    datasets = {}
    for name, ids in groups.items():
        x = scaler.transform(features[ids])
        datasets[name] = make_windows(x, labels[ids], ids, config["seq_len"])
        if len(np.unique(datasets[name][1])) != 2:
            raise ValueError(f"{name} windows lack one class")
    train_x, train_y, _ = datasets["train"]
    val_x, val_y, _ = datasets["validation"]
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0)
    if not len(loader):
        raise ValueError("Training data smaller than a batch")
    model = STTNCP(**config).to(device)
    weights = len(train_y)/(2*np.bincount(train_y, minlength=2))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history, best_state, best_f1, stale, best_epoch = [], None, -1, 0, 0
    start = time.perf_counter()
    progress = output / "progress.txt"
    for epoch in range(1, epochs+1):
        model.train()
        sums = np.zeros(3)
        for step, (xb, yb) in enumerate(loader, 1):
            xb, yb = xb.to(device), yb.to(device)
            view1, view2 = xb + torch.randn_like(xb)*0.02, xb + torch.randn_like(xb)*0.02
            z1 = model.project(model.forward_backbone(view1))
            z2 = model.project(model.forward_backbone(view2))
            loss, ce, contrast = combined_loss(model(xb), yb, z1, z2, criterion,
                                                lambda_contrastive=0.5, temperature=0.07)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sums += [loss.item(), ce.item(), contrast.item()]
            if step % 20 == 0 or step == len(loader):
                with progress.open("a") as file:
                    file.write(f"{(epoch-1)*len(loader)+step}/{epochs*len(loader)} sttn_cp_2026 epoch={epoch}\n")
        scheduler.step()
        val_scores = score(model, val_x, batch_size, device)
        val_f1 = float(f1_score(val_y, val_scores >= 0.5, zero_division=0))
        row = dict(epoch=epoch, loss=float(sums[0]/len(loader)),
            classification_loss=float(sums[1]/len(loader)), contrastive_loss=float(sums[2]/len(loader)),
            validation_f1_at_0_5=val_f1)
        history.append(row)
        with (output / "history.jsonl").open("a") as file:
            import json
            file.write(json.dumps(row)+"\n")
        print(f"STTN-CP epoch {epoch}/{epochs}: val_F1@0.5={val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_state = copy.deepcopy(model.state_dict())
            best_f1, best_epoch, stale = val_f1, epoch, 0
            torch.save({"model_config": config,
                "state_dict": {key: value.cpu() for key, value in best_state.items()},
                "epoch": epoch}, output / "best_model.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    predictions = {name: score(model, data[0], batch_size, device)
                   for name, data in datasets.items() if name != "train"}
    import joblib
    joblib.dump(scaler, output / "preprocessing.joblib")
    info = dict(model_config=config, trainable_parameters=count_trainable_parameters(model),
        batch_size=batch_size, max_epochs=epochs, patience=patience, completed_epochs=len(history),
        best_epoch=best_epoch, best_validation_f1_at_0_5=best_f1, device=str(device),
        class_weights=weights.tolist(), optimizer="Adam", learning_rate=1e-3,
        scheduler="CosineAnnealingLR", noise_std=0.02, lambda_contrastive=0.5, temperature=0.07,
        elapsed_seconds=time.perf_counter()-start, deterministic_algorithms_warn_only=True,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == "cuda" else 0)
    return predictions, datasets, info
