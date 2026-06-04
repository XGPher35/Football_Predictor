import argparse
import os
import pickle
import time
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader

from nn_football.data import read_epl_csvs
from nn_football.preprocess import build_dataset
from nn_football.model import BiLSTMAttentionMTL
from nn_football.eval import compute_class_metrics, compute_regression_metrics


def make_dataloaders(
    homes,
    aways,
    stats,
    outcome,
    goals,
    stats_tgt,
    dates,
    batch_size=64,
    val_frac=0.2,
):
    N = homes.shape[0]
    order = np.argsort(dates)
    split = int(N * (1 - val_frac))
    train_idx = order[:split]
    val_idx = order[split:]

    def to_loader(idxs):
        t_h = torch.from_numpy(homes[idxs]).float()
        t_a = torch.from_numpy(aways[idxs]).float()
        t_s = torch.from_numpy(stats[idxs]).float()
        t_out = torch.from_numpy(outcome[idxs]).long()
        t_goals = torch.from_numpy(goals[idxs]).float()
        t_stats = torch.from_numpy(stats_tgt[idxs]).float()
        ds = TensorDataset(t_h, t_a, t_s, t_out, t_goals, t_stats)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    return to_loader(train_idx), to_loader(val_idx)


def train(args):
    df = read_epl_csvs(args.data_dir)
    homes, aways, stat_feats, outcome, goals, stats_tgt, scalers, dates = build_dataset(
        df, seq_len=args.seq_len
    )
    if homes.shape[0] == 0:
        print(
            "No examples with sufficient history. Reduce seq_len or provide more data."
        )
        return

    train_loader, val_loader = make_dataloaders(
        homes,
        aways,
        stat_feats,
        outcome,
        goals,
        stats_tgt,
        dates,
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttentionMTL(
        feature_dim=homes.shape[2],
        embed_dim=args.embed_dim,
        hid_dim=args.hid_dim,
        lstm_layers=args.lstm_layers,
        seq_dropout=args.dropout,
        static_dim=stat_feats.shape[1],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss()
    huber = torch.nn.SmoothL1Loss(beta=1.0)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")

    with open(os.path.join(args.save_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        for h, a, s, out_lbl, goals_lbl, stats_lbl in train_loader:
            h = h.to(device)
            a = a.to(device)
            s = s.to(device)
            out_lbl = out_lbl.to(device)
            goals_lbl = goals_lbl.to(device)
            stats_lbl = stats_lbl.to(device)

            preds = model(h, a, s)
            loss_out = ce(preds["outcome"], out_lbl)
            loss_goals = huber(preds["goals"], goals_lbl)
            loss_stats = huber(preds["stats"], stats_lbl)
            loss = args.l1 * loss_out + args.l2 * loss_goals + args.l3 * loss_stats

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item()) * h.size(0)

        train_loss = running / len(train_loader.dataset)
        # validation
        model.eval()
        import numpy as np

        all_probs = []
        all_labels = []
        all_goals_pred = []
        all_goals_t = []
        all_stats_pred = []
        all_stats_t = []
        with torch.no_grad():
            for h, a, s, out_lbl, goals_lbl, stats_lbl in val_loader:
                h = h.to(device)
                a = a.to(device)
                s = s.to(device)
                out_lbl = out_lbl.to(device)
                goals_lbl = goals_lbl.to(device)
                stats_lbl = stats_lbl.to(device)

                preds = model(h, a, s)
                probs = torch.softmax(preds["outcome"], dim=-1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(out_lbl.cpu().numpy())
                all_goals_pred.append(preds["goals"].cpu().numpy())
                all_goals_t.append(goals_lbl.cpu().numpy())
                all_stats_pred.append(preds["stats"].cpu().numpy())
                all_stats_t.append(stats_lbl.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_goals_pred = np.concatenate(all_goals_pred, axis=0)
        all_goals_t = np.concatenate(all_goals_t, axis=0)
        all_stats_pred = np.concatenate(all_stats_pred, axis=0)
        all_stats_t = np.concatenate(all_stats_t, axis=0)

        cls_metrics = compute_class_metrics(all_probs, all_labels)
        goals_metrics = compute_regression_metrics(all_goals_pred, all_goals_t)
        stats_metrics = compute_regression_metrics(all_stats_pred, all_stats_t)

        val_logloss = cls_metrics["log_loss"]
        print(
            f"Epoch {epoch}/{args.epochs} — train_loss: {train_loss:.4f} — val_logloss: {val_logloss:.4f} — val_f1: {cls_metrics['f1_macro']:.4f} — goals_rmse: {goals_metrics['rmse']:.3f}"
        )

        if val_logloss < best_val:
            best_val = val_logloss
            save_checkpoint(
                model,
                opt,
                epoch,
                args,
                cls_metrics,
                goals_metrics,
                stats_metrics,
                args.save_dir,
                feature_dim=homes.shape[2],
                best=True,
            )

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                model,
                opt,
                epoch,
                args,
                cls_metrics,
                goals_metrics,
                stats_metrics,
                args.save_dir,
                feature_dim=homes.shape[2],
                best=False,
            )


def save_checkpoint(
    model,
    opt,
    epoch,
    args,
    cls_metrics,
    goals_metrics,
    stats_metrics,
    save_dir,
    feature_dim,
    best=False,
):
    name = "model_best.pt" if best else f"model_epoch_{epoch}.pt"
    path = os.path.join(save_dir, name)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "feature_dim": feature_dim,
        "config": vars(args),
        "metrics": {
            "class": cls_metrics,
            "goals": goals_metrics,
            "stats": stats_metrics,
        },
    }
    torch.save(payload, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/epl")
    p.add_argument("--seq_len", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=32)
    p.add_argument("--hid_dim", type=int, default=64)
    p.add_argument("--lstm_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l1", type=float, default=1.0)
    p.add_argument("--l2", type=float, default=0.5)
    p.add_argument("--l3", type=float, default=0.1)
    p.add_argument("--save_dir", type=str, default="artifacts")
    p.add_argument("--save_every", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
