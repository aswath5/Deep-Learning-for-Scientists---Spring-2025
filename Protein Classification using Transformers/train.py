# train.py
import torch
from lightning import Trainer
from torch.utils.data import DataLoader
from train_model import ESMClassifier
from dataset import PAFDataset, collate_fn
import pandas as pd
import pickle


def train_and_predict():
    print("Using", "CUDA" if torch.cuda.is_available() else "CPU")

    # Load data
    data_path = "/ocean/projects/cis240109p/shanmuga/data"
    train_df = pd.read_csv(f"{data_path}/train_data.csv")
    val_df = pd.read_csv(f"{data_path}/val_data.csv")
    test_df = pd.read_csv(f"{data_path}/test_data.csv")
    label_list = pickle.load(open(f"{data_path}/selected_families.pkl", "rb"))
    label_map = {label: idx for idx, label in enumerate(label_list)}
    idx2label = {v: k for k, v in label_map.items()}

    # Datasets
    train_ds = PAFDataset(train_df, label_map)
    val_ds = PAFDataset(val_df, label_map)
    test_ds = PAFDataset(test_df)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Model
    model = ESMClassifier(n_classes=len(label_map))
    trainer = Trainer(max_epochs=10, accelerator="auto")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    model.plot_accuracy()

    # Prediction
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    preds = []

    for batch in test_loader:
        encoded = model.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.embedder(**encoded)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            logits = model.classifier(cls_emb)
            predicted = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(predicted)

    pred_labels = [idx2label[i] for i in preds]
    submission = pd.DataFrame({
        "sequence_name": test_df["sequence_name"],
        "family_id": pred_labels
    })
    submission.to_csv("submission.csv", index=False)
    print("submission.csv saved!")
