"""Deterministic CPU training with scene-held-out validation and saved histories."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .data import load_cases,training_graphs
from .model import GraphRanker,MLPRanker,Predictor,configure,collate


def train_model(data_dir: str|Path,output: str|Path,*,kind: str="graph",seed: int=0,epochs: int=18,batch_size: int=64,learning_rate: float=0.002):
    if kind not in ("graph","mlp","graph_no_provenance"): raise ValueError("Unknown model kind")
    if epochs<1 or batch_size<1 or learning_rate<=0: raise ValueError("Invalid training hyperparameters")
    configure(seed); start=time.perf_counter(); data_dir=Path(data_dir)
    train=training_graphs(load_cases(data_dir/'train.jsonl'),strip_provenance=kind=="graph_no_provenance")
    val=training_graphs(load_cases(data_dir/'validation.jsonl'),strip_provenance=kind=="graph_no_provenance")
    if not train or not val: raise ValueError("Nonempty train and validation splits are required")
    model=MLPRanker() if kind=="mlp" else GraphRanker()
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    loss_fn=nn.BCEWithLogitsLoss()
    rng=np.random.default_rng(seed); history=[]; best_loss=float('inf'); best_state=None; best_epoch=0
    # Validation batches can be reused: no dropout or random augmentation.
    val_batches=[collate(val[j:j+batch_size]) for j in range(0,len(val),batch_size)]
    for epoch in range(1,epochs+1):
        model.train(); loss_sum=0.; order=rng.permutation(len(train))
        for j in range(0,len(order),batch_size):
            gs=[train[i] for i in order[j:j+batch_size]]; batch=collate(gs)
            optimizer.zero_grad(set_to_none=True); logits=model(batch); loss=loss_fn(logits,batch[-1])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
            loss_sum+=float(loss.detach())*len(gs)
        model.eval(); vloss=0.; correct=0
        with torch.inference_mode():
            for batch in val_batches:
                logits=model(batch); y=batch[-1]
                vloss+=float(loss_fn(logits,y))*len(y); correct+=int(((logits>0)==y.bool()).sum())
        vloss/=len(val)
        history.append({"epoch":epoch,"train_bce":loss_sum/len(train),"validation_bce":vloss,"validation_accuracy":correct/len(val)})
        if vloss<best_loss:
            best_loss=vloss; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch
    model.load_state_dict(best_state)
    metadata={"seed":seed,"epochs":epochs,"best_epoch":best_epoch,"batch_size":batch_size,"learning_rate":learning_rate,
              "train_samples":len(train),"validation_samples":len(val),"validation_bce":best_loss,
              "selection":"lowest validation BCE only; test and OOD splits never used in optimization",
              "torch_version":str(torch.__version__),"training_seconds":time.perf_counter()-start,
              "parameter_count":sum(p.numel() for p in model.parameters()),
              "manifest_sha256":hashlib.sha256((data_dir/'manifest.json').read_bytes()).hexdigest(),"history":history}
    predictor=Predictor(model,kind,metadata); predictor.save(output)
    Path(output).with_suffix('.json').write_text(json.dumps({"kind":kind,**metadata},indent=2)+'\n', encoding="utf-8")
    return predictor
