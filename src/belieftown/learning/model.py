"""Small relational GNN and feature-matched MLP, trained on exact solver labels."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch import nn
from .graph import FEATURE_DIM,GLOBAL_DIM,N_RELATIONS,GraphData,encode_graph,tabular_features


def configure(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def collate(graphs: list[GraphData]):
    b=len(graphs); n=max(len(g.x) for g in graphs)
    x=torch.zeros((b,n,FEATURE_DIM),dtype=torch.float32)
    adj=torch.zeros((b,N_RELATIONS,n,n),dtype=torch.float32)
    mask=torch.zeros((b,n),dtype=torch.float32)
    glob=torch.from_numpy(np.stack([g.global_features for g in graphs]))
    targets=torch.tensor([g.target_index for g in graphs],dtype=torch.long)
    dests=torch.tensor([g.destination_index for g in graphs],dtype=torch.long)
    y=torch.tensor([g.y for g in graphs],dtype=torch.float32)
    tabs=torch.from_numpy(np.stack([tabular_features(g) for g in graphs]))
    for k,g in enumerate(graphs):
        m=len(g.x); x[k,:m]=torch.from_numpy(g.x); mask[k,:m]=1.
        if len(g.edges):
            e=g.edges; adj[k,e[:,0],e[:,1],e[:,2]]=1.
    adj=adj/adj.sum(-1,keepdim=True).clamp_min(1.)
    return x,adj,mask,glob,targets,dests,tabs,y


class GraphRanker(nn.Module):
    def __init__(self,hidden: int=32,layers: int=2):
        super().__init__(); self.hidden=hidden; self.layers=layers
        self.input=nn.Linear(FEATURE_DIM,hidden)
        self.self_layers=nn.ModuleList([nn.Linear(hidden,hidden) for _ in range(layers)])
        self.relations=nn.ParameterList([nn.Parameter(torch.empty(N_RELATIONS,hidden,hidden)) for _ in range(layers)])
        for w in self.relations: nn.init.xavier_uniform_(w)
        self.head=nn.Sequential(nn.Linear(hidden*3+GLOBAL_DIM,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,batch):
        x,adj,mask,glob,target,dest,tab,_=batch
        h=torch.relu(self.input(x))*mask.unsqueeze(-1)
        for self_layer,w in zip(self.self_layers,self.relations):
            messages=torch.einsum('brij,bjh->brih',adj,h)
            aggregated=torch.einsum('brih,rhk->bik',messages,w)
            h=torch.relu(self_layer(h)+aggregated)*mask.unsqueeze(-1)
        pooled=h.sum(1)/mask.sum(1,keepdim=True).clamp_min(1.)
        idx=torch.arange(len(h))
        return self.head(torch.cat([pooled,h[idx,target],h[idx,dest],glob],dim=-1)).squeeze(-1)


class MLPRanker(nn.Module):
    def __init__(self,hidden: int=64):
        super().__init__(); self.hidden=hidden
        self.net=nn.Sequential(nn.Linear(GLOBAL_DIM+FEATURE_DIM*2,hidden),nn.ReLU(),nn.Linear(hidden,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,batch): return self.net(batch[6]).squeeze(-1)


class Predictor:
    def __init__(self,model,kind: str,metadata: dict|None=None):
        self.model=model.eval(); self.kind=kind; self.metadata=metadata or {}
    def score(self,snapshot,problem,patches) -> list[float]:
        if not patches: return []
        values=[]
        with torch.inference_mode():
            for start in range(0,len(patches),64):
                graphs=[encode_graph(snapshot,problem,p,strip_provenance=self.kind=="graph_no_provenance") for p in patches[start:start+64]]
                values.extend(self.model(collate(graphs)).cpu().tolist())
        return values
    def save(self,path: str|Path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"state_dict":self.model.state_dict(),"kind":self.kind,"metadata":self.metadata,
                    "schema_version":1,"hidden":getattr(self.model,"hidden",32),
                    "layers":getattr(self.model,"layers",2),
                    "mlp_hidden":getattr(self.model,"hidden",64) if self.kind=="mlp" else 64},path)
    @classmethod
    def load(cls,path: str|Path):
        # weights_only=True avoids arbitrary Python execution on deserialization.
        obj=torch.load(path,map_location="cpu",weights_only=True)
        if obj.get("schema_version")!=1: raise ValueError("Unsupported checkpoint version")
        kind=obj["kind"]
        if kind not in ("graph","mlp","graph_no_provenance"): raise ValueError("Unknown checkpoint kind")
        model=MLPRanker(obj.get("mlp_hidden",64)) if kind=="mlp" else GraphRanker(obj["hidden"],obj["layers"])
        model.load_state_dict(obj["state_dict"])
        return cls(model,kind,obj.get("metadata",{}))
