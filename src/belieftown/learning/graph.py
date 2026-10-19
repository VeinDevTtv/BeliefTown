"""Permutation-invariant evidence graphs. Hidden parcel truth is never a feature."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..planner import distances
from ..interventions import heuristic_scores

FEATURE_DIM=20
GLOBAL_DIM=14
# Directed relation types. Reverse edges are explicit where they are useful.
RELATIONS=("room_edge","actor_at","at_actor","observed_by","observer_of",
           "reports_room","reported_by","root_of_message","message_of_root",
           "forwarded_from","forwarded_to","received_by","received_message",
           "sender_of","sent_by","patch_root","root_patch","patch_recipient",
           "recipient_patch","patch_sender","sender_patch")
N_RELATIONS=len(RELATIONS)
PROVENANCE_RELATIONS={7,8,9,10}


@dataclass
class GraphData:
    x: np.ndarray
    edges: np.ndarray  # [edge_count, 3] = (relation, destination, source)
    global_features: np.ndarray
    target_index: int
    destination_index: int
    y: float=0.


def encode_graph(snapshot,problem,patch,*,strip_provenance: bool=False) -> GraphData:
    nodes=[]; index={}; edges=[]
    def node(key,kind):
        idx=len(nodes); index[key]=idx; x=np.zeros(FEATURE_DIM,dtype=np.float32); x[kind]=1.; nodes.append(x); return idx
    def edge(r,src,dst): edges.append((r,dst,src))
    n=len(snapshot.scene.rooms)
    b=snapshot.belief(problem.target_actor)
    d,_=distances(n,snapshot.scene.edges)
    scale=max(1.,float(d.max()))
    target_position=snapshot.positions[problem.target_actor]
    actors=sorted(snapshot.scene.actors,key=lambda a:a.id)
    known=snapshot.known(problem.target_actor)
    roots=sorted(snapshot.roots.values(),key=lambda o:o.id)
    msgs=sorted(snapshot.messages.values(),key=lambda m:m.id)
    for room in range(n):
        j=node(("room",room),0)
        nodes[j][5]=b[room]; nodes[j][6]=float(room==problem.target_room)
        nodes[j][7]=float(room==target_position); nodes[j][8]=d[target_position,room]/scale
    for a in actors:
        j=node(("actor",a.id),1)
        nodes[j][6]=float(a.id==problem.target_actor)
        p=snapshot.belief(a.id); nodes[j][9]=float(-np.sum(p[p>0]*np.log(p[p>0])))/np.log(n)
        nodes[j][10]=float(a.id in problem.protected_actors)
        r=index[("room",snapshot.positions[a.id])]; edge(1,j,r); edge(2,r,j)
    for u,v,w in snapshot.scene.edges:
        edge(0,index[("room",u)],index[("room",v)]); edge(0,index[("room",v)],index[("room",u)])
    for root in roots:
        j=node(("root",root.id),2)
        age=snapshot.time-root.time
        nodes[j][11]=root.reliability; nodes[j][12]=age/max(1,snapshot.time)
        nodes[j][13]=float(root.id in known)
        probs=root.probabilities(n)
        nodes[j][14]=probs[problem.target_room]
        nodes[j][15]=float(root.reported_room==problem.target_room)
        nodes[j][16]=float(root.kind.startswith("inspection"))
        actor=index[("actor",root.observer)]; room=index[("room",root.reported_room)]
        edge(3,j,actor); edge(4,actor,j); edge(5,j,room); edge(6,room,j)
    for msg in msgs:
        j=node(("message",msg.id),3)
        nodes[j][12]=(snapshot.time-msg.time)/max(1,snapshot.time)
        root=index[("root",msg.root_id)]; sender=index[("actor",msg.sender)]; recipient=index[("actor",msg.recipient)]
        edge(7,root,j); edge(8,j,root); edge(11,j,recipient); edge(12,recipient,j); edge(13,sender,j); edge(14,j,sender)
    for msg in msgs:
        if msg.parent_id and ("message",msg.parent_id) in index:
            edge(9,index[("message",msg.parent_id)],index[("message",msg.id)])
            edge(10,index[("message",msg.id)],index[("message",msg.parent_id)])
    for k,delivery in enumerate(sorted(patch)):
        j=node(("delivery",k),4)
        root=snapshot.roots[delivery.root_id]
        nodes[j][13]=float(delivery.root_id in known)
        nodes[j][15]=float(root.reported_room==problem.target_room)
        nodes[j][10]=float(delivery.recipient in problem.protected_actors)
        r=index[("root",delivery.root_id)]; a=index[("actor",delivery.recipient)]; s=index[("actor",delivery.sender)]
        edge(15,r,j); edge(16,j,r); edge(17,j,a); edge(18,a,j); edge(19,s,j); edge(20,j,s)
    heuristic=heuristic_scores(snapshot,problem,[patch])[0]
    unique={x.root_id for x in patch}-known
    gs=np.asarray([len(patch)/3,n/7,len(roots)/20,len(msgs)/30,len(unique)/3,
                   b[problem.target_room],float(np.max(b)),float(problem.target_actor in problem.protected_actors),
                   np.tanh(heuristic/5),d[target_position,problem.target_room]/scale,
                   snapshot.time/20,float(snapshot.plan(problem.target_actor).destination==problem.target_room),
                   float(len(patch)>0 and not unique),sum(snapshot.roots[r].reliability for r in unique)/3],dtype=np.float32)
    if strip_provenance:
        edges=[e for e in edges if e[0] not in PROVENANCE_RELATIONS]
        # Remove direct known-root / duplicate signals as well as graph links.
        for x in nodes: x[13]=0.
        gs[4]=gs[8]=gs[12]=gs[13]=0.
    return GraphData(np.asarray(nodes,dtype=np.float32),np.asarray(edges,dtype=np.int64).reshape(-1,3),gs,
                     index[("actor",problem.target_actor)],index[("room",problem.target_room)])


def tabular_features(g: GraphData) -> np.ndarray:
    """Strong non-graph comparator: globals + feature means/maxima, no edges."""
    return np.concatenate([g.global_features,g.x.mean(0),g.x.max(0)]).astype(np.float32)
