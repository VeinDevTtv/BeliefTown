/* BeliefTown UI: every live decision comes from the Python application service. */
'use strict';
const boot=window.BELIEFTOWN_BOOT||{};
const $=id=>document.getElementById(id);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clone=x=>JSON.parse(JSON.stringify(x));
let state=null, tick=4, selected='courier', lastSearch=null, timer=null, offlineFamily=boot.defaultFamily||'missed_relocation';
const pretty=s=>String(s).replace(/_/g,' ').replace(/^./,x=>x.toUpperCase());
function notice(text,error=false){const n=$('notice');n.hidden=!text;n.textContent=text;n.classList.toggle('error',error);}
async function api(action,args={}){
  if(boot.offline)throw new Error('This is a recorded, verified showcase. Run the local Python app for new interventions.');
  const response=await fetch('/api/'+action,{method:'POST',headers:{'Content-Type':'application/json','X-BeliefTown-Token':boot.token},body:JSON.stringify(args)});
  const data=await response.json();if(!response.ok)throw new Error(data.error||'Request failed');return data;
}
async function perform(fn){
  document.body.classList.add('loading');
  try{await fn();}catch(err){notice(err.message,true);}finally{document.body.classList.remove('loading');}
}
function options(id,items,value){const el=$(id);el.innerHTML=items.map(x=>`<option value="${esc(x[0])}">${esc(x[1])}</option>`).join('');if(value!=null)el.value=String(value);}
function setState(value,{fresh=false}={}){
  state=value;
  if(fresh){tick=state.scene.search_start;selected=state.scene.actors[0].id;lastSearch=null;}
  tick=Math.min(tick,state.scene.horizon);
  $('timeline').max=state.scene.horizon;$('timeline').value=tick;
  $('description').textContent=state.scene.description;
  if(fresh){
    options('scenario',state.families.map(f=>[f,pretty(f)]),state.scene.family);
    options('target-actor',state.scene.actors.map(a=>[a.id,a.name]),selected);
    options('target-room',state.scene.rooms.map((n,i)=>[i,n]),1);
    options('sender',state.scene.actors.map(a=>[a.id,a.name]),state.scene.actors[1]?.id);
    options('suppress-root',state.scene.observations.map(r=>[r.id,r.id]),state.scene.observations.find(r=>r.id==='new-library')?.id);
    options('suppress-message',state.scene.messages.map(m=>[m.id,m.id]));
    options('method',[['heuristic','Likelihood heuristic'],['uninformed','Uninformed exact'],...(state.models||[]).map(m=>[m,pretty(m)])],'heuristic');
    $('mode').value=state.mode;
  }
  if(boot.offline){
    $('connection').textContent='OFFLINE · RECORDED REPLAYS';
    for(const el of document.querySelectorAll('.live-only'))el.disabled=true;
    for(const id of ['mode','max-size','method','protect','target-actor','target-room'])$(id).disabled=true;
    $('find').textContent='Show verified featured patch';
  }
  render();
}
function frame(branch=true){return (branch?state.branch:state.baseline)[tick];}
function actor(f,id=selected){return f.actors.find(a=>a.id===id)||f.actors[0];}
function coords(n){
  if(n===4)return [[83,68],[285,68],[83,227],[285,227]];
  return Array.from({length:n},(_,i)=>[184+118*Math.cos(-Math.PI/2+i*2*Math.PI/n),150+108*Math.sin(-Math.PI/2+i*2*Math.PI/n)]);
}
function mapSVG(f,branch){
  const p=coords(f.rooms.length),active=actor(f),accent=branch?'#7de2ca':'#92b5ff';
  let svg=`<svg viewBox="0 0 368 295" role="img" aria-label="${branch?'Intervention':'Original'} world at turn ${f.time}"><defs><pattern id="grid-${branch}" width="18" height="18" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".65" fill="#2a3544"/></pattern></defs><rect width="368" height="295" fill="url(#grid-${branch})"/>`;
  for(const [u,v,cost] of f.edges){const [x,y]=p[u],[xx,yy]=p[v];svg+=`<line x1="${x}" y1="${y}" x2="${xx}" y2="${yy}" stroke="#3a475c" stroke-width="2"/><text x="${(x+xx)/2+5}" y="${(y+yy)/2-5}" font-size="9" fill="#8e9eb4">${cost}</text>`;}
  if(active.path.length>1){svg+=`<polyline points="${active.path.map(r=>p[r].join(',')).join(' ')}" fill="none" stroke="${accent}" stroke-width="3" stroke-dasharray="5 5" opacity=".7"/>`;}
  f.rooms.forEach((room,i)=>{
    const [x,y]=p[i],desired=active.destination===i,truth=f.truth===i;
    svg+=`<rect x="${x-61}" y="${y-32}" width="122" height="64" rx="10" fill="${desired?'#1d303b':'#1b2330'}" stroke="${desired?accent:'#415067'}" stroke-width="${desired?1.7:1}"/>`;
    svg+=`<text x="${x}" y="${y-10}" text-anchor="middle" fill="#e4edf8" font-size="11" font-family="system-ui">${esc(room)}</text>`;
    if(truth)svg+=`<text x="${x+48}" y="${y-19}" text-anchor="middle" fill="#efcd86" font-size="15"><title>Actual parcel location: designer-only information</title>◎</text>`;
    const here=f.actors.filter(a=>a.room===i);
    here.forEach((a,k)=>{const xx=x+(k-(here.length-1)/2)*25;const col=['#92b5ff','#c8b7f1','#efcd86'][f.actors.findIndex(b=>b.id===a.id)%3];svg+=`<circle cx="${xx}" cy="${y+13}" r="10" fill="#26384c" stroke="${col}"/><text x="${xx}" y="${y+17}" text-anchor="middle" font-size="9" font-family="system-ui" fill="${col}"><title>${esc(a.name)}${a.found?' · parcel found':''}</title>${esc(a.name[0])}</text>`;});
  });
  svg+=`<text x="184" y="286" text-anchor="middle" font-size="9" fill="#93a4b9" font-family="system-ui">${esc(active.name)} → ${esc(f.rooms[active.destination])}${active.found?' · parcel found':''}</text></svg>`;
  return svg;
}
function renderActors(f){
  $('actor-cards').innerHTML=f.actors.map(a=>`<button class="actor-card ${a.id===selected?'selected':''}" data-actor="${esc(a.id)}" aria-pressed="${a.id===selected}"><div class="actor-head"><span class="actor-name">${esc(a.name)}</span><span class="avatar">${esc(a.name[0])}</span></div><div class="actor-action">${a.found?'Parcel located':'Next search destination'}<strong>${esc(f.rooms[a.destination])}</strong></div><div class="mini-bars">${a.belief.map((v,i)=>`<span class="mini-bar" style="height:${Math.max(5,v*100)}%" title="${esc(f.rooms[i])}: ${(v*100).toFixed(1)}%"></span>`).join('')}</div><div class="actor-meta"><span>${a.known.length} unique roots</span><span>H = ${a.entropy.toFixed(2)}</span></div></button>`).join('');
  for(const b of document.querySelectorAll('[data-actor]'))b.onclick=()=>{selected=b.dataset.actor;if(!boot.offline)$('target-actor').value=selected;render();};
}
function renderEvidence(f,a){
  const known=new Set(a.known),roots=f.roots.filter(r=>known.has(r.id));
  $('evidence-count').textContent=`${a.receipts.length} receipts / ${roots.length} independent roots`;
  $('evidence-table').innerHTML=roots.map(root=>{
    const receipts=a.receipts.filter(x=>x.root_id===root.id),name=state.scene.actors.find(x=>x.id===root.observer)?.name||root.observer;
    const report=root.likelihood?`Likelihood [${root.likelihood.join(', ')}]`:`${f.rooms[root.reported_room]} · q=${root.reliability.toFixed(2)}`;
    return `<tr><td>${esc(root.id)}</td><td>${esc(name)}</td><td>t=${root.time}</td><td>${receipts.map(r=>`t=${r.received_at}`).join(', ')}</td><td>${esc(report)}</td></tr>`;
  }).join('')||'<tr><td colspan="5">No observations received at this turn.</td></tr>';
  const allReceipts=a.receipts.slice(-14),height=Math.max(180,allReceipts.length*30+35),ys=new Map();
  allReceipts.forEach((r,i)=>{if(!ys.has(r.root_id))ys.set(r.root_id,[]);ys.get(r.root_id).push(25+i*30);});
  let svg=`<svg viewBox="0 0 650 ${height}" style="height:${height}px" role="img" aria-label="Observation roots and repeated receipts for ${esc(a.name)}">`;
  for(const [root,values] of ys){const y=values.reduce((x,v)=>x+v,0)/values.length;
    values.forEach(v=>{svg+=`<path d="M200 ${y} C245 ${y} 260 ${v} 305 ${v}" fill="none" stroke="#52697f" stroke-width="1.5"/><path d="M450 ${v} C500 ${v} 510 ${height/2} 555 ${height/2}" fill="none" stroke="#385b59"/>`;});
    svg+=`<rect x="15" y="${y-12}" width="185" height="24" rx="6" fill="#253246" stroke="#50627b"/><text x="107" y="${y+4}" text-anchor="middle" fill="#d9e6fa" font-size="9"><title>${esc(root)}</title>${esc(root.length>25?root.slice(0,24)+'…':root)}</text>`;
  }
  allReceipts.forEach((r,i)=>{const y=25+i*30,msg=f.messages.find(m=>m.id===r.message_id);svg+=`<rect x="305" y="${y-10}" width="145" height="21" rx="5" fill="#1e3436" stroke="#466362"/><text x="377" y="${y+4}" text-anchor="middle" fill="#b8e5d9" font-size="9"><title>${esc(msg?`${msg.id}; parent: ${msg.parent_id||'original observation'}`:'Direct observation')}</title>${r.message_id?'Delivery':'Observed'} at t=${r.received_at}</text>`;});
  svg+=`<circle cx="592" cy="${height/2}" r="32" fill="#284139" stroke="#7de2ca"/><text x="592" y="${height/2+4}" text-anchor="middle" font-size="10" fill="#d9faee">${esc(a.name)}</text></svg>`;
  $('provenance').innerHTML=svg;
}
function renderManual(f){
  const sender=$('sender').value,known=new Set(actor(f,sender).known),old=$('manual-root').value;
  options('manual-root',f.roots.filter(r=>known.has(r.id)).map(r=>[r.id,r.id]),old);
  if(!$('manual-root').value&&$('manual-root').options.length)$('manual-root').selectedIndex=0;
}
function renderSolution(){
  if(!lastSearch){$('solution').innerHTML='<span class="muted">No search has been run at this checkpoint.</span>';$('apply').disabled=true;$('audit').disabled=true;return;}
  const r=lastSearch.record,patch=r.patch;
  const labels={verified_minimum_within_bounds:'Verified minimum · within bounds',verified_feasible:'Verified feasible · minimum unresolved',no_solution_within_bounds:'No solution within these bounds',budget_exhausted:'Budget exhausted · unresolved'};
  let text=`<strong>${esc(labels[r.status]||r.status)}</strong>`;
  if(patch!==null){text+=patch.length?`${patch.length} permitted ${patch.length===1?'delivery':'deliveries'} changes the next search.`:'The requested action is already selected. No delivery needed.';text+=patch.map(d=>`<div class="patch-item"><b>${esc(d.sender)} → ${esc(d.recipient)}</b><br>${esc(d.root_id)}</div>`).join('');}
  else text+='<span class="muted">No executable patch has been returned.</span>';
  text+=`<div class="stats"><span><b>${r.verifier_calls}</b>verifier calls</span><span><b>${(r.elapsed_seconds*1000).toFixed(1)}</b>milliseconds</span><span><b>${r.candidate_space_size}</b>bounded subsets</span></div>`;
  $('solution').innerHTML=text;$('apply').disabled=patch===null||state.patched_times.includes(lastSearch.bundle.time);$('audit').disabled=false;
}
function render(){
  if(!state)return;
  const f=frame(),base=frame(false),a=actor(f),original=actor(base,a.id);
  $('tick-label').textContent=`TURN ${tick}`;$('time-value').textContent=`${tick} / ${state.scene.horizon}`;$('timeline').value=tick;
  $('map-before').innerHTML=mapSVG(base,false);$('map-after').innerHTML=mapSVG(f,true);renderActors(f);
  $('inspector-title').textContent=`${a.name}’s next search`;$('entropy-label').textContent=`H = ${a.entropy.toFixed(3)} NATS`;
  $('belief-bars').innerHTML=f.rooms.map((name,i)=>`<div class="belief-row"><div class="belief-line"><span>${esc(name)}</span><span class="mono">${(a.belief[i]*100).toFixed(1)}%</span></div><div class="belief-track"><div class="belief-fill" style="width:${a.belief[i]*100}%"></div><span class="belief-original" style="left:${Math.min(99.5,original.belief[i]*100)}%"></span></div></div>`).join('');
  $('routes').innerHTML=a.alternatives.slice(0,4).map(r=>`<div class="route"><span class="route-name">${r.order.map(i=>esc(f.rooms[i])).join(' → ')}</span><span class="route-cost">${r.cost.toFixed(2)}</span></div>`).join('');
  renderEvidence(f,a);renderManual(f);renderSolution();
  $('events').innerHTML=f.events.slice(-70).map(e=>`<div class="event-row"><span class="event-time">t=${e.time}</span><span class="event-kind">${esc(pretty(e.type))}</span><span class="event-details">${esc(e.id||e.actor||'')}${e.sender?' · '+esc(e.sender)+' → '+esc(e.recipient):''}${e.type==='move'?' · '+esc(f.rooms[e.from])+' → '+esc(f.rooms[e.to]):''}${e.type==='inspection'?' · '+(e.found?'parcel found':'room empty'):''}</span></div>`).join('')||'<p class="muted">No events yet.</p>';
}
function download(name,data){const blob=new Blob([typeof data==='string'?data:JSON.stringify(data,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function pause(){if(timer)clearInterval(timer);timer=null;$('play').textContent='▶';}
$('timeline').oninput=()=>{pause();tick=Number($('timeline').value);render();};
$('step').onclick=()=>{pause();tick=Math.min(tick+1,state.scene.horizon);render();};
$('play').onclick=()=>{if(timer){pause();return;}if(tick>=state.scene.horizon)tick=state.scene.search_start;$('play').textContent='Ⅱ';timer=setInterval(()=>{if(tick>=state.scene.horizon){pause();return;}tick++;render();},650);};
$('scenario').onchange=()=>perform(async()=>{pause();lastSearch=null;if(boot.offline){offlineFamily=$('scenario').value;setState(clone(boot.scenes[offlineFamily].state),{fresh:true});}else setState(await api('scene',{family:$('scenario').value}),{fresh:true});notice('');});
$('generate').onclick=()=>perform(async()=>{pause();setState(await api('scene',{family:$('scenario').value,seed:Number($('seed').value)}),{fresh:true});notice('A deterministic procedural scene was generated. Its seed is recorded in the exported JSON.');});
$('reset').onclick=()=>perform(async()=>{pause();lastSearch=null;if(boot.offline)setState(clone(boot.scenes[offlineFamily].state),{fresh:true});else setState(await api('reset'),{fresh:true});notice('Branch reset. The original scene has not been changed.');});
$('mode').onchange=()=>perform(async()=>{lastSearch=null;setState(await api('fork',{mode:$('mode').value}));notice('Policy baseline changed. Patch verification is available in provenance-aware mode.');});
$('target-actor').onchange=()=>{selected=$('target-actor').value;render();};
$('sender').onchange=()=>renderManual(frame());
$('find').onclick=()=>perform(async()=>{
  pause();
  if(boot.offline){lastSearch=clone(boot.scenes[offlineFamily].featured);tick=lastSearch.bundle.time;render();notice('This certificate and its replay were computed by the Python verifier. The offline page does not run a new search.');return;}
  lastSearch=await api('search',{time:tick,actor:$('target-actor').value,room:Number($('target-room').value),max_size:Number($('max-size').value),method:$('method').value,protected:$('protect').checked?[$('target-actor').value]:[]});
  renderSolution();notice(lastSearch.record.patch===null?'Search completed without a feasible patch. Read the status carefully.':'Patch checked by the exact simulator. Apply it to watch the matched replay.');
});
$('apply').onclick=()=>perform(async()=>{
  pause();if(!lastSearch)return;tick=lastSearch.bundle.time;
  if(boot.offline){state.branch=clone(lastSearch.replay);state.patched_times=[tick];render();}else setState(await api('apply'));
  notice('Patch applied. Compare the two maps, then press Play to follow the consequences.');
});
$('audit').onclick=()=>{if(lastSearch)download('belieftown-audit.json',lastSearch.bundle);};
$('deliver').onclick=()=>perform(async()=>{
  pause();const before=actor(frame(),$('target-actor').value).belief.slice();
  setState(await api('deliver',{time:tick,sender:$('sender').value,recipient:$('target-actor').value,root_id:$('manual-root').value}));
  lastSearch=null;const after=actor(frame(),$('target-actor').value).belief;
  const delta=after.reduce((s,v,i)=>s+Math.abs(v-before[i]),0);render();
  notice(delta<1e-12?'The delivery was recorded, but the belief did not change: this observation root was already known.':`Delivery applied. Total belief change (L1) = ${delta.toFixed(4)}.`);
});
$('fork-root').onclick=()=>perform(async()=>{pause();lastSearch=null;setState(await api('fork',{observation:$('suppress-root').value}));notice('Observation suppressed. All dependent messages, inspections, and actions were regenerated.');});
$('fork-message').onclick=()=>perform(async()=>{pause();lastSearch=null;setState(await api('fork',{message:$('suppress-message').value}));notice('Transmission suppressed. A valid alternate evidence path, if available, remains intact.');});
$('import').onclick=()=>$('file').click();
$('file').onchange=()=>perform(async()=>{const file=$('file').files[0];if(!file)return;if(file.size>2000000)throw new Error('Scene JSON must be under 2 MB.');const d=JSON.parse(await file.text());setState(await api('scene',{scene:d.episode?.scene||d.scene||d}),{fresh:true});notice('Scene imported and validated.');});
$('download-episode').onclick=()=>perform(async()=>{const value=boot.offline?{kind:'belieftown.recorded-showcase',scene:state.scene,frames:state.branch}:await api('episode');download('belieftown-episode.json',value);});
(async()=>{try{if(boot.offline){setState(clone(boot.scenes[offlineFamily].state),{fresh:true});notice('Offline showcase: playback and inspection are interactive; the featured patches were precomputed and verified. Run the local app for arbitrary scenes and live searches.');}else{const r=await fetch('/api/state',{headers:{'X-BeliefTown-Token':boot.token}});if(!r.ok)throw new Error('Could not connect to the local simulator.');setState(await r.json(),{fresh:true});}}catch(err){notice(err.message,true);}})();
