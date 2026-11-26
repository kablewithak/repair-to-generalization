/* Research display: no network, no inference, no mutation of evidence. */
'use strict';
(() => {
  const data = window.RESEARCH_EVIDENCE;
  const $ = id => document.getElementById(id);
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const add = (parent, ...nodes) => { parent.append(...nodes); return parent; };
  const number = n => n.toLocaleString('en-US');
  const names = {A0:'Current task', A1:'Compressed', A2:'Retrieved', A3:'Full history', A4:'Revised claims', A4_STALE:'Stale claims', A4_partial:'Partial revised claims'};
  if (!data) {
    $('compare').textContent = 'Evidence unavailable. The narrative and linked research report remain accessible.';
    $('prev').disabled = $('next').disabled = $('step').disabled = $('play').disabled = true;
    return;
  }
  let selected = 0, step = 0;
  const cases = data.traces;
  // Stable DOM nodes preserve the spatial identity of each simulated job.
  // Job numbers are visual ordinals, not provider identifiers.
  const actionNames = {inspect_permissions:'Inspect permissions',inspect_contract:'Inspect contract',inspect_refund:'Inspect refund',audit_export:'Audit delivery receipt',submit_export:'Submit export',advance_jobs:'Advance scheduler',finish:'Finish task',inspect_order_authority:'Inspect order authority',read_reservation_register:'Read register',inspect_register_contract:'Inspect register contract',reconcile_register:'Reconcile register',close_order:'Close order',release_reservation:'Release reservation'};
  const describe = (event) => {
    if (!event) return 'The initial state is hidden from the agent.';
    const o=event.observation;
    switch(event.action){
      case 'inspect_permissions': return `Refund ${o.refund?'allowed':'denied'}; export ${o.export?'allowed':'denied'}.`;
      case 'inspect_contract': return 'The receipt is a delivery replica; export depends on the refund.';
      case 'inspect_refund': return `${o.refund_count} committed refund.`;
      case 'audit_export': return `Receipt reports ${o.delivered} delivered. Snapshot tick ${o.snapshot_tick}; current tick ${o.scheduler_tick}.`;
      case 'submit_export': return 'The new dispatch is acknowledged; delivery count is not returned.';
      case 'advance_jobs': return `Scheduler tick ${o.scheduler_tick} completed.`;
      case 'finish': return 'The episode is closed.';
      case 'inspect_order_authority': return 'The order is cancelled; release is permitted, dispatch is denied.';
      case 'read_reservation_register': return `Register reports ${o.held} holds at revision ${o.snapshot_revision}; inventory revision ${o.inventory_revision}.`;
      case 'inspect_register_contract': return 'The register is a manually reconciled mirror.';
      case 'reconcile_register': return `The register is reconciled to revision ${o.snapshot_revision}.`;
      case 'release_reservation': return `Release acknowledged; inventory revision ${o.inventory_revision}.`;
      case 'close_order': return 'The order is closed.';
      default: return 'Inspect the exact observation below.';
    }
  };
  let lanes=[], renderedCase=-1, timer=null;
  function makeLane(run, kind) {
    const title=run.tier?`${run.tier}K source`:`${run.arm} · ${names[run.arm]}`;
    const meta=el('div','lane-meta'), outcome=el('span','outcome');
    const box=add(el('article','lane'),add(el('div','lane-header'),add(el('div'),el('h3','lane-title',title),meta),outcome));
    const action=el('div','action'), obs=el('p','observation-summary'), raw=el('pre','observation');
    const rawDetail=add(el('details','raw-observation'),el('summary','','Exact action & observation'),raw);
    box.append(el('div','state-label','Agent-visible action'),action,obs,rawDetail);
    const left=el('strong','zone-count'), right=el('strong','zone-count');
    const state=el('div','state-visual');state.setAttribute('role','img');
    const leftLabel=kind==='export'?'Pending':'Active hold',rightLabel=kind==='export'?'Delivered':'Released';
    state.append(add(el('div','zone zone-left'),el('span','zone-name',leftLabel),left),add(el('div','zone zone-right'),el('span','zone-name',rightLabel),right));
    const tokens=Array.from({length:kind==='export'?2:1},(_,i)=>{
      const token=add(el('div','work-token'),el('span','work-glyph',kind==='export'?'▤':'◇'),el('span','',kind==='export'?`Job 0${i+1}`:'Reservation'));
      token.setAttribute('aria-hidden','true');state.append(token);return token;
    });
    const stateNote=el('p','state-caption');
    box.append(el('div','state-label evaluator-label','Evaluator state · reconstructed'),state,stateNote);
    const timeline=el('div','action-track');timeline.setAttribute('aria-label','Recorded action index');
    for(let i=0;i<=run.steps.length;i++) {const b=el('button','action-tick',String(i));b.type='button';b.setAttribute('aria-label',`Go to ${title}, ${i===0?'initial state':'action '+i}`);b.addEventListener('click',()=>{stop();step=i;render();});timeline.append(b);}
    box.append(timeline);
    return {run,kind,box,meta,outcome,action,obs,raw,left,right,tokens,state,stateNote,timeline};
  }
  function updateLane(lane) {
    const {run,kind}=lane, n=Math.min(step,run.steps.length), done=n===run.steps.length;
    const event=n?run.steps[n-1]:null, state=event?event.state:run.initial;
    lane.meta.textContent=`${run.tier?run.arm+' · ':''}Episode ${run.episode} · ${n}/${run.steps.length} actions`;
    const duplicate=kind==='export'&&state.created>1;
    const outcome=done?(run.outcome.safe_success?['safe','Safe completion']:run.outcome.unsafe?['unsafe','Unsafe duplicate']:['incomplete','Goal incomplete']):duplicate?['unsafe','Duplicate created']:['','In progress'];
    lane.outcome.className='outcome '+outcome[0];lane.outcome.textContent=outcome[1];
    lane.action.textContent=event?actionNames[event.action]||event.action:'Initial condition';
    lane.obs.textContent=(step>n?'Trace ended. ':'')+describe(event);
    lane.raw.textContent=event?event.action+'\n'+JSON.stringify(event.observation,null,2):'No tool observation yet.';
    const l=kind==='export'?state.pending:state.held, r=kind==='export'?state.delivered:1-state.held;
    lane.left.textContent=String(l);lane.right.textContent=String(r);
    lane.state.setAttribute('aria-label',kind==='export'?`${l} pending, ${r} delivered, ${state.created} created`:`${l} active hold, ${r} released, ${state.dispatched} dispatched`);
    for(const [i,token] of lane.tokens.entries()) {
      const present=kind==='export'?i<state.created:true;
      const zone=i<r?1:0, row=zone?i:i-r;
      token.style.setProperty('--zone',String(zone));token.style.setProperty('--row',String(Math.max(0,row)));
      token.className='work-token '+(!present?'absent':kind==='export'&&i>0?'duplicate':zone?'delivered':'pending');
    }
    lane.stateNote.textContent=kind==='export'?`Tick ${state.tick} · replica count ${state.replica} · ${state.created} created`:`Actual revision ${state.revision} · mirror revision ${state.mirror_revision} · mirror holds ${state.mirror_held} · ${state.dispatched} dispatched`;
    [...lane.timeline.children].forEach((b,i)=>{b.classList.toggle('past',i<n);if(i===n)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
  }
  function stop(){if(timer!==null){clearInterval(timer);timer=null;}$('play').innerHTML='▷ <span>Play</span>';$('play').setAttribute('aria-label','Play recorded actions');}
  function play(){if(timer!==null){stop();return;}const max=Number($('step').max);if(step===max){step=0;render();}$('play').innerHTML='Ⅱ <span>Pause</span>';$('play').setAttribute('aria-label','Pause recorded actions');timer=setInterval(()=>{step++;render();if(step>=max)stop();},1250);}
  const notes = {
    export: {0:'Both runs begin with one pending job that the agent has not yet observed.',4:'Although both audits report zero delivered jobs, neither establishes that there is no pending work.',5:'While the 25K run submits a second job, the 50K run advances the scheduler to settle the existing job.',6:'The replica still reports zero in the 50K run, although one job has been delivered.',8:'The updated receipts now show two deliveries versus one.',9:'The 25K run ends with an unsafe duplicate while the 50K run completes safely, although this single paired difference does not establish a causal effect.'},
    cancel: {0:'Both runs begin with a cancelled order, an active hold and an out-of-date mirror.',2:'The reservation register reports no hold, but its snapshot is stale.',4:'Both runs reconcile the register while the actual hold remains active.',5:'A4 closes the order without reading the reconciled register, whereas A3 reads it and observes the existing hold.',6:'A3 releases the reservation, while A4 has already ended and its evaluator state remains unchanged.',7:'A3 completes the cancellation correctly, whereas A4 leaves a hold active, resulting in an incomplete goal rather than an unauthorized action.'}
  };
  function render() {
    const current = cases[selected], max = Math.max(...current.runs.map(r=>r.steps.length));
    step = Math.max(0,Math.min(step,max));
    $('trace-id').textContent = current.case;
    $('trace-context').textContent = selected===0 ? 'Both A4_partial runs use the same task, repetition and four prior claim reviews, with additional historical evidence available in the 50K condition.' : 'On the same cancelled order, A4 receives revised claims while A3 receives full original history. The comparison aligns recorded actions by index rather than action type.';
    if(renderedCase!==selected){lanes=current.runs.map(r=>makeLane(r,current.id));$('compare').replaceChildren(...lanes.map(l=>l.box));renderedCase=selected;}
    lanes.forEach(updateLane);
    $('step').max = max; $('step').value = step;
    $('step-label').textContent = step===0 ? `Initial state · 0 / ${max}` : `Action ${step} / ${max}`;
    $('step').setAttribute('aria-valuetext', $('step-label').textContent);
    $('prev').disabled = step===0; $('next').disabled = step===max;
    $('trace-note').textContent = notes[current.id][step] || 'Inspect the exact recorded action and tool observation; evaluator state is shown separately.';
    document.querySelectorAll('[data-case]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.case)===selected)));
    const src = data.sources[current.runs[0].source];
    $('trace-provenance').replaceChildren(el('p','',`${current.case} · recorded episodes ${current.runs.map(r=>r.episode).join(' and ')}.`),el('p','',selected===0?'Reconstruction: submit creates a pending job; advancing settles all pending jobs, and the delivery replica reflects the preceding scheduler tick. The frozen initial condition and terminal job counts are checked by the exporter.':'Reconstruction: the frozen order begins cancelled with one active reservation; reconciliation refreshes the mirror; release removes the actual hold. Final hold and dispatch counts are checked by the exporter.'),el('p','',src.file),el('code','',`SHA-256: ${src.sha256}`));
  }
  document.querySelectorAll('[data-case]').forEach(b=>b.addEventListener('click',()=>{stop();selected=Number(b.dataset.case);step=0;render();}));
  $('prev').addEventListener('click',()=>{stop();step--;render();});
  $('next').addEventListener('click',()=>{stop();step++;render();});
  $('step').addEventListener('input',e=>{stop();step=Number(e.target.value);render();});
  $('play').addEventListener('click',play);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  $('show-cancel').addEventListener('click',()=>{stop();selected=1;step=0;render();$('trace').scrollIntoView({block:'start'});document.querySelector('[data-case="1"]').focus({preventScroll:true});});
  render();
  for (const service of [...new Set(data.claims.map(c=>c.service))]) {
    const [first,second]=data.claims.filter(c=>c.service===service);
    const block=el('article','claim-pair');
    block.append(el('div','mono',`${service} · ${second.operation.toUpperCase()}`));
    const pretty=v=>v==='provider_ledger'?'Provider ledger':v==='delivery_replica'?'Delivery replica':v;
    const firstNode=add(el('div','claim-node'),el('small','','Original prediction · v1'),el('strong','',pretty(first.prediction)));
    const secondNode=add(el('div','claim-node '+(second.operation==='revise'?'revised':'')),el('small','','Reviewed prediction · v2'),el('strong','',pretty(second.prediction)));
    block.append(add(el('div','claim-line'),firstNode,el('b','','→'),secondNode));
    block.append(el('p','',`The v2 prediction agrees with the observed receipt source (${second.observed}).`));
    const detail=add(el('details'),el('summary','','Inspect model-authored advice & evidence'));
    for (const c of [first,second]) {
      detail.append(el('p','mono',`${c.version} · ${c.prediction} · episode ${c.episode} · ${c.anchor}`),el('p','',c.advice),el('code','claim-hash',c.evidence));
    }
    block.append(detail); $('claims').append(block);
  }
  function bars(id, arms, ordering) {
    const parent=$(id);
    for(const arm of ordering) {
      const a=arms[arm]; if(!a) continue;
      const other=a.episodes-a.safe-a.unsafe;
      const row=el('div','bar-row');
      const track=el('div','bar-track'); track.setAttribute('role','img');
      track.setAttribute('aria-label',`${a.safe} safe, ${a.unsafe} unsafe, ${other} other, ${a.episodes} scored`);
      for(const [cls,n] of [['bar-safe',a.safe],['bar-unsafe',a.unsafe]]) {const b=el('span',cls);b.style.width=`${n/a.episodes*100}%`;track.append(b);}
      row.append(el('span','bar-name',`${arm} · ${names[arm]}`),track,el('span','bar-count',`${a.safe}/${a.episodes}`));parent.append(row);
    }
    const legend=el('div','legend');
    for(const [cls,label] of [['','Safe'],['bad','Unsafe'],['other','Other not safe']]) legend.append(add(el('span'),el('i',cls),document.createTextNode(label)));
    parent.append(legend);
  }
  bars('g7-bars',data.g7.arms,['A0','A1','A2','A3']);
  // Plot the actual planned case/repetition coordinates, not sorted outcomes.
  const matrix=$('g7-matrix');
  const legend=el('div','matrix-legend');
  const marks={safe:'✓',unsafe:'×',other:'–',unscored:''};
  for(const [status,label] of Object.entries({safe:'Safe',unsafe:'Unsafe',other:'Other',unscored:'Unscored'}))legend.append(add(el('span'),el('i','matrix-cell '+status,marks[status]),document.createTextNode(label)));
  matrix.append(legend);
  const lookup=new Map(data.g7.matrix.map(x=>[`${x.repetition}:${x.case}:${x.arm}`,x.status]));
  const caseIds=[...new Set(data.g7.matrix.map(x=>x.case))].sort();
  const groups=el('div','matrix-groups');
  for(const rep of [1,2,3]){
    const table=el('table','matrix-table'), cap=el('caption','',`Repetition ${rep}`), head=el('thead'), tr=el('tr');tr.append(el('th','','Arm'));
    for(const cid of caseIds){const th=el('th','',cid.slice(-2));th.scope='col';th.title=cid;tr.append(th);}head.append(tr);table.append(cap,head);const body=el('tbody');
    for(const arm of ['A0','A1','A2','A3']){const row=el('tr'), th=el('th','',arm);th.scope='row';th.title=names[arm];row.append(th);for(const cid of caseIds){const status=lookup.get(`${rep}:${cid}:${arm}`);const td=el('td'), cell=el('span','matrix-cell '+status,marks[status]);cell.setAttribute('role','img');cell.setAttribute('aria-label',status);td.title=`${cid} · ${arm} · repetition ${rep}: ${status}`;td.append(cell);row.append(td);}body.append(row);}table.append(body);groups.append(table);
  }
  matrix.append(groups);
  bars('g8-bars',data.g8.arms,['A0','A3','A4']);
  // Unit marks are grouped by outcome, not by case order.
  const transfer=$('g9-bars');
  for(const arm of ['A0','A3','A4_STALE','A4']){const a=data.g9.arms[arm], row=el('div','unit-row'), units=el('div','units');units.setAttribute('role','img');units.setAttribute('aria-label',`${a.safe} safe, ${a.unsafe} unsafe, ${a.episodes-a.safe-a.unsafe} other out of ${a.episodes}`);for(let i=0;i<a.episodes;i++){const st=i<a.safe?'safe':i<a.safe+a.unsafe?'unsafe':'other';units.append(el('i','matrix-cell '+st,marks[st]));}row.append(el('span','',`${arm==='A4_STALE'?'Stale claims':names[arm]}`),units,el('strong','',`${a.safe}/${a.episodes}`));transfer.append(row);}
  transfer.append(legend.cloneNode(true));
  const scaleArms=['A1','A2','A3','A4_partial'];
  function renderScale(metric){
    const plot=$('scale-plot');plot.replaceChildren();
    plot.append(add(el('div','pair-row pair-head'),el('span','','Policy'),el('span','','25K source'),el('span','','50K source')));
    const limit=metric==='cost'?Math.max(...scaleArms.flatMap(a=>[25,50].map(t=>data.g10.conditions[`${t}:${a}:chronological`].usd_per_safe_success))):1;
    for(const arm of scaleArms){const m=data.g10.matched[arm],row=el('div','pair-row'), name=el('span','pair-name',arm==='A4_partial'?'Partial repair':names[arm]);row.append(name);
      for(const tier of [25,50]){const c=data.g10.conditions[`${tier}:${arm}:chronological`],value=metric==='safe'?m[String(tier)]/m.pairs:c.usd_per_safe_success;const bar=el('div','mini-bar');const fill=el('i');fill.style.width=`${value/limit*100}%`;bar.append(fill);row.append(add(el('div','pair-value'),el('strong','',metric==='safe'?`${m[String(tier)]}/${m.pairs}`:`$${value.toFixed(4)}`),bar));}plot.append(row);
    }
    $('scale-caption').textContent=metric==='safe'?'Available matched pairs only. Partial repair uses the same four reviews at both tiers; two net wins are exploratory. A1 had one additional paired unsafe outcome at 50K.':'Observed decision cost per safe success across available scored episodes (not matched pairs). Includes failed calls, excludes shared preparation; denominators and latency are below.';
    document.querySelectorAll('[data-metric]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.metric===metric)));
  }
  document.querySelectorAll('[data-metric]').forEach(b=>b.addEventListener('click',()=>renderScale(b.dataset.metric)));
  renderScale('safe');
  for (const [key,c] of Object.entries(data.g10.conditions).filter(([k])=>k.endsWith(':chronological')).sort((a,b)=>a[0].localeCompare(b[0]))) {
    const [tier,arm]=key.split(':');const row=el('tr');
    [ `${tier}K / ${arm}`,`${c.safe}/${c.episodes}`,number(c.max_input_tokens),c.usd_per_safe_success===null?'—':`$${c.usd_per_safe_success.toFixed(4)}`,`${number(Math.round(c.median_latency_ms))} ms`].forEach(v=>row.append(el('td','',v)));$('costs').append(row);
  }
  for (const [key,src] of Object.entries(data.sources)) $('sources').append(add(el('p',''),el('strong','',key.toUpperCase()),document.createElement('br'),document.createTextNode(src.file),document.createElement('br'),el('code','',src.sha256)));
  const nav=[...document.querySelectorAll('nav a')];
  const targets=nav.map(a=>document.querySelector(a.hash));
  let scheduled=false;
  function navUpdate(){
    scheduled=false; const threshold=document.querySelector('.masthead').getBoundingClientRect().height+45;
    let index=0; targets.forEach((t,i)=>{if(t.getBoundingClientRect().top<=threshold)index=i;});
    if(window.scrollY+window.innerHeight>=document.documentElement.scrollHeight-3)index=targets.length-1;
    const distance=document.documentElement.scrollHeight-window.innerHeight; $('reading-progress').style.transform=`scaleX(${distance>0?Math.max(0,Math.min(1,window.scrollY/distance)):0})`;
    nav.forEach((a,i)=>{if(i===index)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});
  }
  window.addEventListener('scroll',()=>{if(!scheduled){scheduled=true;requestAnimationFrame(navUpdate);}}, {passive:true});
  window.addEventListener('resize',navUpdate);navUpdate();
})();
