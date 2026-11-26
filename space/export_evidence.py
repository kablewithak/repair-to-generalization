"""Allowlisted research-display export. Run locally against exact private reports.
No raw requests, hidden reasoning, or credentials are exported. No network.
"""
import argparse, hashlib, json
from pathlib import Path

SOURCES = {
 'scale': ('savant_g10_scaling_20261126_100602.jsonl','21127dc6abd6d6d57f41786ff0276102b6239ce35e61b51439b861439ca209bf'),
 'transfer': ('savant_g9_fulfillment_dev_20261126_010454.jsonl','4aaaf1fc1d5557c237b412bc87ffb2e1716cfc180ced5e5ee2c135187ebefd48'),
 'repair': ('savant_g8_independent_20261125_235424.jsonl','b38b3159c8a93914d5dabf015850175bc108138a41091c5f9b471f2b07461e0c'),
 'reconstruct': ('savant_g7_main_v2_20261125_170804.jsonl','6b7e76bd5fe51f1a13e6aefea3d2c6dd212b81ba999e83dc2f4270548accb24e'),
 'dev': ('savant_g8_checkpoint_20261125_223554.jsonl','2c7dcd2a2ee423cdb13c06a210df17559ea2313b28cd9084fb790074d1a387a1'),
}

def load(root, key):
    name, expected = SOURCES[key]
    raw = (root/name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'Hash mismatch: {name}')
    return [json.loads(line) for line in raw.splitlines()]

def trace(rows, key, case, arm, tier=None):
    prefix = 'g10_scaling' if key=='scale' else 'g9_dev'
    def match(r):
        return r.get('case_id')==case and r.get('arm')==arm and (tier is None or (r.get('tier')==tier and r.get('repetition')==1 and r.get('ordering')=='chronological'))
    spec = next(c for c in rows[0]['cases'] if c['case_id']==case)
    steps = [r for r in rows if r['record_type']==prefix+'_public_step' and match(r)]
    end, = [r for r in rows if r['record_type']==prefix+'_episode_finished' and match(r)]
    assert end['replay_valid'] and len(steps)==end['outcome']['actions']
    # Evaluator state is a derived reconstruction from the frozen case and
    # simulator rules, not an agent observation. Only these two fixed cases.
    if key=='scale':
        assert spec['export_initial']=='queued' and spec['refund_committed'] and spec['receipt']=='delivery_replica'
        state={'pending':1,'delivered':0,'created':1,'tick':0,'replica':0}
    else:
        assert spec['held'] and spec['mirror_stale'] and spec['cancelled']
        state={'held':1,'dispatched':0,'mirror_held':0,'revision':1,'mirror_revision':0}
    initial=dict(state); events=[]
    for i,r in enumerate(steps):
        assert r['step']==i and r['episode']==end['episode']
        a,o=r['action'],r['observation']
        if key=='scale':
            if a=='submit_export':
                assert o['message']=='dispatch_acknowledged'; state['pending']+=1; state['created']+=1
            elif a=='advance_jobs':
                state['replica']=state['delivered']; state['delivered']+=state['pending'];state['pending']=0;state['tick']+=1
                assert o['scheduler_tick']==state['tick']
            elif a=='audit_export':
                assert o['delivered']==state['replica'] and o['scheduler_tick']==state['tick']
            else: assert a in ('inspect_permissions','inspect_contract','inspect_refund','finish')
        else:
            if a=='reconcile_register':
                state['mirror_held']=state['held'];state['mirror_revision']=state['revision']
                assert o['snapshot_revision']==state['mirror_revision']
            elif a=='release_reservation':
                assert state['held']==1;state['held']=0;state['revision']+=1
                assert o['inventory_revision']==state['revision']
            elif a=='read_reservation_register':
                assert o['held']==state['mirror_held'] and o['inventory_revision']==state['revision'] and o['snapshot_revision']==state['mirror_revision']
            else: assert a in ('inspect_order_authority','inspect_register_contract','close_order')
        events.append({'action':a,'observation':o,'state':dict(state)})
    out=end['outcome']
    if key=='scale': assert (state['created'],state['delivered'])==(out['jobs_created'],out['jobs_delivered'])
    else: assert (state['held'],state['dispatched'])==(out['held'],out['dispatched'])
    return {'episode':end['episode'],'case':case,'arm':arm,'tier':tier,'initial':initial,'steps':events,'outcome':out,'source':key}

def build(root):
    r={key:load(root,key) for key in SOURCES}
    scale=r['scale'][-1]; transfer=r['transfer'][-1]; recon=r['reconstruct'][-1]; dev=r['dev'][-1]
    claims=[]
    for row in r['repair']:
        if row['record_type']=='g8_cl_reflection':
            c=row['claim']; m=c['claim']
            claims.append({'service':row['service'],'version':row['version'],'prediction':c['prediction'],'advice':c['advice'],'anchor':c['anchor'],'evidence':m['evidence_ids'][0],'operation':row['operation'],'observed':row['observed_prediction'],'episode':row['episode']})
    assert len(claims)==4 and all(c['prediction']==c['observed'] for c in claims)
    scored = [x for x in r['scale'] if x['record_type']=='g10_scaling_episode_finished' and x['ordering']=='chronological']
    matched = {}
    for arm, contrast in scale['scale_contrasts'].items():
        groups = {tier:{(x['repetition'],x['case_id']):x['outcome'] for x in scored if x['arm']==arm and x['tier']==tier} for tier in (25,50)}
        keys = groups[25].keys() & groups[50].keys()
        a,b = (sum(groups[t][k]['safe_success'] for k in keys) for t in (25,50))
        assert len(keys)==contrast['pairs'] and b-a==contrast['net_safe_wins']
        matched[arm]={'25':a,'50':b,'pairs':len(keys),'net':b-a,'additional_unsafe':len(contrast['additional_unsafe'])}
    data={'sources':{k:{'file':v[0],'sha256':v[1]} for k,v in SOURCES.items()},'traces':[
        {'id':'export','label':'Duplicate export','case':'study-main-03 · repetition 1','runs':[trace(r['scale'],'scale','study-main-03','A4_partial',25),trace(r['scale'],'scale','study-main-03','A4_partial',50)]},
        {'id':'cancel','label':'Cancelled order','case':'fulfill-dev-05 · development','runs':[trace(r['transfer'],'transfer','fulfill-dev-05','A4'),trace(r['transfer'],'transfer','fulfill-dev-05','A3')]}],
        'claims':claims,'g7':{'scored':recon['scored_episodes'],'planned':recon['planned_episodes'],'arms':{a:{'safe':v['safe_successes'],'episodes':v['episodes'],'unsafe':v['unsafe']} for a,v in recon['by_arm'].items()}},
        'g8':{'arms':{a:{'safe':v['safe_successes'],'episodes':v['checkpoints'],'unsafe':v['unsafe']} for a,v in dev['by_arm'].items()},'replication_scored':r['repair'][-1]['scored_main_episodes']},
        'g9':{'arms':{a:{k:v[k] for k in ('safe','episodes','unsafe')} for a,v in transfer['by_arm'].items()}},
        'g10':{'scored':scale['scored_episodes'],'planned':scale['planned_episodes'],'conditions':scale['by_condition'],'contrasts':scale['scale_contrasts'],'matched':matched,'order':scale['order_contrasts']}}
    # Preserve every planned G7 case identity and explicitly distinguish unscored cells.
    planned = [tuple(x) for x in r['reconstruct'][0]['order']]
    finished = {(x['repetition'],x['case_id'],x['arm']):x['outcome'] for x in r['reconstruct'] if x['record_type']=='g7_episode_finished'}
    assert len(planned)==len(set(planned))==144 and len(finished)==59 and finished.keys()<=set(planned)
    def status(outcome):
        return 'unscored' if outcome is None else 'safe' if outcome['safe_success'] else 'unsafe' if outcome['unsafe'] else 'other'
    data['g7']['matrix']=[{'repetition':rep,'case':cid,'arm':arm,'status':status(finished.get((rep,cid,arm)))} for rep,cid,arm in planned]
    for arm, counts in data['g7']['arms'].items():
        group=[x['status'] for x in data['g7']['matrix'] if x['arm']==arm]
        assert group.count('safe')==counts['safe'] and group.count('unsafe')==counts['unsafe'] and len(group)-group.count('unscored')==counts['episodes']
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reports',type=Path);p.add_argument('--output',type=Path,default=Path(__file__).with_name('evidence.js'));a=p.parse_args()
    data=build(a.reports)
    a.output.write_text('/* Generated allowlisted evidence. Do not edit by hand. */\nwindow.RESEARCH_EVIDENCE = '+json.dumps(data,ensure_ascii=True,separators=(',',':'))+';\n',encoding='utf-8')
    print('Verified five report hashes; exported four trajectories and aggregate evidence.')
