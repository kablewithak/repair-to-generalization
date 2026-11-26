"""G10 matched 25K/50K characterization; not a rescue of G7-G9.

Original five compressed lessons and four accepted claim reviews are frozen.
Compress only added service scopes. A4_partial uses the same four reviewed
claims at both tiers; no fifth-review resampling or synthesized replacement.
All comparisons are diagnostic, not fresh confirmation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .experience_ledger import Access, canonical, execute, hash_value
from .g6_live import ACTION_ONLY_REVISION, parse_linked, payload_for, unique_object
from .g6_qualification import OUTPUT_LIMIT, FRAMING_MARGIN, PROJECT_CAP_USD, RESERVE_USD
from .g8_independent import TransportMeter, NETWORK_RECOVERIES
from .g10_50k_qualification import SCOPES, MAX_INPUT_BYTES
from .g10_scaling_source import load_inputs, BASE_SHA256, EXTENSION_SHA, COMPRESSION_SHA, CLAIMS_SHA
from .linked_world import CONTRACT, MAX_ACTIONS
from .savant_pilot import MODEL, INPUT_PER_MILLION, OUTPUT_PER_MILLION, PilotError, InvalidModelResponse, Usage, subq_request
from .study_design import combined_cases, split_manifest, stratum, RETRIEVAL
from .study_memory import CompressedLesson, MemoryPolicies, episodes

REVISION = "g10_matched_scaling_v1"
ARMS = ("A1","A2","A3","A4_partial")
TIERS = (25,50)
REPEATS = 3
ORDER_SUBSET = ("study-main-01","study-main-03","study-main-10")
MAX_RUN_USD = 6.0
MIN_PRIOR_USD = 3.22
# Historical preparation includes the failed initial compression and final claim review retry.
BASE_COMPRESSION_USD = .0028244 + .0084962
PARTIAL_REFLECTION_USD = .0163315 + .0008158
SYSTEM = (
    "Compress the public interaction history for this service into one short advisory lesson "
    "(at most 1600 characters, target 700). Include applicability, uncertainty and observed "
    "failure patterns where supported. Do not invent hidden facts or evaluator feedback. "
    "Select one actual representative supporting step alias; provenance does not prove every "
    "statement. Keep prose self-contained and return only the schema."
)


def cases():
    subset = split_manifest()["scaling_subset"]
    return tuple(c for c in combined_cases("main") if c.case_id in subset)


def order():
    rows = []
    for rep in range(1,REPEATS+1):
        for i,case in enumerate(cases()):
            conditions = [(t,a,"chronological") for t in TIERS for a in ARMS]
            if case.case_id in ORDER_SUBSET:
                conditions += [(t,"A3","reverse_episode_blocks") for t in TIERS]
            shift = (i+rep-1)%len(conditions)
            rows += [(rep,case.case_id,*cond) for cond in conditions[shift:]+conditions[:shift]]
    return tuple(rows)


MAX_CALLS = len(order())*MAX_ACTIONS + len(SCOPES) + NETWORK_RECOVERIES


def extra_payload(ledger, service):
    public, anchors = [], {}
    for group in episodes(ledger):
        if group[0]["body"]["service"]!=service:
            continue
        for i,event in enumerate(group):
            alias = f"e{event['episode']}_{i}"
            public.append(dict(event,event_id=alias))
            if event["kind"]=="step":
                anchors[alias] = event["event_id"]
    if not anchors:
        raise ValueError("scale_empty_compression_scope")
    payload = payload_for({"service":service,"public_interactions":public},ACTION_ONLY_REVISION)
    payload["messages"][0]["content"] = SYSTEM
    payload["response_format"]["json_schema"] = dict(name="scale_scope_compression",strict=True,schema={
        "type":"object","properties":{"lesson":{"type":"string"},
        "evidence":{"type":"string","enum":sorted(anchors)}},
        "required":["lesson","evidence"],"additionalProperties":False})
    return payload,anchors


def parse_extra(data, service, anchors):
    try:
        u=data["usage"]
        p,c=u["prompt_tokens"],u["completion_tokens"]
        k=(u.get("prompt_tokens_details") or {}).get("cached_tokens",0)
        if any(type(n) is not int or n<0 for n in (p,c,k)) or k>p:
            raise ValueError("usage")
        usage=Usage(p,k,c)
    except (KeyError,TypeError,ValueError,AttributeError) as error:
        raise InvalidModelResponse("scale_compression_usage") from error
    try:
        choices=data["choices"]
        if type(choices) is not list or len(choices)!=1:
            raise ValueError("choices")
        choice=choices[0]
        if choice["finish_reason"]=="length":
            raise InvalidModelResponse("output_limit",usage)
        if choice["finish_reason"]!="stop":
            raise ValueError("finish_reason")
        body=json.loads(choice["message"]["content"],object_pairs_hook=unique_object)
        if type(body) is not dict or set(body)!={"lesson","evidence"}:
            raise ValueError("fields")
        if type(body["lesson"]) is not str or not body["lesson"].strip() or len(body["lesson"])>1600:
            raise ValueError("lesson")
        if type(body["evidence"]) is not str or body["evidence"] not in anchors:
            raise ValueError("evidence")
        return CompressedLesson(service,body["lesson"],(anchors[body["evidence"]],)),usage
    except InvalidModelResponse:
        raise
    except (KeyError,TypeError,ValueError,IndexError) as error:
        raise InvalidModelResponse("scale_compression:"+str(error),usage) from error


class Policies:
    def __init__(self,base,full,lessons,claims,extras):
        if full.events[:len(base.events)]!=base.events:
            raise ValueError("scale_non_nested_history")
        self.ledgers={25:base,50:full}
        self.retrievers={t:MemoryPolicies(l,lessons) for t,l in self.ledgers.items()}
        if tuple(x.service for x in extras)!=SCOPES or len(claims)!=4:
            raise ValueError("scale_incomplete_snapshot")
        self.lessons={25:[x.public() for x in lessons],50:[x.public() for x in lessons+extras]}
        self.claims=json.loads(canonical(claims))
        self.reversed={t:[e for block in reversed(episodes(l)) for e in block] for t,l in self.ledgers.items()}

    def memory(self,tier,arm,ordering,public):
        if ordering=="reverse_episode_blocks":
            if arm!="A3":
                raise ValueError("scale_order_arm")
            return self.reversed[tier]
        if ordering!="chronological":
            raise ValueError("scale_order")
        if arm=="A1":
            return self.lessons[tier]
        if arm=="A2":
            return self.retrievers[tier].retrieve(public)
        if arm=="A3":
            return self.ledgers[tier].public()
        if arm=="A4_partial":
            return self.ledgers[tier].public()+self.claims
        raise ValueError("scale_arm")

    def freeze(self):
        return dict(ledger_heads={t:l.head for t,l in self.ledgers.items()},
                    source_hashes={t:hash_value(l.public()) for t,l in self.ledgers.items()},
                    source_events={t:len(l.events) for t,l in self.ledgers.items()},
                    source_json_bytes={t:len(canonical(l.public()).encode("utf-8")) for t,l in self.ledgers.items()},
                    lesson_snapshots=self.lessons,claims=self.claims,
                    reversed_hashes={t:hash_value(v) for t,v in self.reversed.items()},
                    retrieval=RETRIEVAL)


def setup(base,full,commit,prior,cap,extension_cost):
    return dict(record_type="g10_scaling_setup",revision=REVISION,commit=commit,
                source_reports=[BASE_SHA256,EXTENSION_SHA,COMPRESSION_SHA,CLAIMS_SHA],
                source_ledger_heads=[base.ledger.head,full.head],source_probes_first_input=[30823,51057],
                source_cost_usd={25:base.accounted_usd,50:base.accounted_usd+extension_cost},
                historical_compression_usd=BASE_COMPRESSION_USD,historical_reflection_usd=PARTIAL_REFLECTION_USD,
                cases=[asdict(c) for c in cases()],order=order(),order_subset=ORDER_SUBSET,
                model=MODEL,protocol=ACTION_ONLY_REVISION,reasoning_effort="low",
                prompt_sha256=hash_value([payload_for({},ACTION_ONLY_REVISION),CONTRACT]),
                extra_compression_sha256=hash_value([extra_payload(full,s)[0] for s in SCOPES]),
                max_calls=MAX_CALLS,max_actions=MAX_ACTIONS,max_input_bytes=MAX_INPUT_BYTES,
                max_output_tokens=OUTPUT_LIMIT,framing_margin=FRAMING_MARGIN,
                prior_spend_usd=prior,authorized_cap_usd=cap,project_cap_usd=PROJECT_CAP_USD,reserve_usd=RESERVE_USD,
                worst_case_usd=MAX_CALLS*((MAX_INPUT_BYTES+FRAMING_MARGIN)*INPUT_PER_MILLION
                    +OUTPUT_LIMIT*OUTPUT_PER_MILLION)/1e6,
                criterion="complete matched characterization at both tiers; no claim of positive learning "
                          "or transfer; preserve all negative outcomes",
                transport_policy="one fresh request for exact unknown-usage network failure, at most four "
                                 "in entire run; other failures stop; no semantic retries",
                claim_limit="A4_partial holds four prior accepted reviews only; no complete A4 or autonomous "
                            "repair claim. Same diagnostic subset exposed in G7; not independent confirmation. "
                            "Added service scopes can contain analogous evidence, not a pure token-length effect. "
                            "No shared A0 estimate or new relevant-evidence intervention.")


def summarize(rows, calls, acquisition, extra_cost):
    by_condition={}
    for tier in TIERS:
        for arm,ordering in [(a,"chronological") for a in ARMS]+[("A3","reverse_episode_blocks")]:
            key=f"{tier}:{arm}:{ordering}"
            rs=[r for r in rows if (r["tier"],r["arm"],r["ordering"])==(tier,arm,ordering)]
            cs=[c for c in calls if (c.get("tier"),c.get("arm"),c.get("ordering"))==(tier,arm,ordering)]
            safe=sum(r["outcome"]["safe_success"] for r in rs)
            cost=sum(c["accounted_usd"] for c in cs)
            preparation=acquisition[tier]+(BASE_COMPRESSION_USD+(extra_cost if tier==50 else 0)
                if arm=="A1" else PARTIAL_REFLECTION_USD if arm=="A4_partial" else 0)
            latencies=sorted(c["latency_ms"] for c in cs)
            known=[c["usage"] for c in cs if c["usage"] is not None]
            by_condition[key]=dict(episodes=len(rs),safe=safe,unsafe=sum(r["outcome"]["unsafe"] for r in rs),
                completed=sum(r["outcome"]["completed"] for r in rs),calls=len(cs),
                valid_calls=sum(c["error"] is None for c in cs),
                input_tokens=sum(u["prompt_tokens"] for u in known),
                cached_tokens=sum(u["cached_tokens"] for u in known),
                completion_tokens=sum(u["completion_tokens"] for u in known),
                max_input_tokens=max((u["prompt_tokens"] for u in known),default=0),
                median_latency_ms=statistics.median(latencies) if latencies else None,
                p95_latency_ms=latencies[math.ceil(.95*len(latencies))-1] if len(latencies)>=20 else None,
                accounted_decision_usd=cost,preparation_usd=preparation,
                usd_per_safe_success=cost/safe if safe else None,
                standalone_usd_per_safe_success=(cost+preparation)/safe if safe else None)
    indexed={(r["repetition"],r["case_id"],r["tier"],r["arm"],r["ordering"]):r["outcome"] for r in rows}
    if len(indexed)!=len(rows):
        raise ValueError("scale_duplicate_outcome")
    def contrast(left,right,subset):
        pairs=[]
        for rep in range(1,REPEATS+1):
            for case in cases():
                if case.case_id not in subset:
                    continue
                a=(rep,case.case_id,*left); b=(rep,case.case_id,*right)
                if a in indexed and b in indexed:
                    pairs.append((rep,case.case_id,indexed[a],indexed[b]))
        w=[(r,c) for r,c,a,b in pairs if a["safe_success"] and not b["safe_success"]]
        l=[(r,c) for r,c,a,b in pairs if b["safe_success"] and not a["safe_success"]]
        u=[(r,c) for r,c,a,b in pairs if a["unsafe"] and not b["unsafe"]]
        return dict(pairs=len(pairs),wins=w,losses=l,net_safe_wins=len(w)-len(l),additional_unsafe=u)
    ids=[c.case_id for c in cases()]
    complete=set(indexed)==set(order())
    return dict(scored_episodes=len(rows),planned_episodes=len(order()),comparison_complete=complete,
                by_condition=by_condition,
                scale_contrasts={a:contrast((50,a,"chronological"),(25,a,"chronological"),ids) for a in ARMS},
                order_contrasts={t:contrast((t,"A3","reverse_episode_blocks"),(t,"A3","chronological"),ORDER_SUBSET) for t in TIERS},
                preparation_note="Standalone costs charge shared acquisition/qualification and historical preparation "
                                 "once per hypothetical condition, not additive project spend; observed prep failures included.",
                g10_pass=False)


class Run:
    def __init__(self,policies,meter,emit,identity):
        self.policies,self.meter,self.emit,self.identity=policies,meter,emit,identity
        self.rows=[]

    def run(self):
        self.meter.parser=lambda d:parse_linked(d,ACTION_ONLY_REVISION)
        lookup={c.case_id:c for c in cases()}
        for i,(rep,cid,tier,arm,ordering) in enumerate(order(),1):
            case=lookup[cid]
            episode=83+i
            identity=dict(episode=episode,repetition=rep,case_id=cid,tier=tier,arm=arm,
                          ordering=ordering,stratum=stratum(case))
            self.identity.clear()
            self.identity.update(identity)
            self.emit(dict(record_type="g10_scaling_episode_started",**identity))
            step=0
            def decide(public):
                nonlocal step
                memory=self.policies.memory(tier,arm,ordering,public)
                decision=self.meter.decide(payload_for(dict(public,prior_interactions=memory),
                    ACTION_ONLY_REVISION),episode,step)
                step+=1
                return decision.action
            def observe(public):
                self.emit(dict(record_type="g10_scaling_public_step",**identity,step=step-1,**public))
            prior=self.policies.ledgers[tier]
            ledger,record=execute(case,prior,episode,False,Access.CURRENT,decide,observe=observe)
            moves=iter(record.actions)
            check,rebuilt=execute(case,prior,episode,False,Access.CURRENT,lambda _:next(moves))
            if ledger!=prior or check!=ledger or rebuilt!=record or next(moves,None) is not None:
                raise PilotError("scale_replay")
            row=dict(record_type="g10_scaling_episode_finished",**identity,record=asdict(record),
                     outcome=json.loads(record.outcome_json),replay_valid=True)
            self.rows.append(row)
            self.emit(row)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","extension","compression","claims","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--prior-spend-usd",type=float,required=True)
    parser.add_argument("--max-usd",type=float,default=5.0)
    parser.add_argument("--live",action="store_true")
    args=parser.parse_args()
    if not args.live:
        parser.error("explicit --live required")
    if not (math.isfinite(args.prior_spend_usd) and args.prior_spend_usd>=MIN_PRIOR_USD
            and math.isfinite(args.max_usd) and 0<args.max_usd<=MAX_RUN_USD
            and args.prior_spend_usd+args.max_usd+RESERVE_USD<=PROJECT_CAP_USD):
        parser.error("invalid expenditure envelope")
    base,full,lessons,claims,extension_cost=load_inputs(args.source,args.extension,args.compression,args.claims)
    key=os.getenv("SUBQ_API_KEY")
    if not key:
        parser.error("SUBQ_API_KEY missing")
    commit=subprocess.run(["git","rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
    manifest=setup(base,full,commit,args.prior_spend_usd,args.max_usd,extension_cost)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as stream:
        def emit(row):
            stream.write(canonical(row)+"\n")
            stream.flush()
        emit(dict(manifest,started_utc=datetime.now(timezone.utc).isoformat()))
        calls,identity,extras=[],{},[]
        def tagged(row):
            tagged_row=dict(row,**{k:v for k,v in identity.items() if k!="episode"})
            if row["record_type"]=="call_finished":
                calls.append(tagged_row)
            emit(tagged_row)
        meter=TransportMeter(args.max_usd,lambda p:subq_request(p,api_key=key),tagged,
                             max_calls=MAX_CALLS,max_input_bytes=MAX_INPUT_BYTES)
        failure=None
        run=None
        try:
            for i,service in enumerate(SCOPES,1):
                identity.clear()
                identity.update(stage="compression",service=service)
                request,anchors=extra_payload(full,service)
                meter.parser=lambda d,s=service,a=anchors:parse_extra(d,s,a)
                extras.append(meter.decide(request,i,0))
            policies=Policies(base.ledger,full,lessons,claims,tuple(extras))
            emit(dict(record_type="g10_scaling_memory_frozen",**policies.freeze()))
            run=Run(policies,meter,emit,identity)
            run.run()
        except PilotError as error:
            failure=str(error)
        extra_cost=sum(c["accounted_usd"] for c in calls if c.get("stage")=="compression")
        result=summarize(run.rows if run else [],calls,
                         {25:base.accounted_usd,50:base.accounted_usd+extension_cost},extra_cost)
        result.update(record_type="g10_scaling_terminal",error=failure,claim_limit=manifest["claim_limit"],
                      calls=meter.budget.calls,reported_usd=meter.reported_usd,
                      unknown_reserved_usd=meter.unknown_reserved_usd,accounted_usd=meter.budget.spent_usd,
                      additional_compression_usd=extra_cost,transport_recoveries=meter.recoveries,
                      report=str(args.output))
        emit(result)
    print(json.dumps(result,indent=2))
    return 0 if failure is None and result["comparison_complete"] else 1


if __name__=="__main__":
    raise SystemExit(main())
