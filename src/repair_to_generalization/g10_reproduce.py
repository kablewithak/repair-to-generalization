"""Offline reproduction of the observed G10 comparison prefix.

Never calls a model, never resumes the interrupted experiment. Reconstructs
source inputs, compression, memory rendering, requests, realized transitions,
costs and the terminal paired summary. The report hash identifies the exact
private evidence; it is not a provider signature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experience_ledger import canonical
from .g8_network_resume import Prefix, require, same
from .g9_development_source import FailedSourceReplay, FAILURE
from .g10_scaling_source import load_inputs, read_report
from .g10_scaling import Policies, Run, SCOPES, extra_payload, parse_extra, setup, summarize
from .savant_pilot import PilotError

REPORT_SHA = "21127dc6abd6d6d57f41786ff0276102b6239ce35e61b51439b861439ca209bf"


class TaggedReplay(FailedSourceReplay):
    def __init__(self,prefix,identity):
        super().__init__(prefix)
        self.identity=identity
        self.finished_calls=[]

    def take(self,kind):
        row=super().take(kind)
        if kind in ("call_started","call_finished"):
            tags={k:v for k,v in self.identity.items() if k!="episode"}
            require(all(row.get(k)==v for k,v in tags.items()),"release_call_tags")
            if kind=="call_finished":
                self.finished_calls.append(row)
            return {k:v for k,v in row.items() if k not in tags}
        return row


def verify(base,full,lessons,claims,extension_cost,rows,sha):
    manifest,terminal=rows[0],rows[-1]
    expected=setup(base,full,manifest["commit"],manifest["prior_spend_usd"],
                   manifest["authorized_cap_usd"],extension_cost)
    require(set(manifest)==set(expected)|{"started_utc"}
            and all(same(manifest[k],v) for k,v in expected.items()),"release_setup")
    require(terminal["record_type"]=="g10_scaling_terminal"
            and terminal["error"]==FAILURE and terminal["g10_pass"] is False
            and terminal["comparison_complete"] is False
            and terminal["transport_recoveries"]==0,"release_terminal")
    prefix=Prefix(sha,manifest,tuple(rows[1:-1]),terminal)
    identity={}
    meter=TaggedReplay(prefix,identity)
    extras=[]
    for i,service in enumerate(SCOPES,1):
        identity.clear()
        identity.update(stage="compression",service=service)
        request,anchors=extra_payload(full,service)
        meter.parser=lambda data,s=service,a=anchors:parse_extra(data,s,a)
        extras.append(meter.decide(request,i,0))
    policies=Policies(base.ledger,full,lessons,claims,tuple(extras))
    meter.emit(dict(record_type="g10_scaling_memory_frozen",**policies.freeze()))
    run=Run(policies,meter,meter.emit,identity)
    try:
        run.run()
    except PilotError as error:
        require(str(error)==FAILURE,"release_wrong_failure")
    else:
        raise ValueError("release_missing_failure")
    extra=sum(c["accounted_usd"] for c in meter.finished_calls if c.get("stage")=="compression")
    summary=summarize(run.rows,meter.finished_calls,
                      {25:base.accounted_usd,50:base.accounted_usd+extension_cost},extra)
    require(all(same(terminal[k],v) for k,v in summary.items()),"release_summary")
    require(same(terminal["additional_compression_usd"],extra),"release_compression_cost")
    return dict(report_sha256=sha,offline_replay_valid=True,model_calls_made=0,
                verified_recorded_calls=meter.calls,accounted_usd=meter.spent_usd,
                error=terminal["error"],**summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","extension","compression","claims","report"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    base,full,lessons,claims,cost=load_inputs(args.source,args.extension,args.compression,args.claims)
    rows=read_report(args.report,REPORT_SHA)
    result=verify(base,full,lessons,claims,cost,rows,REPORT_SHA)
    print(json.dumps(result,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
