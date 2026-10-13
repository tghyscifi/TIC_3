"""Export an existing generated suite into a NEW ONNX suite and verify its oracle.

This never loads original checkpoints or imports training code. Source suite
snapshots must use explicit integer parameter mode and already contain Cut nodes.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from onnx2prim.project_snn_models import MODEL_IDS
from onnx2prim.temporal_export import load_temporal_graph
from onnx2prim.temporal_onnx import save_temporal_onnx
from onnx2prim.temporal_parser import parse_onnx_temporal
from onnx2prim.temporal_reference import run_temporal_reference
from onnx2prim.quantization import QuantizationConfig


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-suite',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    source=Path(args.source_suite).resolve();out=Path(args.output).resolve()
    if out.exists():parser.error('Output directory must be new')
    records=json.loads((source/'report.json').read_text(encoding='utf-8'))['models']
    out.mkdir(parents=True,exist_ok=False)
    report={'scope':'explicit integer ONNX contract roundtrip, not raw training import',
            'source_suite':str(source),'checkpoint_loaded':False,'chip_deployment_verified':False,'models':[]}
    for i,record in enumerate(records):
        model_id=record['model_id'];directory=(source/record['directory']).resolve()
        if model_id not in MODEL_IDS or directory.parent!=source:raise ValueError('Invalid source suite record')
        graph=load_temporal_graph(directory/'model.json',directory/'weights.npz')
        q=QuantizationConfig(**json.loads((directory/'quant.json').read_text(encoding='utf-8')))
        if q.cuts or q.parameter_mode!='integer':raise ValueError('Require explicit Cut nodes and integer parameters')
        frames=np.load(directory/'input.npy',allow_pickle=False)
        destination=out/(f'{i:02d}_'+model_id.replace('/','_'));destination.mkdir()
        save_temporal_onnx(graph,destination/'model.onnx')
        restored=parse_onnx_temporal(destination/'model.onnx',q)
        expected=run_temporal_reference(graph,q,frames);actual=run_temporal_reference(restored,q,frames)
        for key,value in expected.items():np.testing.assert_array_equal(actual[key],value)
        np.save(destination/'input.npy',frames,allow_pickle=False)
        (destination/'quant.json').write_text(json.dumps(asdict(q),indent=2),encoding='utf-8')
        report['models'].append({'model_id':model_id,'directory':destination.name,
            'integer_roundtrip_reference_verified':True,'compared_trace_count':len(expected),
            'primitive_behavior_verification':'not run by this exporter'})
        print(model_id+': ONNX integer-contract roundtrip passed',flush=True)
    report['all_inventory_families_exported']=set(r['model_id'] for r in report['models'])==set(MODEL_IDS)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0

if __name__=='__main__':raise SystemExit(main())
