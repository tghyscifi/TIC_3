"""python -m onnx2prim MODEL --input INPUT.npy --quantization Q.json --output NEW_DIR [--verify]"""
import argparse
import json
import sys
import numpy as np
from .diagnostics import CompilerError, DependencyError
from .quantization import QuantizationConfig

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--input',required=True)
    parser.add_argument('--quantization',required=True)
    parser.add_argument('--output',required=True,help='New output directory; existing directories are rejected')
    parser.add_argument('--verify',action='store_true',help='Compare every primitive compute block with independent integer reference')
    backends = parser.add_mutually_exclusive_group()
    backends.add_argument('--graph',action='store_true',help='Use static ANN DAG backend with explicit node quantization')
    backends.add_argument('--snn',action='store_true',help='Use explicit com.tianjic LIF temporal compute backend')
    backends.add_argument('--temporal',action='store_true',help='Use full-size host-staged temporal DAG (ONNX or model.json snapshot)')
    args = parser.parse_args(argv)
    try:
        with open(args.quantization,encoding='utf-8') as f: raw_config=json.load(f)
        x=np.load(args.input,allow_pickle=False)
        if args.temporal:
            from pathlib import Path
            from .temporal_compiler import compile_onnx_temporal,compile_temporal,verify_temporal_behavior
            from .temporal_export import load_temporal_graph,export_temporal_artifacts
            q=QuantizationConfig(**raw_config)
            if Path(args.model).suffix.lower()=='.json':
                graph=load_temporal_graph(args.model,Path(args.model).with_name('weights.npz'))
                compilation=compile_temporal(graph,q,x)
            else: compilation=compile_onnx_temporal(args.model,q,x)
            if args.verify: verify_temporal_behavior(compilation)
            output=export_temporal_artifacts(compilation,args.output)
        elif args.snn:
            from .snn_compiler import compile_onnx_snn,verify_snn_behavior,export_snn_artifacts
            compilation=compile_onnx_snn(args.model,QuantizationConfig(**raw_config),x)
            if args.verify: verify_snn_behavior(compilation)
            output=export_snn_artifacts(compilation,args.output)
        elif args.graph:
            from .graph_compiler import compile_onnx_graph,verify_graph_behavior,export_graph_artifacts
            from .graph_reference import GraphQuantizationConfig
            compilation=compile_onnx_graph(args.model,GraphQuantizationConfig(**raw_config),x)
            if args.verify: verify_graph_behavior(compilation)
            output=export_graph_artifacts(compilation,args.output)
        else:
            from .compiler import compile_onnx,verify_behavior,export_artifacts
            compilation=compile_onnx(args.model,QuantizationConfig(**raw_config),x)
            if args.verify: verify_behavior(compilation)
            output=export_artifacts(compilation,args.output)
    except (CompilerError,DependencyError,OSError,TypeError,ValueError) as exc:
        print(str(exc),file=sys.stderr)
        return 2
    print(f'Compute prototype exported to {output}')
    print('Primitive behavior verified' if args.verify else 'Behavior verification NOT run')
    print('Router, cycle timing, hardware exe and chip verification NOT run')
    if args.snn: print('SNN JSON is a one-frame template; chip temporal input/replay driver NOT implemented')
    if args.temporal: print('Temporal DAG uses explicit HOST staging/window control; autonomous routed-chip execution NOT implemented')
    return 0

if __name__ == '__main__': raise SystemExit(main())
