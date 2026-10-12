"""Full-size integer compute audit, NOT checkpoint or autonomous chip validation.

Fresh output only. Original training scripts and TestEngine are never invoked.
Each model writes a portable graph/input/quantization fixture and honest report.
--export-stages additionally writes zero-input hardware templates for chosen IDs.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from onnx2prim import __version__
from onnx2prim.project_snn_models import MODEL_IDS, FLOAT_QUANTIZATION_POLICY, build_project_graph
from onnx2prim.quantization import QuantizationConfig
from onnx2prim.temporal_compiler import compile_temporal, verify_temporal_behavior
from onnx2prim.temporal_export import save_temporal_graph, export_temporal_artifacts


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New directory; no overwrite')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--models', nargs='+', choices=MODEL_IDS, default=list(MODEL_IDS))
    parser.add_argument('--export-stages', nargs='*', choices=MODEL_IDS, default=[])
    args = parser.parse_args(argv)
    if args.steps < 1 or args.seed < 0:
        parser.error('Positive --steps and nonnegative --seed required')
    if len(set(args.models)) != len(args.models) or not set(args.export_stages) <= set(args.models):
        parser.error('Models must be unique; exported IDs must be selected')
    out = Path(args.output).resolve()
    if out.exists():
        parser.error('Choose a new output directory; existing results are preserved')
    out.mkdir(parents=True, exist_ok=False)
    report = {'version': __version__, 'scope': 'full-size explicit integer host-staged compute',
              'synthetic_parameters': True, 'real_checkpoints_loaded': False,
              'lossless_float_equivalence_verified': False, 'host_staging_required': True,
              'autonomous_chip_deployment': 'NOT implemented', 'hardware_exe_or_chip_run': False,
              'steps': args.steps, 'models': []}
    q = QuantizationConfig({})
    for index, model_id in enumerate(args.models):
        start = time.perf_counter()
        print(f'[{index+1}/{len(args.models)}] {model_id}: full-size integer compute', flush=True)
        options = {'quantization_policy': FLOAT_QUANTIZATION_POLICY} if model_id == 'snnlenet/train_lif_float' else {}
        # Fixed per-ID seeds are invariant under subset selection/reordering.
        seed = args.seed + MODEL_IDS.index(model_id)
        graph = build_project_graph(model_id, seed=seed, **options)
        rng = np.random.default_rng(seed + 2)
        low, high = (-1, 2) if graph.input_dtype == 'int2' else (-8, 9)
        frames = rng.integers(low, high, (args.steps,) + graph.input_shape, dtype=np.int64)
        directory = out / (f'{index:02d}_' + model_id.replace('/', '_'))
        save_temporal_graph(graph, directory)
        np.save(directory / 'input.npy', frames, allow_pickle=False)
        write_json(directory / 'quant.json', asdict(q))
        entry = {'model_id': model_id, 'directory': directory.name,
                 'input_shape': list(graph.input_shape), 'seed': seed, 'readout': graph.readout,
                 'coverage': graph.metadata['coverage'],
                 'approximations': graph.metadata.get('approximations', []),
                 'legacy_routing_verified': False, 'checkpoint_equivalence_verified': False,
                 'selector_status': graph.metadata.get('selector_inspection', {}).get('status', 'not applicable')}
        try:
            c = compile_temporal(graph, q, frames)
            actual = verify_temporal_behavior(c)
            entry.update(status='passed', compared_trace_count=len(actual),
                         kernel_count=sum(len(tiles) for tiles in c.kernels.values()),
                         partial_sum_kernel_count=len(c.reductions))
            if model_id in args.export_stages:
                export_temporal_artifacts(c, directory / 'compiled')
                entry['compiled_artifacts'] = directory.name + '/compiled'
            # Safe archive IDs; logical trace names reside only in JSON metadata.
            keys = {f'a{i:06d}': name for i, name in enumerate(actual)}
            np.savez(directory / 'reference_outputs.npz', **{key: c.reference_outputs[name] for key, name in keys.items()})
            np.savez(directory / 'behavior_outputs.npz', **{key: actual[name] for key, name in keys.items()})
            write_json(directory / 'trace_index.json', keys)
        except Exception as exc:
            entry.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        entry['seconds'] = round(time.perf_counter() - start, 3)
        report['models'].append(entry)
        report['completed_model_count'] = len(report['models'])
        report['all_selected_integer_compute_passed'] = len(report['models']) == len(args.models) and all(e['status'] == 'passed' for e in report['models'])
        report['selected_inventory_complete'] = set(args.models) == set(MODEL_IDS)
        write_json(out / 'report.json', report)
        write_json(directory / 'report.json', entry)
        print(f"  {entry['status']} ({entry['seconds']}s)", flush=True)
    print(f'Report: {out / "report.json"}', flush=True)
    print('This verifies integer compute only; host staging, quantization and routing gaps are explicit.')
    return 0 if report['all_selected_integer_compute_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
