"""Fail-closed offline reconciliation of the frozen final benchmark.

No backend is instantiated and no provider or benchmark execution path is used.
Run: .venv/bin/python -m scripts.reconcile_benchmark
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from decision_optimizer.benchmark import (
    ALL_EXPERIMENT_CONFIGS, _evaluate, _implementation_hash, _problem_from_json,
    _problem_hash, _stage_accounting, _telemetry_from_dict, question_observations,
)
from decision_optimizer.dayplan import DayPlanSolution
from decision_optimizer.jev import JevDecision
from decision_optimizer.shift_schedule import ShiftScheduleSolution
from decision_optimizer.telemetry import estimate_openai_cost
from scripts.label_jev_fixtures import DEFAULT_OUTPUT, is_primary_scoring_eligible
from scripts.run_benchmark import load_benchmark_cases

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'benchmark_results/final_sol61_v2'
AUDIT_PATH = ROOT / 'docs/final_benchmark_integrity_audit_details.json'
VERSION = 'final-offline-reconciliation-v1'
FORMULATION_FIELDS = {'formulation_match', 'formulation_match_structural'}
FORMULATION_CASES = {'straightforward_staffing', 'infeasible_staffing_is_extracted_not_repaired'}
RECOVERED_PAIR = ('workout_before_four_hard', 'gpt-6-luna')


def require(condition, message):
    if not condition:
        raise ValueError('Reconciliation stopped: ' + message)


def read_jsonl(path):
    # Unlike resumable collection, do not ignore malformed/torn observations.
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(actual, expected):
    if isinstance(actual, float) or isinstance(expected, float):
        return isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
    return actual == expected


def verify_raw(raw_dir, audit):
    expected = {str(Path(name).relative_to('benchmark_results/final_sol61_v2')): digest
                for name, digest in audit['original_artifact_sha256'].items()}
    actual = {name: sha256(raw_dir / name) for name in expected}
    require(actual == expected, 'immutable artifact hashes differ from the integrity audit')
    require(len(actual) == 55, 'expected all 55 audited raw artifacts')
    return actual


def counts(rows):
    return {'rows': len(rows), 'status': dict(Counter(r['status'] for r in rows)),
            'execution_success_without_usable_output': sum(r['status'] == 'success' and not r['canonical_evaluation']['valid_output'] for r in rows),
            'terminal_failure_classifications': dict(Counter(r['failure_stage'] + '/' + r['failure_category'] for r in rows if r['status'] == 'terminal_failure'))}


def reconcile(raw_dir=RAW_DIR, output_dir=None):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir) if output_dir else raw_dir
    audit = json.loads(AUDIT_PATH.read_text())
    hashes = verify_raw(raw_dir, audit)
    rows = read_jsonl(raw_dir / 'latest.jsonl')
    derived = deepcopy(rows)
    logs = read_jsonl(raw_dir / 'jev_decisions.jsonl')
    require(len({log['decision_id'] for log in logs}) == len(logs) == 230, 'decision log coverage')
    cases = {case.case_id: case for case in load_benchmark_cases()}
    expected_cells = {(case, config.model, config.use_jev, config.solution_engine) for case in cases for config in ALL_EXPERIMENT_CONFIGS}
    actual_cells = [(r['case_id'], r['model'], r['configuration']['use_jev'], r['configuration']['solution_engine']) for r in rows]
    require(len(rows) == len(set(actual_cells)) == len({r['result_key'] for r in rows}) == 112 and set(actual_cells) == expected_cells, '112 unique cells required')
    require(counts(rows) == {'rows': 112, 'status': {'success': 103, 'terminal_failure': 9},
                           'execution_success_without_usable_output': 8,
                           'terminal_failure_classifications': {'extraction/timeout': 4, 'extraction/invalid_model_output': 4, 'solve_or_evaluate/timeout': 1}}, 'outcomes or failure classifications changed')
    # Read by recorded identities, never require new source fingerprints to match old observations.
    caches = {}
    cache_paths = {}
    for name in hashes:
        if name.startswith('cache/'):
            payload = json.loads((raw_dir / name).read_text())
            key = (payload['case_id'], payload['model'], payload['stage'])
            require(key not in caches and payload['schema_version'] == 2, 'duplicate or incompatible cache')
            caches[key] = payload
            cache_paths[key] = name
    require(len(caches) == 52, 'cache coverage')
    labels_artifact = json.loads(DEFAULT_OUTPUT.read_text())
    labels = {case['name']: {label['question_id']: label for label in case['expected_jev']} for case in labels_artifact['cases']}
    label_counts = Counter(label['review_status'] for items in labels.values() for label in items.values())
    require(label_counts == {'human_reviewed_approved': 54, 'generator_derived': 71, 'subjective_excluded_from_primary_scoring': 3}, 'frozen label policy changed')
    coverage = Counter()
    coverage_by_pair = []
    stage_ledger = []
    direct_stages = {}
    pair_observations = {}
    for case_id, model in sorted({(r['case_id'], r['model']) for r in rows}):
        case = cases[case_id]
        extraction = caches[(case_id, model, 'extraction')]
        extracted = _problem_from_json(case.domain, extraction.get('problem'))
        if 'terminal_error' not in extraction:
            require(extraction['problem_hash'] == _problem_hash(extracted), 'extraction cache hash')
        jev = caches.get((case_id, model, 'jev'))
        decisions = [] if jev is None else [JevDecision.model_validate(d) for d in jev['decisions']]
        if jev:
            require(jev['extraction_fingerprint'] == extraction['fingerprint'] and jev['extraction_problem_hash'] == extraction['problem_hash'], 'Jev extraction pairing')
            require(jev['requested_jev_model'] == 'jev-latest' and jev['resolved_jev_model'] == 'jev-1.13.0', 'Jev model provenance')
            matched_logs = [log for log in logs if (log['case_id'], log['model']) == (case_id, model)]
            require(len(matched_logs) == len(decisions), 'cache/log decision count')
            for decision in jev['decisions']:
                matching = [log for log in matched_logs if all(log.get(k) == v for k, v in decision.items())]
                require(len(matching) == 1 and matching[0]['jev_fingerprint'] == jev['fingerprint'] and matching[0]['resolved_jev_model'] == jev['resolved_jev_model'], 'cache/log evidence differs')
        observations = question_observations(case, extracted, decisions)
        pair_observations[(case_id, model)] = observations
        by_id = {o['canonical_question_id']: o for o in observations['observed_questions'] if o['canonical_question_id'] is not None}
        require(set(by_id) <= set(labels[case_id]), 'alignment has no frozen label')
        primary_ids = {key for key, label in labels[case_id].items() if is_primary_scoring_eligible(label)}
        local = {'canonical': len(labels[case_id]), 'aligned': len(by_id), 'missing': len(labels[case_id]) - len(by_id),
                 'unaligned_generated': observations['unaligned_generated'], 'unaligned_decisions': len(observations['unaligned_jev_decisions']),
                 'primary': len(primary_ids), 'primary_aligned': len(primary_ids & set(by_id)), 'primary_missing': len(primary_ids - set(by_id))}
        coverage.update(local)
        coverage_by_pair.append({'case_id': case_id, 'model': model, **local,
                                 'missing_ids': sorted(set(labels[case_id]) - set(by_id)),
                                 'unaligned_questions': [o for o in observations['observed_questions'] if o['canonical_question_id'] is None]})
        for stage, payload in [('extraction', extraction), ('jev', jev)]:
            if payload is not None:
                stage_ledger.append({'case_id': case_id, 'model': model, 'stage': stage,
                                     'source': cache_paths[(case_id, model, stage)],
                                     'telemetry': payload['telemetry']})
        for row in rows:
            if (row['case_id'], row['model']) != (case_id, model):
                continue
            if row['configuration']['solution_engine'] == 'direct_llm' and extracted is not None:
                require(row['status'] == 'success' or ((case_id, model) == RECOVERED_PAIR and row['configuration']['use_jev']), 'unexpected Direct failure')
                telemetry = deepcopy(row['standalone_architecture_telemetry_estimate'])
                # Shared stages occur in every standalone estimate; subtract each exactly once.
                fields = ['n_model_call_attempts', 'n_model_calls', 'openai_input_tokens', 'openai_output_tokens',
                          'estimated_cost', 'n_jev_call_attempts', 'n_jev_calls', 'n_jev_questions', 'jev_input_tokens']
                for shared in [extraction, jev if row['configuration']['use_jev'] else None]:
                    if shared:
                        for field in fields:
                            telemetry[field] -= shared['telemetry'][field]
                require(all(telemetry[k] >= -1e-12 for k in fields), 'negative stage accounting')
                telemetry['estimated_cost'] = round(telemetry['estimated_cost'], 10)
                telemetry['latency_total_s'] = 0  # no separate persisted latency claim for this derived ledger
                direct_stages[row['result_key']] = telemetry
                stage_ledger.append({'case_id': case_id, 'model': model, 'stage': 'direct_jev_' + str(row['configuration']['use_jev']).lower(),
                                     'source': 'latest.jsonl#' + row['result_key'], 'telemetry': {field: telemetry[field] for field in fields},
                                     'derivation': 'Standalone row less shared extraction and Jev; accounting fields only.'})
    require(dict(coverage) == audit['coverage_totals'], 'additional label coverage discrepancy')
    ledger_totals = Counter()
    for entry in stage_ledger:
        t = entry['telemetry']
        require(equal(t['estimated_cost'], estimate_openai_cost(entry['model'], t['openai_input_tokens'], t['openai_output_tokens'])), 'stage token cost differs')
        for field in ['n_model_call_attempts', 'n_model_calls', 'openai_input_tokens', 'openai_output_tokens', 'estimated_cost', 'n_jev_call_attempts', 'n_jev_calls', 'n_jev_questions', 'jev_input_tokens']:
            ledger_totals[field] += t[field]
    expected_totals = {**audit['reconciled_openai'], **audit['reconciled_jev'], 'estimated_cost': audit['reconciled_openai_estimated_cost']}
    require(all(equal(ledger_totals[k], v) for k, v in expected_totals.items()), 'unique stage totals differ from audit')
    corrections = []
    formulation_keys = []
    evidence_keys = []
    accounting_keys = []
    allocations = Counter()
    for raw, row in zip(rows, derived):
        case_id, model = row['case_id'], row['model']
        case = cases[case_id]
        extraction = caches[(case_id, model, 'extraction')]
        jev = caches.get((case_id, model, 'jev')) if row['configuration']['use_jev'] else None
        extracted = _problem_from_json(case.domain, extraction.get('problem'))
        recovered = row['status'] == 'terminal_failure' and (case_id, model) == RECOVERED_PAIR and row['configuration']['use_jev']
        final_json = jev['problem'] if recovered else row.get('final_problem')
        final_problem = _problem_from_json(case.domain, final_json)
        if extracted is not None:
            require(row['extraction_problem_hash'] == extraction['problem_hash'], 'row extraction hash')
            require(row['intermediate_reuse']['extraction_fingerprint'] == extraction['fingerprint'], 'row extraction fingerprint')
        else:
            require(row['extraction_problem_hash'] == (None if 'terminal_error' in extraction else extraction['problem_hash']), 'null extraction hash')
        if row['status'] == 'success':
            require(row['extraction']['problem'] == extraction['problem'], 'persisted extraction differs')
            require(final_json == (jev['problem'] if jev else extraction['problem']), 'shared final formulation differs')
            if jev:
                require(row['intermediate_reuse']['jev_fingerprint'] == jev['fingerprint'] and row['jev_decisions'] == jev['decisions'] and row['resolved_jev_model'] == jev['resolved_jev_model'], 'row Jev evidence differs')
        require(row['configuration']['model'] == model and row['resolved_openai_models'] == row['standalone_architecture_telemetry_estimate']['resolved_openai_models'] and all(m == model for m in row['resolved_openai_models']), 'requested/resolved OpenAI provenance')
        solution_json = row.get('solution')
        solution = None if solution_json is None else (DayPlanSolution if case.domain == 'dayplan' else ShiftScheduleSolution).model_validate(solution_json)
        replay = _evaluate(case, solution, final_problem, _telemetry_from_dict(row['standalone_architecture_telemetry_estimate'])).model_dump(mode='json')
        differences = {key: {'before': raw['canonical_evaluation'][key], 'after': value} for key, value in replay.items() if not equal(value, raw['canonical_evaluation'][key])}
        known = case_id in FORMULATION_CASES and not row['configuration']['use_jev']
        require(not differences or (known and set(differences) == FORMULATION_FIELDS and all(v == {'before': False, 'after': True} for v in differences.values())), 'additional evaluation discrepancy: ' + row['result_key'] + ' ' + str(differences))
        if differences:
            row['canonical_evaluation'].update({key: item['after'] for key, item in differences.items()})
            formulation_keys.append(row['result_key'])
        raw_alignment = question_observations(case, extracted, [JevDecision.model_validate(d) for d in raw['jev_decisions']])
        require(raw['question_alignment'] == raw_alignment, 'raw alignment replay differs')
        if recovered:
            row.update(final_problem=deepcopy(jev['problem']), jev_decisions=deepcopy(jev['decisions']), resolved_jev_model=jev['resolved_jev_model'],
                       question_alignment=deepcopy(pair_observations[(case_id, model)]))
            row['intermediate_reuse']['jev_fingerprint'] = jev['fingerprint']
            evidence_keys.append(row['result_key'])
        paid_raw = raw['actual_paid_call_attempts_created_for_this_row']
        stage_args = []
        solve = direct_stages.get(row['result_key'])
        extraction_attempts = paid_raw['openai_call_attempts'] - (0 if solve is None else solve['n_model_call_attempts'])
        require(extraction_attempts in (0, extraction['telemetry']['n_model_call_attempts']), 'extraction allocation')
        if extraction_attempts:
            stage_args.append((_telemetry_from_dict(extraction['telemetry']), 'openai', 0))
            allocations[(case_id, model, 'extraction')] += 1
        if solve is not None:
            stage_args.append((_telemetry_from_dict(solve), 'openai', 0))
        if paid_raw['jev_call_attempts']:
            require(jev is not None and paid_raw['jev_call_attempts'] == jev['telemetry']['n_jev_call_attempts'], 'Jev allocation')
            stage_args.append((_telemetry_from_dict(jev['telemetry']), 'jev', 0))
            allocations[(case_id, model, 'jev')] += 1
        paid = _stage_accounting(*stage_args)
        # Preserve original measured latency; the reconciliation makes no new latency estimate.
        paid.pop('new_stage_latency_s')
        if 'new_stage_latency_s' in paid_raw:
            paid['new_stage_latency_s'] = paid_raw['new_stage_latency_s']
        require(all(equal(paid[k], v) for k, v in paid_raw.items() if k != 'new_stage_latency_s'), 'unexpected recorded accounting discrepancy')
        row['actual_paid_call_attempts_created_for_this_row'] = paid
        accounting_keys.append(row['result_key'])
        changed_fields = {key: {'present': key in raw, 'value': deepcopy(raw.get(key))} for key in row if row[key] != raw.get(key)}
        row['reconciliation'] = {'version': VERSION, 'raw_result_key': raw['result_key'],
                                 'raw_values_superseded_for_analysis': changed_fields,
                                 'sources': [cache_paths[(case_id, model, 'extraction')]] + ([cache_paths[(case_id, model, 'jev')], 'jev_decisions.jsonl'] if jev else []),
                                 'paid_observations_generated_or_replaced': False}
        corrections.append({'result_key': row['result_key'], 'fields': sorted(changed_fields)})
    require(set(allocations) == set(caches) and all(n == 1 for n in allocations.values()), 'shared stages must be charged exactly once')
    require(len(formulation_keys) == 8 and len(evidence_keys) == 1, 'known corrections not reproduced')
    require(counts(derived) == counts(rows), 'outcome semantics changed')
    paid_totals = {key: sum(r['actual_paid_call_attempts_created_for_this_row'][key] for r in derived) for key in ['openai_call_attempts', 'openai_responses_with_metadata', 'openai_input_tokens', 'openai_output_tokens', 'openai_estimated_cost_usd', 'jev_call_attempts', 'jev_completed_calls', 'jev_input_tokens', 'jev_questions', 'openai_usage_unknown_calls', 'jev_usage_unknown_calls']}
    require(paid_totals == {**paid_totals, 'openai_call_attempts': 76, 'openai_responses_with_metadata': 74, 'openai_input_tokens': 86583, 'openai_output_tokens': 46461, 'jev_call_attempts': 24, 'jev_completed_calls': 24, 'jev_input_tokens': 50946, 'jev_questions': 230, 'openai_usage_unknown_calls': 2, 'jev_usage_unknown_calls': 0} and equal(paid_totals['openai_estimated_cost_usd'], .2874103), 'derived row accounting totals')
    paid_totals['openai_estimated_cost_usd'] = round(paid_totals['openai_estimated_cost_usd'], 10)
    paid_totals['timeout_billing_usd'] = None
    paid_totals['jev_cost_usd'] = None
    input_hashes = {**hashes, 'audit_details': sha256(AUDIT_PATH), 'audit_report': sha256(ROOT / 'docs/final_benchmark_integrity_audit.md')}
    auxiliary_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in [DEFAULT_OUTPUT, ROOT / 'tests/fixtures/jev_expected_review.tsv', ROOT / 'tests/fixtures/jev_benchmark_selection.json', ROOT / 'project_plan.md', ROOT / 'docs/benchmark_methodology.md', *sorted((ROOT / 'tests/fixtures').rglob('*.json'))]}
    source = {'version': VERSION, 'raw_schema_version': 2, 'derived_schema_version': 1,
              'implementation_hash': _implementation_hash(), 'script_sha256': sha256(Path(__file__)),
              'observed_implementation_hash': audit['provenance_implementation_hash']}
    manifest_path = output_dir / 'reconciliation_manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    timestamp = previous.get('timestamp') if previous.get('input_sha256') == input_hashes and previous.get('source') == source and previous.get('auxiliary_input_sha256') == auxiliary_hashes else datetime.now(timezone.utc).isoformat()
    manifest = {'timestamp': timestamp, 'source': source, 'input_sha256': input_hashes, 'auxiliary_input_sha256': auxiliary_hashes,
                'before': counts(rows), 'after': counts(derived), 'experiment_totals': paid_totals,
                'unique_stage_ledger': stage_ledger, 'label_counts': labels_artifact['review_summary'], 'coverage_totals': dict(coverage), 'coverage_by_pair': coverage_by_pair,
                'correction_rules': [
                    {'rule': 'normalize_shift_json_pydantic_before_formulation_comparison', 'affected_result_keys': formulation_keys},
                    {'rule': 'unique_stage_accounting_supersedes_incomplete_incremental_fields; shared stages charged once; unknown timeout usage stays unknown', 'affected_result_keys': accounting_keys},
                    {'rule': 'attach_completed_shared_jev_cache_and_logs_to_downstream_failure', 'affected_result_keys': evidence_keys}],
                'corrections': corrections,
                'field_counts_before_after': {field: {'before': sum(r['canonical_evaluation'][field] for r in rows), 'after': sum(r['canonical_evaluation'][field] for r in derived)} for field in sorted(FORMULATION_FIELDS)},
                'recovered_accounting_on_terminal_rows': [{'result_key': r['result_key'], 'known_usage': r['actual_paid_call_attempts_created_for_this_row']} for r in derived if r['status'] == 'terminal_failure' and (r['actual_paid_call_attempts_created_for_this_row']['openai_input_tokens'] or r['actual_paid_call_attempts_created_for_this_row']['jev_input_tokens'])],
                'validation': {'offline_replayed_rows': 112, 'unexpected_discrepancies': 0, 'raw_hashes_verified_before_and_after': 55},
                'paid_observations_generated_or_replaced': False, 'authoritative_phase_10_input': 'reconciled.jsonl',
                'limitations': ['Token costs are recorded estimates, not invoices; two timeout requests have unknown usage and billing.',
                                'Recovered failed Direct formulation is shared upstream evidence; the exact timed-out request body was not persisted.',
                                'No Jev dollar pricing is assigned.']}
    require(verify_raw(raw_dir, audit) == hashes, 'raw artifacts changed during reconciliation')
    # Write only after every validation has passed. These fixed new filenames never overwrite raw.
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'reconciled.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in derived))
    with (output_dir / 'reconciled_summary.csv').open('w', newline='') as handle:
        evaluation_fields = list(rows[0]['canonical_evaluation'])
        paid_fields = ['openai_call_attempts', 'openai_responses_with_metadata', 'openai_input_tokens', 'openai_output_tokens', 'openai_estimated_cost_usd', 'jev_call_attempts', 'jev_completed_calls', 'jev_input_tokens', 'jev_questions', 'openai_usage_unknown_calls']
        fields = ['result_key', 'case_id', 'architecture', 'status', *evaluation_fields, *['actual_' + f for f in paid_fields]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in derived:
            writer.writerow({**{key: row[key] for key in fields[:4]}, **row['canonical_evaluation'], **{'actual_' + f: row['actual_paid_call_attempts_created_for_this_row'][f] for f in paid_fields}})
    manifest['derived_sha256'] = {name: sha256(output_dir / name) for name in ['reconciled.jsonl', 'reconciled_summary.csv']}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    require(verify_raw(raw_dir, audit) == hashes, 'raw artifacts changed after writing')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='optional separate directory for derived outputs')
    args = parser.parse_args()
    manifest = reconcile(output_dir=args.output_dir)
    print(json.dumps({'rows': manifest['after']['rows'], 'experiment_totals': manifest['experiment_totals'], 'raw_artifacts_verified': 55}, indent=2))
