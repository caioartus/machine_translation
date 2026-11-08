"""Summarise paired human error annotations produced by analyze_errors.py.

Run from the repository root, after annotating annotation_pairs.csv:
    python scripts/compare_annotations.py --annotations analysis/report/wmt22/annotation_pairs.csv \
        --baseline uniform_n5000_seed0 --systems length_n5000_seed0 joint_n5000_seed0

Only sentence pairs with completed annotations on BOTH outputs are compared.
ALT and OK mean acceptable (severity 0); error types can be separated by semicolons.
Counts describe the stratified annotation sample, not population error rates.
"""
import argparse
from pathlib import Path

import pandas as pd
from data_utils import assign_bins
from error_analysis_utils import unique_ids

ERROR_CODES = ['REP', 'OMIT', 'ADD', 'SENSE', 'NE', 'TERM', 'NUM', 'FLU', 'UNTR']
ACCEPTABLE = {'OK', 'ALT'}
GROUPS = [('overall', None), ('length', 'length_bin'), ('rarity', 'rarity_bin'), ('cell', 'cell')]


def read_annotations(path, key_path=None):
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    if key_path:
        key = pd.read_csv(key_path, keep_default_na=False, dtype=str)
        join = ['sample_id', 'id', 'system_code']
        if 'system' in df or not set(join).issubset(df) or not set(join + ['system']).issubset(key):
            raise ValueError('Anonymous sheet/key require sample_id, id, system_code; sheet must omit system')
        if key.sample_id.duplicated().any() or df.sample_id.duplicated().any():
            raise ValueError('Anonymous sample ids must be unique')
        df = df.merge(key[join + ['system']], on=join, how='left', validate='one_to_one')
        if df.system.isna().any():
            raise ValueError('Anonymous sheet contains rows absent from the key')
    required = {'id', 'system', 'zh', 'ref', 'hyp', 'length_bin', 'rarity_bin', 'cell',
                'error_type', 'severity', 'notes'}
    missing = required - set(df)
    if missing:
        raise ValueError(f'Missing annotation columns: {sorted(missing)}')
    if df.duplicated(['id', 'system']).any():
        raise ValueError('Each (id, system) must appear exactly once')
    codes, severities, completed = [], [], []
    for row in df.itertuples(index=False):
        label, severity = row.error_type.strip(), row.severity.strip()
        context = f'{row.id} / {row.system}'
        if not label and not severity:
            codes.append(set()); severities.append(None); completed.append(False)
            continue
        if not label or not severity:
            raise ValueError(f'{context}: fill both error_type and severity, or leave both blank')
        values = {v.strip().upper() for v in label.split(';') if v.strip()}
        if not values or values - set(ERROR_CODES) - ACCEPTABLE:
            raise ValueError(f'{context}: invalid error code(s): {label}')
        if values & ACCEPTABLE and len(values) != 1:
            raise ValueError(f'{context}: OK or ALT must be used alone')
        if severity not in {'0', '1', '2', '3'}:
            raise ValueError(f'{context}: severity must be 0, 1, 2 or 3')
        sev = int(severity)
        if (values <= ACCEPTABLE) != (sev == 0):
            raise ValueError(f'{context}: OK/ALT require severity 0; errors require severity 1-3')
        codes.append(values); severities.append(sev); completed.append(True)
    df['codes'], df['severity_value'], df['completed'] = codes, severities, completed
    return df


def compare(df, baseline, systems):
    available = set(df['system'])
    if baseline not in available:
        raise ValueError(f'Baseline not found: {baseline}')
    if not systems or set(systems) - available or baseline in systems:
        raise ValueError('Choose at least one available system different from the baseline')
    base = df[df.system == baseline].set_index('id')
    details, summaries, types = [], [], []
    for name in systems:
        after = df[df.system == name].set_index('id')
        # Include missing rows in the coverage denominator, but never treat them as correct.
        ids = base.index.union(after.index, sort=False)
        pairs = []
        for sid in ids:
            if sid not in base.index or sid not in after.index:
                continue
            before, now = base.loc[sid], after.loc[sid]
            for col in ['zh', 'ref', 'length_bin', 'rarity_bin', 'cell']:
                if before[col] != now[col]:
                    raise ValueError(f'{sid}: mismatched {col} between {baseline} and {name}')
            if not before.completed or not now.completed:
                continue
            old = before.codes - ACCEPTABLE
            new = now.codes - ACCEPTABLE
            status = ('persistent_error' if new else 'fixed') if old else ('introduced' if new else 'both_correct')
            pairs.append({
                'id': sid, 'baseline': baseline, 'system': name,
                'length_bin': before.length_bin, 'rarity_bin': before.rarity_bin, 'cell': before.cell,
                'zh': before.zh, 'ref': before.ref, 'hyp_before': before.hyp, 'hyp_after': now.hyp,
                'notes_before': before.notes, 'notes_after': now.notes,
                'errors_before': ';'.join(sorted(old)), 'errors_after': ';'.join(sorted(new)),
                'severity_before': int(before.severity_value), 'severity_after': int(now.severity_value),
                'transition': status, 'removed_types': ';'.join(sorted(old - new)),
                'added_types': ';'.join(sorted(new - old)),
            })
        details.extend(pairs)
        for kind, column in GROUPS:
            if column is None:
                groups = [('all', pairs, len(ids))]
            else:
                labels = sorted(set(base[column]) | set(after[column]))
                groups = []
                for label in labels:
                    group_ids = set(base.index[base[column] == label]) | set(after.index[after[column] == label])
                    groups.append((label, [p for p in pairs if p[column] == label], len(group_ids)))
            for label, rows, total in groups:
                rec = {'baseline': baseline, 'system': name, 'group_type': kind, 'group': label,
                       'n_sampled': total, 'n_paired': len(rows), 'n_unpaired': total - len(rows)}
                for status in ['fixed', 'introduced', 'persistent_error', 'both_correct']:
                    rec[status] = sum(r['transition'] == status for r in rows)
                rec['errors_before'] = rec['fixed'] + rec['persistent_error']
                rec['errors_after'] = rec['introduced'] + rec['persistent_error']
                rec['net_errors_removed'] = rec['fixed'] - rec['introduced']
                rec['severity_lower'] = sum(r['severity_after'] < r['severity_before'] for r in rows)
                rec['severity_higher'] = sum(r['severity_after'] > r['severity_before'] for r in rows)
                rec['severity_same'] = sum(r['severity_after'] == r['severity_before'] for r in rows)
                rec['error_share_before'] = rec['errors_before'] / len(rows) if rows else None
                rec['error_share_after'] = rec['errors_after'] / len(rows) if rows else None
                rec['rate_scope'] = 'completed_paired_stratified_sample_not_population'
                summaries.append(rec)
                for code in ERROR_CODES:
                    old = [code in r['errors_before'].split(';') for r in rows]
                    new = [code in r['errors_after'].split(';') for r in rows]
                    types.append({**{k: rec[k] for k in ['baseline', 'system', 'group_type', 'group', 'n_paired']},
                                  'error_type': code, 'before': sum(old), 'after': sum(new),
                                  'removed': sum(a and not b for a, b in zip(old, new)),
                                  'introduced': sum(b and not a for a, b in zip(old, new)),
                                  'persistent': sum(a and b for a, b in zip(old, new))})
                    types[-1]['before_share'] = sum(old) / len(rows) if rows else None
                    types[-1]['after_share'] = sum(new) / len(rows) if rows else None
    detail_columns = ['id', 'baseline', 'system', 'length_bin', 'rarity_bin', 'cell', 'zh', 'ref',
                      'hyp_before', 'hyp_after', 'notes_before', 'notes_after', 'errors_before', 'errors_after', 'severity_before',
                      'severity_after', 'transition', 'removed_types', 'added_types']
    return pd.DataFrame(details, columns=detail_columns), pd.DataFrame(summaries), pd.DataFrame(types)


def population_rates(details, summary, features):
    """Poststratify only fully judged pairs with every population cell represented.

    Assumes the sheet is the original within-cell random sample (not worst cases).
    No estimate is produced if annotation coverage is incomplete in any cell.
    """
    unique_ids(features, "population features")
    if not set(details.id).issubset(set(features.id)):
        raise ValueError("annotation ids are absent from population features")
    binned = assign_bins(features)
    expected_cells = binned.set_index("id")["cell"]
    if not details.cell.eq(details.id.map(expected_cells)).all():
        raise ValueError("annotation cells disagree with population features for the same ids")
    population = binned["cell"].value_counts()
    records = []
    for (baseline, system), block in summary.groupby(["baseline", "system"], sort=False):
        cells = block[block.group_type == "cell"].set_index("group")
        complete = all(c in cells.index and cells.loc[c, "n_paired"] > 0 and
                       cells.loc[c, "n_paired"] == cells.loc[c, "n_sampled"] for c in population.index)
        pairs = details[(details.baseline == baseline) & (details.system == system)]
        if not set(cells.index).issubset(set(population.index)):
            raise ValueError("annotation cells are absent from population features")
        for code in ["ANY"] + ERROR_CODES:
            record = {"baseline": baseline, "system": system, "error_type": code,
                      "population_n": int(population.sum()), "sample_n": len(pairs),
                      "status": "estimated" if complete else "not_estimated_incomplete_cell_annotations"}
            if complete:
                for side in ["before", "after"]:
                    estimate = 0.
                    for cell, count in population.items():
                        values = pairs[pairs.cell == cell][f"errors_{side}"]
                        flag = values.ne("") if code == "ANY" else values.map(lambda v: code in v.split(";"))
                        estimate += count / population.sum() * flag.mean()
                    record[f"weighted_error_share_{side}"] = estimate
                record["weighted_delta"] = record["weighted_error_share_after"] - record["weighted_error_share_before"]
            records.append(record)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--key', help='annotation_key.csv when reading an anonymous sheet')
    parser.add_argument('--population_features', help='optional matching full-test feature CSV for poststratification; random sample only')
    parser.add_argument('--baseline', required=True, help='exact run name, not a seed glob')
    parser.add_argument('--systems', nargs='+', required=True, help='exact run names after the method')
    parser.add_argument('--out_dir', help='default: <annotation folder>/comparison_<baseline>')
    args = parser.parse_args()
    try:
        details, summary, types = compare(read_annotations(args.annotations, args.key), args.baseline, args.systems)
    except ValueError as error:
        parser.error(str(error))
    out = Path(args.out_dir) if args.out_dir else Path(args.annotations).parent / f'comparison_{args.baseline}'
    out.mkdir(parents=True, exist_ok=True)
    details.to_csv(out / 'paired_errors.csv', index=False)
    summary.to_csv(out / 'error_transitions.csv', index=False)
    types.to_csv(out / 'error_types.csv', index=False)
    if args.population_features:
        population_rates(details, summary, pd.read_csv(args.population_features)).to_csv(
            out / 'population_error_estimates.csv', index=False)
    print(summary[summary.group_type == 'overall'].to_string(index=False))
    print('Only completed pairs count. Stratified sample counts are not full-test error rates.')
    print(f'Saved paired_errors.csv, error_transitions.csv and error_types.csv in {out}')


if __name__ == '__main__':
    main()
