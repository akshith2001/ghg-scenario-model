"""Reproduce the two full-year central integration checks (not the full study)."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from ghg_model.real_case import variant_profiles, base_parameters
from ghg_model.hourly_dispatch import cross_evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/energy_merge_validation.json')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = {}
    for year in (2024, 2025):
        path = root / f'data/energycharts_lu_de_{year}_hourly.csv'
        neighbours = root / f'data/energycharts_neighbours_{year}_hourly.csv'
        prof = variant_profiles(path)['regional']
        params = base_parameters(prof)
        run = cross_evaluate(prof, params)
        result[str(year)] = {
            'hours': prof.n_hours,
            'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, neighbours)},
            'parameters': asdict(params),
            'plans': {rule: plan.to_dict() for rule, plan in run['plans'].items()},
            'regret_share': run['regret_share'],
            'parameters_note': 'Default costs, without foresight derating or price feedback; integration validation only.',
        }
        print(f'{year}: {prof.n_hours} hours validated', flush=True)
    result['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((root / 'src/ghg_model').glob('*.py'))}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
