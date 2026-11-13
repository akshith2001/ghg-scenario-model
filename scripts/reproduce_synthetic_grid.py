"""Reproduce corrected studies in a separate directory with versioned provenance.

Run from the repository root: python scripts/reproduce_synthetic_grid.py
"""
import hashlib
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version

from ghg_model import synthetic_grid as sg


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root / 'output/synthetic_grid_reviewed')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace an existing run')
    args = parser.parse_args()
    output = args.output
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error('Output is not empty; choose --output for a fresh run or explicitly use --overwrite')
    output.mkdir(parents=True, exist_ok=True)
    data = root / 'data/real/bdg2_hospitality_daily.csv'
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    python=sys.version, platform=platform.platform(), epochs=300,
                    seeds=[0,1,2,3,4], rolling_origins=list(sg.ROLL_ORIGINS),
                    versions={p:version(p) for p in ['numpy','scipy','scikit-learn']},
                    runner_sha256=digest(Path(__file__)),
                    source_sha256=digest(Path(sg.__file__)), data_sha256=digest(data),
                    data_lf_sha256=hashlib.sha256(data.read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
                    completed=[])
    def save():
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    for study in ['simulated','real','rolling']:
        sg.main(['--study',study,'--epochs','300','--real-data',str(data),
                 '--output',str(output/study)])
        manifest['completed'].append(study)
        save()
    manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
    save()


if __name__ == '__main__':
    main()
