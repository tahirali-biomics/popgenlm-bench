"""Release contracts: portable files, explicit reconstruction limits, pinned notebooks."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_public_manifest_terms_and_scope():
    manifest = json.loads((ROOT / 'data/benchmarks/v0.4/manifest.json').read_text())
    assert manifest['release'] == 'v0.4.0'
    assert 'Botanic' in manifest['raw_reconstruction']
    assert 'not rerun' in manifest['raw_reconstruction']
    assert all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in manifest['files'])
    terms = (ROOT / 'docs/v0.4_data_terms.md').read_text()
    assert 'Output Terms of Use' in terms and 'withholds' in terms


def test_public_notebooks_pin_same_archive_and_avoid_machine_paths():
    notebooks = sorted((ROOT / 'notebooks/v0.4').glob('*.ipynb'))
    assert len(notebooks) == 8
    for file in notebooks:
        n = json.loads(file.read_text())
        source = '\n'.join(''.join(c['source']) for c in n['cells'])
        assert 'releases/download/v0.4.0/popgenlm-v04-public-inputs.zip' in source
        assert '776c145febcc29d2c781a38c5a65b63fea2d5c647f0dd64d90e28d46f8676a9f' in source
        assert '/projects/' not in source and '/home/' not in source
        assert 'Botanic raw scores are omitted' in source
        assert all(c['execution_count'] is None and not c['outputs'] for c in n['cells'] if c['cell_type'] == 'code')


def test_promoted_asset_checksums():
    for line in (ROOT / 'data/benchmarks/v0.4/SHA256SUMS.txt').read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
