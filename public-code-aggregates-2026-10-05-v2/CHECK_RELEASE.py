"""Standard-library integrity check, not empirical reconstruction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys


def verify(root: Path) -> dict:
    manifest = json.loads((root / 'release-manifest.json').read_text())
    versions = ('public-code-aggregates-2026-10-05-v1', 'public-code-aggregates-2026-10-05-v2')
    if manifest.get('version') not in versions:
        raise ValueError('Unknown release version')
    for key in ('raw_or_row_level_data_included', 'trained_models_included',
                'full_empirical_pipeline_portable', 'new_independent_confirmation'):
        if manifest.get(key) is not False:
            raise ValueError('Evidence boundary upgraded: ' + key)
    expected = {'release-manifest.json'}
    for row in manifest['files']:
        name = row['file']
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or str(rel) != name or '.git' in rel.parts or name in expected:
            raise ValueError('Invalid or duplicate path: ' + name)
        expected.add(name)
        path = root / name
        if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
            raise ValueError('Symlink not allowed: ' + name)
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Payload mismatch: ' + name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p.relative_to(root).parts[0] != '.git'}
    if actual != expected:
        raise ValueError('Unexpected/missing payload: ' + repr(sorted(actual ^ expected)))
    commands = [
        ('at', ['release-bvival/check_source_bundle.py', '--package-root', '.']),
        ('historical-primary', ['release-bvival/check_historical_reconstruction_package.py', '--package-root', '.']),
        ('historical-comparators', ['release-bvival/historical_comparator_package.py', '--package-root', '.']),
    ]
    if manifest['version'] == versions[1]:
        if (manifest.get('public_payload_download_verified') is not False or
                manifest.get('original_software_license_does_not_relicense_aggregate_evidence') is not True):
            raise ValueError('Prepared release or license scope boundary upgraded')
        commands.append(('source-inspection', ['release-bvival/check_source_bundle.py', '--package-root', '.']))
    for folder, args in commands:
        subprocess.run([sys.executable, '-I', '-B', '-S', *args],
                       cwd=root / 'recipes' / folder, check=True)
    return {'status': 'PREPARED_PAYLOAD_INTEGRITY_VERIFIED_NOT_PUBLIC_ACCESS_OR_EMPIRICAL_RETRAINING',
            'payload_files': len(expected) - 1, 'recipe_packages': len(commands)}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
