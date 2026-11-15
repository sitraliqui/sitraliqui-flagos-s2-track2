"""Rebuild the review archive with its recorded, preinstalled build dependencies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def package_payload(wheel, package):
    with zipfile.ZipFile(wheel) as archive:
        if archive.testzip() is not None:
            raise ValueError(f'Corrupt wheel: {wheel}')
        return {name: archive.read(name) for name in archive.namelist()
                if name.startswith(package + '/') and not name.endswith('/')}


def build(project, python, row, output):
    dep_names = list(row['build_dependencies'])
    check = ('import importlib.metadata as m,json; '
             f'print(json.dumps({{n:m.version(n) for n in {dep_names!r}}}))')
    installed = json.loads(subprocess.check_output([str(python), '-B', '-c', check], text=True))
    if installed != row['build_dependencies']:
        raise ValueError(f'{project}: expected {row["build_dependencies"]}, got {installed}')
    reference = ROOT / row['wheel']
    if digest(reference) != row['sha256']:
        raise ValueError(f'{project}: invalid reference wheel')
    package = {'vllm-plugin-FL': 'vllm_fl', 'FlagGems': 'flag_gems'}[project]
    with tempfile.TemporaryDirectory(prefix='source-', dir=output) as temporary:
        source = Path(temporary) / project
        shutil.copytree(ROOT / project, source)
        env = dict(os.environ, SETUPTOOLS_SCM_PRETEND_VERSION=row['version'], VLLM_VENDOR='',
                   SOURCE_DATE_EPOCH='1794182400', PYTHONDONTWRITEBYTECODE='1',
                   PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_INDEX='1')
        invocation = f'from setuptools.build_meta import build_wheel; build_wheel({str(output)!r})'
        with (output / f'{project}.build.log').open('xb') as log:
            subprocess.run([str(python), '-B', '-c', invocation], cwd=source, env=env,
                           stdout=log, stderr=subprocess.STDOUT, timeout=360, check=True)
    built = output / reference.name
    actual = package_payload(built, package)
    expected = package_payload(reference, package)
    if actual != expected:
        mismatch = sorted(name for name in actual.keys() | expected.keys() if actual.get(name) != expected.get(name))
        raise ValueError(f'{project}: rebuilt package differs: {mismatch}')
    return {'project': project, 'wheel': built.name, 'sha256': digest(built),
            'reference_archive_byte_identical': digest(built) == row['sha256'],
            'package_files_identical': len(actual)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin-python', type=Path, required=True)
    parser.add_argument('--gems-python', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    interpreters = {'vllm-plugin-FL': args.plugin_python.resolve(), 'FlagGems': args.gems_python.resolve()}
    receipt = json.loads((ROOT / 'provenance/build-and-install.json').read_text(encoding='utf-8'))
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = [build(row['project'], interpreters[row['project']], row, output) for row in receipt['builds']]
    (output / 'rebuild-receipt.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
