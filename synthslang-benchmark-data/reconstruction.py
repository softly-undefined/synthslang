"""Restore the original language CSVs using only the Python standard library."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re


def reconstruct(data_dir, output_dir):
    manifest = json.loads((data_dir / 'benchmark_manifest.json').read_text())
    token = re.compile(r'\[MASK(\d+)\]')
    groups = {source['file']: [] for source in manifest['sources']}
    for source in manifest['sources']:
        name = source['file']
        if Path(name).name != name:
            raise ValueError(f'Invalid source filename: {name}')
        lang = Path(name).stem
        folder = data_dir / lang
        terms = {}
        with (folder / f'{lang}_terms.csv').open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                key = row['mask_id']
                if key in terms:
                    raise ValueError(f'Duplicate mask ID: {key}')
                terms[key] = row['term']
        with (folder / f'{lang}_benchmark_masked.csv').open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                restored = {
                    field: terms[row['mask_id']] if field == 'term'
                    else token.sub(lambda match: terms[match[1]], row[field])
                    for field in manifest['fields']
                }
                groups[name].append(restored)

    # Validate every output before writing any files.
    outputs = {}
    for source in manifest['sources']:
        name = source['file']
        if Path(name).name != name:
            raise ValueError(f'Invalid source filename: {name}')
        rows = groups[name]
        if len(rows) != source['rows']:
            raise ValueError(f'Row count mismatch: {name}')
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=manifest['fields'], lineterminator='\n', quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
        contents = stream.getvalue().encode('utf-8')
        if hashlib.sha256(contents).hexdigest() != source['sha256']:
            raise ValueError(f'Original checksum mismatch: {name}')
        outputs[name] = contents

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, contents in outputs.items():
        destination = output_dir / name
        if destination.exists():
            if destination.read_bytes() != contents:
                raise FileExistsError(f'Refusing to overwrite different contents: {destination}')
        else:
            destination.write_bytes(contents)
        print(f'Verified {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'reconstructed')
    args = parser.parse_args()
    reconstruct(args.data_dir, args.output_dir)
