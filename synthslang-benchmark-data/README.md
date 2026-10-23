# Benchmark data

The benchmark contains 1,087 final human-accepted synthetic slang items:
387 Chinese, 400 Russian, and 300 Farsi entries. Each language has its own folder:

```text
fa/
  fa_terms.csv
  fa_benchmark_masked.csv
ru/
  ru_terms.csv
  ru_benchmark_masked.csv
zh/
  zh_terms.csv
  zh_benchmark_masked.csv
reconstruction.py
benchmark_manifest.json
```

Each `*_terms.csv` contains numeric `mask_id` and the original `term`, in original
row order. Each `*_benchmark_masked.csv` contains `mask_id` and all remaining
original fields, including definitions and sentence- and paragraph-level usage.
Match `mask_id` between the two files in the same folder to connect an entry.
The manifest records original column order, row counts, and SHA-256 checksums
used to verify reconstruction.

IDs are global, starting at 1, in Farsi, Russian, then Chinese file order. Exact,
case-sensitive occurrences of **any** listed term in the benchmark fields are
replaced with `[MASK#]`, including occurrences within longer strings. Longer terms
take priority when matches overlap. The term column lives only in `*_terms.csv`.

## Reconstruction

Run with Python 3; no third-party packages are required:

```sh
python3 reconstruction.py
```

This restores `fa.csv`, `ru.csv`, and `zh.csv` into the ignored `reconstructed/`
directory, preserving the original columns, row order, Unicode text, and CSV
formatting. Every file is checked against its original SHA-256 checksum before
writing. Existing files with different contents are never overwritten.
Use `--output-dir PATH` to choose another destination and `--data-dir PATH` to
read the split data from another directory. Keep reconstructed originals out of
the public repository; `.gitignore` excludes the default output directory.

## Slang categories

| Category | Language(s) | Description |
|---|---|---|
| `semantic_shift` | Chinese, Russian, Farsi | An ordinary word is assigned a new figurative meaning that is not directly predictable from its dictionary sense. |
| `morph_blend` | Chinese, Russian | Parts of multiple words or expressions are blended into a new slang term whose meaning goes beyond its components. |
| `affixation` | Chinese, Farsi | A productive colloquial prefix or suffix is attached to a base to create a new slang meaning. |
| `sentence_compression` | Chinese | A longer colloquial expression is compressed into a compact, four-character internet-style expression. |
| `suffixation` | Russian | An expressive Russian suffix is attached to an existing or invented root to create a new term. |
| `foreign_verb` | Russian | A foreign loanword is adapted into Russian verb morphology and given a slang use. |
| `light_verb` | Farsi | A noun or adjective combines with a Persian light verb to form a new idiomatic slang predicate. |

## English glosses

`sentence_in_context_gloss` and `paragraph_in_context_gloss` are model-generated
English renderings of the corresponding source-language contexts. They were
created to support human review and make the examples readable to people who do
not speak the source language. They are not human-authored gold translations,
and benchmark scoring does not use them as translation references.
