# Contributing

FCR is research software. Correct scientific boundaries are part of code quality.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check .
pytest
```

## Pull requests

A PR should state:

- what hypothesis or infrastructure it changes;
- whether it changes any metric or data inclusion rule;
- which tests were added;
- whether results from previous commits remain comparable;
- the evidence level of any new numerical result.

Do not combine a methodological change and a headline result in a way that hides the effect of the change.

## Real-data changes

Any MICrONS/H01 analysis PR must pin dataset provenance and must not contain credentials, raw large datasets or undocumented manual edits.

## Scientific disagreements

Prefer a new baseline, null model, sensitivity analysis or falsification test over argumentative wording. The repository should make it easy for a skeptical researcher to disprove its central hypothesis.
