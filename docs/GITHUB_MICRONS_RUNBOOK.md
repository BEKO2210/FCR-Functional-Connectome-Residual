# GitHub-only MICrONS E0 Pilot

This path runs the first real MICrONS plumbing experiment entirely on GitHub Actions. No local Python environment is required.

> Evidence level: **E0 data plumbing only.** A successful run validates access and export plumbing; it is not evidence for FCR H1-H3.

## 1. Add the CAVE token as a GitHub repository secret

Acquire your CAVE token through the official MICrONS/CAVE flow. Do not paste the token into an issue, pull request, chat, workflow input, repository file, or command line.

In this repository, open:

`Settings -> Secrets and variables -> Actions -> New repository secret`

Create exactly this secret:

```text
Name: CAVE_TOKEN
Value: <your CAVE token>
```

FCR has no API path that stores this token in Git. GitHub provides the secret only to the credential-setup step of the manually triggered workflow.

## 2. Run the 100-neuron pilot

Open:

`Actions -> MICrONS E0 Pilot -> Run workflow`

Use:

```text
materialization_version: 1822
max_nodes: 100
```

The workflow runs, in order:

1. Python 3.12 setup;
2. FCR + pinned CAVEclient installation;
3. ephemeral `~/.cloudvolume/secrets/cave-secret.json` creation from `CAVE_TOKEN`;
4. `fcr microns-doctor --version 1822`;
5. bounded 100-neuron export;
6. independent `fcr microns-validate`;
7. run-summary generation;
8. artifact upload;
9. credential-file deletion with `if: always()`.

The runner itself is ephemeral and is discarded after the job.

## 3. What success produces

The Actions run summary records:

- pinned materialization version and timestamp;
- CAVEclient version;
- node count;
- directed candidate-pair count;
- connected-pair count;
- total observed synapse count;
- export runtime;
- NPZ SHA-256.

For exactly 100 valid nodes, the candidate graph must contain exactly **9,900** directed non-self pairs.

The uploaded artifact contains:

```text
microns_v1822_pilot_100.npz
microns_v1822_pilot_100.provenance.json
doctor.json
validation.json
run-summary.json
export_runtime_seconds.txt
```

Artifacts are retained for 30 days by the workflow. They contain no CAVE token.

## 4. Only then run 500 nodes

Run the same workflow again with:

```text
materialization_version: 1822
max_nodes: 500
```

Do not use the 500-node option to bypass a failed 100-node run. A schema/authentication/integrity failure must be fixed first.

## Security properties

The workflow has only `contents: read` GitHub permissions. `CAVE_TOKEN` is scoped only to the two steps that verify its existence and create the ephemeral credential file. Doctor, export, validation, summarization, and artifact upload do not receive the token as an environment variable.

The credential JSON is never uploaded. It is explicitly deleted in an `always()` cleanup step, and the hosted runner is destroyed after the job.

## Scientific boundary

The deterministic root-ID-limited pilot remains a plumbing test. Do not interpret its connection density, residual length, cell-type composition, or any other statistic as a biological result. The first E2 result still requires the preregistered spatial pilot in Issue #4.
