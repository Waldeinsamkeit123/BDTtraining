# BDTtraining

Independent XGBoost baseline for the boosted ParT V4 factorized classifiers. The
project is self-contained at runtime: it reads the same Pepper NewWP ROOT files,
but it does not import code from NanoTTH or `tthcc_an`.

The implementation was informed by the earlier event-BDT workflow in
<https://github.com/Waldeinsamkeit123/tthcc_an>, while changing its fold handling
so the ParT held-out fold is never used for XGBoost early stopping.

## Environment

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh
cd /scratchfs2/cms/wanghan/BDTtraining
```

Validated versions:

| Package | Version |
| --- | --- |
| Python | 3.13.11 |
| NumPy | 2.2.6 |
| Awkward | 2.8.9 |
| uproot | 5.6.0 |
| scikit-learn | 1.8.0 |
| Matplotlib | 3.10.8 |
| XGBoost | 2.1.3 |

No package installation is needed.

## Source Of Truth

The configs mirror these ParT V4 files:

- `ttH_1L_ParT_2024_boosted_v4.yaml`
- `ttH_0L_ParT_2024_boosted_v4.yaml`
- `train_1L_boosted_v4.sh`
- `train_0L_boosted_v4.sh`
- `prepare_merged_inputs.py`

The 1L selection is:

```text
nCleanedJet >= 2
sum(CleanedJet_tag >= 50) >= 1
sum(CleanedJet_tag >= 40) >= 1
nTargetFatJet >= 1
```

The classes are `ttHZ = ttHcc|ttHbb|ttZqq|ttZcc|ttZbb` and
`ttjets = ttLF|ttcj|ttcc|tt2c|ttbj|ttbb|tt2b`. The positive binary target is
`ttHZ`, so the saved score is `P(ttHZ)`.

The 0L selection and exact `ttHZ`, `ttjets`, and `qcd` definitions are encoded in
`config/bdt_0L_v4_sameinfo.json`. Real-data validation is deferred until
`0L_NewWP_merged` exists; old 0L data must not be substituted. The generic cache,
multiclass training, prediction, feature-importance, and CPU Slurm paths are
already wired, including `slurm/train_0L_v4_sameinfo.sh`, but that wrapper must
not be submitted before preparing and validating the real NewWP cache.

## Feature Boundary

The complete 1L matrix has 140 columns. The complete 0L framework has 129
columns. Collection ordering is preserved exactly as stored in ROOT; no BDT-only
sorting is applied.

| Features | Source | Direct/computed | Formula and missing handling | Count 1L | Count 0L |
| --- | --- | --- | --- | ---: | ---: |
| `CleanedJet_{pt,eta,phi,mass}_{1..8}` | corresponding ROOT collection | direct elements | first 8; absent slots are NaN | 32 | 32 |
| `CleanedJet_tag_{B4..B0,C4..C0}_{1..8}` | `CleanedJet_tag` | computed categorical | equality to 54..50 and 44..40; absent slots are NaN | 80 | 80 |
| `nCleanedJet` | ROOT | direct | scalar | 1 | 1 |
| `CleanedHT` | `CleanedJet_pt` | computed | sum over the same full CleanedJet collection | 1 | 1 |
| `TargetFatJet_{pt,eta,phi}_{1..3}` | corresponding ROOT collection | direct elements | first 3; absent slots are NaN | 9 | 9 |
| `Lepton_{pt,eta,phi,mass}` | corresponding ROOT collection | direct first element | one lepton in selected 1L data | 4 | 0 |
| `Lepton_isMuon`, `Lepton_isElectron` | `Lepton_pdgId` | computed | `pdgId==13`, `pdgId==11`, exactly as current ParT YAML | 2 | 0 |
| `MET_pt`, `MET_phi` | ROOT | direct | same information as the ParT MET pseudo-token | 2 | 0 |
| six Pepper topology variables | same-name ROOT branches | direct | values >=900 become NaN | 6 | 6 |
| three relative lepton/AK8/MET variables | eta/phi ROOT branches | computed | wrapped delta-phi and delta-R | 3 | 0 |

The six direct Pepper topology inputs are `minDR_b`, `DR_Tarb1`, `DR_Tarb2`,
`minDR_TarClean`, `minDEta_TarClean`, and `minDPhi_TarClean`. Their producer code
was not present in either inspected repository, so this project does not
recompute or reinterpret them. The first three contain the Pepper missing-value
sentinel 999 when the required b-jet pair is unavailable; it is converted to
XGBoost native missing (`NaN`).

The computed 1L variables are:

```text
dPhi_lepton_MET = abs(wrap(Lepton_phi[0] - MET_phi))
dPhi_TargetFatJet_MET = abs(wrap(TargetFatJet_phi[0] - MET_phi))
dR_TargetFatJet_lepton = sqrt(dEta^2 + wrap(dPhi)^2)
```

Absolute phi is enabled by default with `include_absolute_phi: true`. No
additional AK4 tag-count summary is included. `nCleanedJet` and `CleanedHT` are
deterministic summaries of information already present in the padded collection.

### Leakage Guard

TargetFatJet mass, random mass, soft-drop/regressed masses, tag categories,
GlobalParT scores, HiggsVsQCD, BBvsCC, and mass-window flags are never loaded as
features. Feature construction fails if an AK8 mass/tagger-like name enters the
matrix.

- AK8 mass leakage: **NO**
- AK8 tagger leakage: **NO**
- calibrated AK4 tagging used: **YES**

## Prepared Cache

The ROOT scan is performed once:

```bash
python scripts/prepare_inputs.py \
  --config config/bdt_1L_v4_sameinfo.json \
  --force
```

The cache stores `X`, `run`, `luminosityBlock`, `event`, `sample_id`, `label`,
`fold_id`, and `physics_weight=abs(weight)`. Its metadata records the selected
counts, exact feature order/specification, source paths/sizes/timestamps, schema,
and a hash of both the config and cache-building source. Training refuses stale
caches.

Generated audit files are under `outputs/1L_v4_sameinfo/prepared/`:

- `metadata.json`
- `feature_table.json` and `.csv`
- `feature_audit.json` and `.csv`
- `topology_audit.json`
- `consistency.json`
- `split_reweight_audit.json`

The event comparison can be reproduced with:

```bash
python scripts/inspect_inputs.py \
  --config config/bdt_1L_v4_sameinfo.json \
  --output outputs/1L_v4_sameinfo/prepared/consistency.json

python scripts/check_splits.py \
  --config config/bdt_1L_v4_sameinfo.json \
  --output outputs/1L_v4_sameinfo/prepared/split_reweight_audit.json
```

`(run,luminosityBlock,event)` has 1,437 cross-sample collisions in the official
1L inputs. The collision-free identity saved for future event matching is
`(sample_id,run,luminosityBlock,event)`. This does not affect the fold definition,
which remains exactly `event % 5`.

## Folds And Reweighting

For model `i`:

```text
held-out OOF: event % 5 == i
training pool: event % 5 != i
early-stop validation inside pool:
  stable_hash(run,luminosityBlock,event,sample_id) % 10 == 0
training inside pool: all remaining events
```

The three masks are deterministic, disjoint, and exhaustive. The held-out fold
is not passed to XGBoost until final inference.

The ParT V4 `flat` histograms each contain one broad bin. The BDT therefore uses
deterministic importance weights: start from `abs(weight)`, scale each class to
the configured class target, then normalize the total weight to the number of
training events. This matches the one-bin class target without pretending to
reproduce Weaver's stochastic rejection/up-sampling implementation. Physics
weights and training weights are always stored/reported separately.

## Training

The baseline uses `tree_method=hist`, `eta=0.05`, depth 4, subsample 0.8,
column sample 0.8, minimum child weight 1, 600 rounds, and 30-round early
stopping. Full five-fold training creates:

```text
outputs/1L_v4_sameinfo/models/fold0.json ... fold4.json
outputs/1L_v4_sameinfo/predictions/oof_predictions.npz
outputs/1L_v4_sameinfo/plots/*.png
outputs/1L_v4_sameinfo/feature_importance.json
outputs/1L_v4_sameinfo/summary.json
```

For a small login-node smoke test only:

```bash
python scripts/train_bdt.py \
  --config config/bdt_1L_v4_sameinfo.json \
  --max-events 10000 \
  --num-boost-round 20 \
  --early-stopping-rounds 5 \
  --output-dir outputs/smoke_1L
```

For production, submit one CPU-only job. It trains the five folds sequentially so
a single process writes the complete OOF payload without cross-job aggregation:

```bash
sbatch slurm/train_1L_v4_sameinfo.sh
```

The wrapper uses the accessible `gpu/cmsgpu/cmsnormal` association but deliberately
does **not** request a GPU. It requests 4 CPU cores, 24 GB RAM, and 48 hours and
selects `device=cpu`. LCG109's XGBoost build reports `USE_CUDA=False`; asking for a
V100 would consume a GPU without making this training faster. A `spub` test was
rejected for this account/QoS, while the CPU-only `gpu` request passed Slurm's
`--test-only` check. No job is submitted by repository setup or validation.

XGBoost gain importance is saved both as raw mean gain and normalized gain. It is
a tree split-improvement statistic, not a physics sensitivity percentage.
