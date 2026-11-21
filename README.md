# Brake & Burst · NFL Big Data Bowl 2027

**Development entry, not a submitted competition entry. All committed results and notebook outputs use synthetic data. Real NFL competition data requires Kaggle login and rule acceptance; it was NOT accessed or used. No account was created, login attempted, notebook published, or remote push made.**

Question: does speed retained through a route break translate from Combine drills to NFL separation, beyond traditional tests? Primary modelling: WR/TE. RB and DB groups: descriptive cut translation. Read [REPORT.md](REPORT.md) for the current measured results and limitations.

## Run

System Python 3.10 is supported. From this directory:

```sh
python3 -m pip install --user -r requirements.txt
python run_pipeline.py --synthetic
pytest -q
python build_notebook.py
jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis.ipynb --ExecutePreprocessor.timeout=600
```

The synthetic fixture is deterministic (seed 2027), deliberately modest (150 players) and contains **all documented columns in all nine CSV files**, with field-bounded 10 Hz trajectories and a noisy, planted latent agility signal independent of some straight-line speed variation. It is not a realistic population simulator. No generated CSV data is committed. `tests/documented_schema.json` is a column/type snapshot mechanically extracted from the supplied competition data-description text, including its `visitor_team_win_probility_added` spelling. Tests verify columns and numeric/bool types, sampling and trajectory bounds. Nullable integer CSV columns naturally reload as floats when nulls occur.

For real data, **the user must**:

1. Create/join a Kaggle account, check eligibility, and accept competition rules personally. Never share credentials or licensed data.
2. With your own authenticated Kaggle CLI, download and extract locally:
   ```sh
   kaggle competitions download -c nfl-big-data-bowl-2027
   unzip nfl-big-data-bowl-2027.zip -d data/real
   python run_pipeline.py --data-dir data/real
   ```
   Point `--data-dir` at the directory containing the CSVs (not an enclosing archive directory). With no flags, `/kaggle/input/nfl-big-data-bowl-2027` is auto-detected, otherwise synthetic mode is clearly announced. Dataset-specific quality or insufficient-sample errors must be resolved, not bypassed.
3. Review cut traces, event coverage, missingness, role/route mix, model metrics and regenerated REPORT.md. Do not submit synthetic findings as football evidence. Outputs contain derived player information and should be reviewed for data-license compliance before publishing real-data artifacts.
4. Build the notebook, attach the competition dataset in Kaggle, run it there, and **publish publicly**. It inlines the implementation and report template and does not import `bdb27` or require this repository.
5. Replace the pending public-notebook URL in the report template and regenerate. Submit the Writeup (not a prediction CSV) before **January 6, 2027 23:59 UTC = January 7, 2027 07:59 China time**. Recheck the official deadline/rules. Finalists are invited to present at the **2027 Scouting Combine**; accepting that invitation is the user's decision.

## Design / outputs

`src/bdb27/` holds data loading/schema, synthetic generation, Combine features, game features, modelling, plotting and pipeline orchestration. `run_pipeline.py` is the CLI; `build_notebook.py` deterministically inlines modules into `analysis.ipynb`. `REPORT_TEMPLATE.md` is populated automatically with measured outputs. `figures/` contains five labelled PNG charts. `outputs/` contains model fold metrics, per-player features, cut records, held-out predictions, permutation importance, bootstrap summary, CV membership audit and hidden-talent / risk CSV+Markdown queues.

Game tracking uses `chunksize` and `usecols`, filtering to needed player-plays and sharding by player before computing features; entire plays survive chunk boundaries and unsorted input. At most one player partition is loaded at once. Combine tracking and player-play metadata are loaded in memory. Filtered temporary shards are removed after use.

Cut detector: local minimum prominence 0.25 yd/s, at least 1 s between detected minima, incoming speed >=2 yd/s, circular turn >=45° over 0.6 s. Entry speed is the median 0.5–0.3 s before the minimum; exit is 1 s after; recovery is time to 80% entry. Requiring a full exit window creates censoring. Sprint timing is relative to the first recorded frame, not an inferred official start. All measurements assume documented 10 Hz sampling. Thresholds are operational defaults, not tuned scientific constants.

Primary target is context-adjusted targeted separation, with training-only position prior and ten-target shrinkage; >=3 targets qualify. All preprocessors and target-context models are fit inside validation folds. Ridge/logistic and gradient boosting compare A/B and draft-adjusted variants across leave-draft-class-out and repeated stratified folds. No test-set tuning. Bootstrap CIs are paired player resamples of fixed held-out predictions, not retraining intervals. The classifier predicts career starter status; cohort-dependent follow-up remains a limitation. Snap share is an opportunity **proxy** because team snap denominators are absent. Career snap totals mix regular season and playoffs and are never used as true regular-season shares.

The report has fewer than ten combined figures/tables and is below 2,000 words. This is a research scaffold validated on synthetic data; real schema/event quirks, sensitivity analysis and empirical football claims remain unvalidated. Code is MIT licensed; competition data retains **CC BY-NC 4.0** restrictions and is not redistributed.
