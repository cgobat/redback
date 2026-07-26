# Changelog

## v1.18.0 (2026-07-27)

### New features
- Add model metadata registry (`BUILTIN_MODEL_METADATA`) with `output_formats`, `supports_constraints`, `speed`, `max_time_days`, and other fields for 43 core models
- Add `cutoff_blackbody` method to `estimate_bb_params` and `estimate_bolometric_luminosity`
- Add FXT (Fast X-ray Transient) count-spectrum transient class
- Add constrained prior loading via `get_priors(model, constraint=True)` for all models with built-in constraints
- Add `max_time_days` field to `ModelMetadata`

### Bug fixes
- Fix `MagnitudePlotter._filters` returning `[None]` for bands-less transients, causing `KeyError` in `bands_to_frequency`
- Fix `list_of_band_indices` crashing when `bands is None`
- Fix duplicate redback log lines caused by `logger.propagate=True` with a root handler present
- Fix `magnetar_nickel` dense time grid and missing diffusion in spectra path
- Fix `fit_model` to use `get_priors()` instead of hardcoded `priors/` path
- Fix prompt prior builders
- Fix prior bounds and model robustness for 6 previously failing models
- Fix `sn1998bw_template` bandwidth stretching factor and reference distance
- Fix FXT unused directory creation
- Suppress spurious `lalsimulation` import warning

### Improvements
- Remove `_constraint_map`; constraints now always applied via `_apply_constraints`
- Stabilize transient model prior draws
- Rename vegas prior files to match model function names
- Add model prior health CI coverage
- Add comprehensive `RedbackTutorial.ipynb` demo notebook

## v1.17.0
See git log for changes prior to v1.18.0.
