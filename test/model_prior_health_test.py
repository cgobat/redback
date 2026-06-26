import warnings

import numpy as np
import pytest

import redback


MODEL_HEALTH_CASES = {
    "arnett": {
        "time": np.array([2.0, 10.0, 30.0]),
        "kwargs": {
            "output_format": "flux_density",
            "frequency": np.ones(3) * 6.0e14,
        },
        "constraint": True,
        "samples": 3,
    },
    "one_component_kilonova_model": {
        "time": np.array([0.5, 1.0, 2.0]),
        "kwargs": {
            "output_format": "flux_density",
            "frequency": np.ones(3) * 6.0e14,
        },
        "samples": 3,
    },
    "csm_nickel": {
        "time": np.array([1.0, 10.0, 30.0]),
        "kwargs": {
            "output_format": "flux_density",
            "frequency": np.ones(3) * 6.0e14,
            "dense_resolution": 100,
        },
        "constraint": True,
        "samples": 2,
    },
}


def _as_numeric_array(output):
    if hasattr(output, "value"):
        output = output.value
    return np.asarray(output, dtype=float)


def _sample_rows(samples):
    if hasattr(samples, "to_dict"):
        return samples.to_dict("records")
    first_value = next(iter(samples.values()), None)
    if np.ndim(first_value) == 0:
        return [samples]
    return [
        {key: value[ii] for key, value in samples.items()}
        for ii in range(len(first_value))
    ]


@pytest.mark.parametrize("model_name", sorted(MODEL_HEALTH_CASES))
def test_curated_prior_draws_are_finite_without_runtime_warnings(model_name):
    case = MODEL_HEALTH_CASES[model_name]
    metadata = redback.model_library.model_metadata_dict[model_name]
    priors = redback.priors.get_priors(model_name, constraint=case.get("constraint", False))
    function = redback.model_library.all_models_dict[model_name]

    assert metadata.default_output_format == case["kwargs"]["output_format"]
    np.random.seed(12345)
    sample_rows = _sample_rows(priors.sample(case["samples"]))

    for sample in sample_rows:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            output = function(case["time"], **sample, **case["kwargs"])
        values = _as_numeric_array(output)
        assert values.shape == case["time"].shape
        assert np.all(np.isfinite(values))
