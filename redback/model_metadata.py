from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelMetadata:
    """Structured metadata for a redback model function."""

    name: str
    model_type: str
    source_module: str | None = None
    output_formats: tuple[str, ...] = ()
    default_output_format: str | None = None
    required_kwargs: tuple[str, ...] = ()
    optional_dependencies: tuple[str, ...] = ()
    has_prior: bool = False
    is_public: bool = True
    supports_extinction: bool = False
    supports_constraints: bool = False
    speed: str = "unknown"


def _validate_string_tuple(field_name: str, values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple of strings.")
    if not all(isinstance(value, str) and value for value in values):
        raise TypeError(f"{field_name} must contain only non-empty strings.")


def validate_model_metadata(
        metadata: ModelMetadata, available_model_names: set[str] | None = None) -> ModelMetadata:
    """Validate and return a model metadata record."""
    if not isinstance(metadata.name, str) or not metadata.name:
        raise ValueError("Model metadata name must be a non-empty string.")
    if available_model_names is not None and metadata.name not in available_model_names:
        raise ValueError(f"Model metadata for '{metadata.name}' has no matching model.")
    if not isinstance(metadata.model_type, str) or not metadata.model_type:
        raise ValueError("Model metadata model_type must be a non-empty string.")
    if metadata.source_module is not None:
        if not isinstance(metadata.source_module, str) or not metadata.source_module:
            raise TypeError("source_module must be a non-empty string or None.")

    _validate_string_tuple("output_formats", metadata.output_formats)
    _validate_string_tuple("required_kwargs", metadata.required_kwargs)
    _validate_string_tuple("optional_dependencies", metadata.optional_dependencies)

    if metadata.default_output_format is not None:
        if not isinstance(metadata.default_output_format, str) or not metadata.default_output_format:
            raise TypeError("default_output_format must be a non-empty string or None.")
        if metadata.output_formats and metadata.default_output_format not in metadata.output_formats:
            raise ValueError("default_output_format must be listed in output_formats.")

    if not isinstance(metadata.has_prior, bool):
        raise TypeError("has_prior must be a boolean.")
    if not isinstance(metadata.is_public, bool):
        raise TypeError("is_public must be a boolean.")
    if not isinstance(metadata.supports_extinction, bool):
        raise TypeError("supports_extinction must be a boolean.")
    if not isinstance(metadata.supports_constraints, bool):
        raise TypeError("supports_constraints must be a boolean.")
    if not isinstance(metadata.speed, str) or not metadata.speed:
        raise ValueError("speed must be a non-empty string.")
    return metadata


def coerce_model_metadata(name: str, metadata: ModelMetadata | Mapping[str, Any]) -> ModelMetadata:
    """Return a ModelMetadata instance from plugin or built-in metadata."""
    if isinstance(metadata, ModelMetadata):
        if metadata.name != name:
            metadata = replace(metadata, name=name)
        return validate_model_metadata(metadata)

    if not isinstance(metadata, Mapping):
        raise TypeError("Model metadata must be a ModelMetadata instance or mapping.")

    values = dict(metadata)
    values.setdefault("name", name)
    for key in ("output_formats", "required_kwargs", "optional_dependencies"):
        if key in values:
            if values[key] is None:
                values.pop(key)
            elif isinstance(values[key], str):
                values[key] = (values[key],)
            else:
                values[key] = tuple(values[key])
    return validate_model_metadata(ModelMetadata(**values))


OPTICAL_OUTPUT_FORMATS = ("flux_density", "magnitude", "flux", "spectra", "sncosmo_source")
SIMPLE_OPTICAL_OUTPUT_FORMATS = ("flux_density", "magnitude", "flux")
BOL_OUTPUT_FORMATS = ("luminosity",)
COUNTS_OUTPUT_FORMATS = ("counts",)
SPECTRUM_OUTPUT_FORMATS = ("spectrum",)


def _metadata(
        name: str, model_type: str, source_module: str, output_formats: tuple[str, ...],
        default_output_format: str | None = None, required_kwargs: tuple[str, ...] = (),
        supports_extinction: bool = False, supports_constraints: bool = False,
        speed: str = "unknown", has_prior: bool = True,
        optional_dependencies: tuple[str, ...] = ()) -> ModelMetadata:
    if default_output_format is None and len(output_formats) == 1:
        default_output_format = output_formats[0]
    return validate_model_metadata(ModelMetadata(
        name=name,
        model_type=model_type,
        source_module=source_module,
        output_formats=output_formats,
        default_output_format=default_output_format,
        required_kwargs=required_kwargs,
        optional_dependencies=optional_dependencies,
        has_prior=has_prior,
        is_public=not name.startswith("_"),
        supports_extinction=supports_extinction,
        supports_constraints=supports_constraints,
        speed=speed,
    ))


def _optical_model(
        name: str, model_type: str, source_module: str, supports_constraints: bool = False,
        speed: str = "medium", output_formats: tuple[str, ...] = OPTICAL_OUTPUT_FORMATS,
        has_prior: bool = True, optional_dependencies: tuple[str, ...] = ()) -> ModelMetadata:
    return _metadata(
        name=name,
        model_type=model_type,
        source_module=source_module,
        output_formats=output_formats,
        default_output_format="flux_density",
        required_kwargs=("output_format", "redshift"),
        supports_extinction=True,
        supports_constraints=supports_constraints,
        speed=speed,
        has_prior=has_prior,
        optional_dependencies=optional_dependencies,
    )


def _bolometric_model(
        name: str, model_type: str, source_module: str, supports_constraints: bool = False,
        speed: str = "fast", has_prior: bool = True,
        optional_dependencies: tuple[str, ...] = ()) -> ModelMetadata:
    return _metadata(
        name=name,
        model_type=model_type,
        source_module=source_module,
        output_formats=BOL_OUTPUT_FORMATS,
        supports_constraints=supports_constraints,
        speed=speed,
        has_prior=has_prior,
        optional_dependencies=optional_dependencies,
    )


BUILTIN_MODEL_METADATA = {
    "gaussian_prompt": _metadata(
        name="gaussian_prompt", model_type="prompt", source_module="prompt_models",
        output_formats=COUNTS_OUTPUT_FORMATS, speed="fast"),
    "skew_gaussian": _metadata(
        name="skew_gaussian", model_type="prompt", source_module="prompt_models",
        output_formats=COUNTS_OUTPUT_FORMATS, speed="fast"),
    "skew_exponential": _metadata(
        name="skew_exponential", model_type="prompt", source_module="prompt_models",
        output_formats=COUNTS_OUTPUT_FORMATS, speed="fast"),
    "fred": _metadata(
        name="fred", model_type="prompt", source_module="prompt_models",
        output_formats=COUNTS_OUTPUT_FORMATS, speed="fast"),
    "fred_extended": _metadata(
        name="fred_extended", model_type="prompt", source_module="prompt_models",
        output_formats=COUNTS_OUTPUT_FORMATS, speed="fast"),

    "arnett": _optical_model(
        name="arnett", model_type="supernova", source_module="supernova_models",
        supports_constraints=True, speed="medium"),
    "arnett_bolometric": _bolometric_model(
        name="arnett_bolometric", model_type="supernova", source_module="supernova_models",
        supports_constraints=True),
    "arnett_with_features": _optical_model(
        name="arnett_with_features", model_type="supernova", source_module="supernova_models",
        supports_constraints=True, speed="medium"),
    "csm_interaction": _optical_model(
        name="csm_interaction", model_type="supernova", source_module="supernova_models",
        supports_constraints=True, speed="medium"),
    "csm_nickel": _optical_model(
        name="csm_nickel", model_type="supernova", source_module="supernova_models",
        supports_constraints=True, speed="medium"),
    "csm_nickel_bolometric": _bolometric_model(
        name="csm_nickel_bolometric", model_type="supernova", source_module="supernova_models",
        supports_constraints=True),
    "csm_shock_and_arnett": _optical_model(
        name="csm_shock_and_arnett", model_type="supernova", source_module="supernova_models",
        supports_constraints=True, speed="medium"),
    "csm_shock_and_arnett_bolometric": _bolometric_model(
        name="csm_shock_and_arnett_bolometric", model_type="supernova", source_module="supernova_models",
        supports_constraints=True),
    "magnetar_nickel": _optical_model(
        name="magnetar_nickel", model_type="supernova", source_module="supernova_models",
        supports_constraints=True, speed="medium"),
    "slsn": _optical_model(
        name="slsn", model_type="supernova", source_module="supernova_models",
        supports_constraints=True, speed="medium"),
    "slsn_bolometric": _bolometric_model(
        name="slsn_bolometric", model_type="supernova", source_module="supernova_models",
        supports_constraints=True),
    "type_1a": _optical_model(
        name="type_1a", model_type="supernova", source_module="supernova_models",
        speed="medium", optional_dependencies=("sncosmo",)),
    "type_1c": _optical_model(
        name="type_1c", model_type="supernova", source_module="supernova_models",
        speed="medium", optional_dependencies=("sncosmo",)),
    "salt2": _optical_model(
        name="salt2", model_type="supernova", source_module="supernova_models",
        speed="medium", optional_dependencies=("sncosmo",)),

    "one_component_kilonova_model": _optical_model(
        name="one_component_kilonova_model", model_type="kilonova",
        source_module="kilonova_models", speed="fast"),
    "two_component_kilonova_model": _optical_model(
        name="two_component_kilonova_model", model_type="kilonova",
        source_module="kilonova_models", speed="fast"),
    "three_component_kilonova_model": _optical_model(
        name="three_component_kilonova_model", model_type="kilonova",
        source_module="kilonova_models", speed="fast"),
    "one_component_ejecta_relation": _optical_model(
        name="one_component_ejecta_relation", model_type="kilonova",
        source_module="kilonova_models", supports_constraints=True, speed="fast"),
    "two_component_bns_ejecta_relation": _optical_model(
        name="two_component_bns_ejecta_relation", model_type="kilonova",
        source_module="kilonova_models", supports_constraints=True, speed="fast"),
    "two_component_nsbh_ejecta_relation": _optical_model(
        name="two_component_nsbh_ejecta_relation", model_type="kilonova",
        source_module="kilonova_models", supports_constraints=True, speed="fast"),

    "gaussianrise_cooling_envelope": _optical_model(
        name="gaussianrise_cooling_envelope", model_type="tde", source_module="tde_models",
        output_formats=SIMPLE_OPTICAL_OUTPUT_FORMATS, supports_constraints=True, speed="medium"),
    "bpl_cooling_envelope": _optical_model(
        name="bpl_cooling_envelope", model_type="tde", source_module="tde_models",
        output_formats=SIMPLE_OPTICAL_OUTPUT_FORMATS, supports_constraints=True, speed="medium"),
    "cooling_envelope": _optical_model(
        name="cooling_envelope", model_type="tde", source_module="tde_models",
        supports_constraints=True, speed="medium"),
    "tde_analytical": _optical_model(
        name="tde_analytical", model_type="tde", source_module="tde_models", speed="fast"),
    "tde_analytical_bolometric": _bolometric_model(
        name="tde_analytical_bolometric", model_type="tde", source_module="tde_models"),
    "tde_fallback": _optical_model(
        name="tde_fallback", model_type="tde", source_module="tde_models", speed="medium"),
    "tde_fallback_bolometric": _bolometric_model(
        name="tde_fallback_bolometric", model_type="tde", source_module="tde_models"),

    "tophat": _metadata(
        name="tophat", model_type="afterglow", source_module="afterglow_models",
        output_formats=("flux_density", "magnitude"), default_output_format="flux_density",
        required_kwargs=("output_format", "redshift"), speed="medium"),
    "gaussian": _metadata(
        name="gaussian", model_type="afterglow", source_module="afterglow_models",
        output_formats=("flux_density", "magnitude"), default_output_format="flux_density",
        required_kwargs=("output_format", "redshift"), speed="medium"),
    "tophat_redback": _metadata(
        name="tophat_redback", model_type="afterglow", source_module="afterglow_models",
        output_formats=("flux_density", "magnitude"), default_output_format="flux_density",
        required_kwargs=("output_format", "redshift"), speed="medium"),
    "gaussian_redback": _metadata(
        name="gaussian_redback", model_type="afterglow", source_module="afterglow_models",
        output_formats=("flux_density", "magnitude"), default_output_format="flux_density",
        required_kwargs=("output_format", "redshift"), speed="medium"),
    "afterglow_models_sed": _metadata(
        name="afterglow_models_sed", model_type="afterglow", source_module="afterglow_models",
        output_formats=("magnitude", "spectra", "flux", "sncosmo_source"),
        default_output_format="magnitude", required_kwargs=("output_format", "redshift"),
        speed="medium"),

    "thermal_synchrotron_fluxdensity": _metadata(
        name="thermal_synchrotron_fluxdensity", model_type="general_synchrotron",
        source_module="general_synchrotron_models", output_formats=("flux_density",),
        speed="medium"),
    "thermal_synchrotron_lnu": _bolometric_model(
        name="thermal_synchrotron_lnu", model_type="general_synchrotron",
        source_module="general_synchrotron_models", speed="medium"),
    "band_function_high_energy": _metadata(
        name="band_function_high_energy", model_type="spectral", source_module="spectral_models",
        output_formats=SPECTRUM_OUTPUT_FORMATS, speed="fast"),
    "blackbody_high_energy": _metadata(
        name="blackbody_high_energy", model_type="spectral", source_module="spectral_models",
        output_formats=SPECTRUM_OUTPUT_FORMATS, speed="fast"),
    "cutoff_powerlaw_high_energy": _metadata(
        name="cutoff_powerlaw_high_energy", model_type="spectral", source_module="spectral_models",
        output_formats=SPECTRUM_OUTPUT_FORMATS, speed="fast"),
}
