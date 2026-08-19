"""Filter-profile handling for Redback.

SVO Filter Profile Service (FPS) identifiers are the canonical filter keys in
Redback.  The bundled ``filters.csv`` table stores one row per SVO FPS filter
and retains SNCosmo names and historical Redback names only as aliases for
backwards compatibility.

Transmission curves are retrieved from SVO FPS by default.  Astroquery's HTTP
cache and the in-process caches in this module avoid repeated network requests.
For legacy filters with no known SVO mapping, or when SVO is unavailable, a
SNCosmo bandpass may be used as an explicit compatibility fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
import warnings
from typing import Mapping

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import ascii


logger = logging.getLogger("redback")

FILTER_TABLE_PATH = Path(__file__).resolve().parent / "tables" / "filters.csv"
LEGACY_SNCOSMO_FILTERS_PATH = (
    Path(__file__).resolve().parent / "tables" / "legacy_sncosmo_filters.txt"
)
_ALIAS_SEPARATOR = "|"
_C_M_S = 299792458.0


@dataclass
class FilterBandpass:
    """A lightweight filter transmission curve independent of SNCosmo.

    Parameters
    ----------
    svofps_id:
        Canonical SVO FPS identifier.  Legacy SNCosmo-only filters use their
        SNCosmo identifier here because no SVO identifier exists for them.
    wave:
        Wavelength samples in Angstrom.
    trans:
        Dimensionless transmission samples.
    source:
        Origin of the transmission curve (normally ``"svo"``).
    """

    svofps_id: str
    wave: np.ndarray
    trans: np.ndarray
    source: str = "svo"

    def __post_init__(self) -> None:
        wave = np.asarray(self.wave, dtype=float)
        trans = np.asarray(self.trans, dtype=float)
        if wave.ndim != 1 or trans.ndim != 1 or wave.shape != trans.shape:
            raise ValueError("Filter wavelength and transmission arrays must be one-dimensional and have equal length")
        finite = np.isfinite(wave) & np.isfinite(trans)
        wave = wave[finite]
        trans = trans[finite]
        if wave.size < 2:
            raise ValueError(f"Filter {self.svofps_id!r} has fewer than two finite transmission samples")
        order = np.argsort(wave)
        wave = wave[order]
        trans = trans[order]
        if np.any(np.diff(wave) <= 0):
            unique_wave, unique_index = np.unique(wave, return_index=True)
            wave = unique_wave
            trans = trans[unique_index]
        wave.setflags(write=False)
        trans.setflags(write=False)
        self.wave = wave
        self.trans = trans

    @property
    def name(self) -> str:
        """Compatibility alias matching the interface used by SNCosmo bandpasses."""
        return self.svofps_id

    def minwave(self) -> float:
        return float(self.wave[0])

    def maxwave(self) -> float:
        return float(self.wave[-1])

    @property
    def wave_eff(self) -> float:
        """Transmission-weighted mean wavelength in Angstrom."""
        norm = np.trapezoid(self.trans, self.wave)
        if norm <= 0:
            raise ValueError(f"Filter {self.svofps_id!r} has non-positive integrated transmission")
        return float(np.trapezoid(self.wave * self.trans, self.wave) / norm)

    @property
    def pivot_wavelength(self) -> float:
        """Pivot wavelength in Angstrom."""
        numerator = np.trapezoid(self.trans * self.wave, self.wave)
        denominator = np.trapezoid(self.trans / self.wave, self.wave)
        if numerator <= 0 or denominator <= 0:
            raise ValueError(f"Cannot compute pivot wavelength for filter {self.svofps_id!r}")
        return float(np.sqrt(numerator / denominator))

    @property
    def effective_width_angstrom(self) -> float:
        """Effective width ``integral(T dlambda) / max(T)`` in Angstrom."""
        maximum = np.nanmax(self.trans)
        if maximum <= 0:
            raise ValueError(f"Filter {self.svofps_id!r} has non-positive transmission")
        return float(np.trapezoid(self.trans, self.wave) / maximum)

    def __call__(self, wavelength) -> np.ndarray:
        return np.interp(wavelength, self.wave, self.trans, left=0.0, right=0.0)


def _split_aliases(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part for part in text.split(_ALIAS_SEPARATOR) if part]


@lru_cache(maxsize=1)
def _filter_table() -> pd.DataFrame:
    table = pd.read_csv(FILTER_TABLE_PATH, keep_default_na=False)
    required = {
        "svofps_id",
        "wavelength [Hz]",
        "wavelength [Angstrom]",
        "reference_flux",
        "sncosmo_name",
        "label",
        "effective_width [Hz]",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"filters.csv is missing required columns: {sorted(missing)}")
    if table["svofps_id"].eq("").any() or table["svofps_id"].isna().any():
        raise ValueError("Every filters.csv row must have an SVO FPS ID")
    duplicated = table.loc[table["svofps_id"].duplicated(), "svofps_id"].tolist()
    if duplicated:
        raise ValueError(f"SVO FPS IDs must be unique in filters.csv: {duplicated}")
    return table


@lru_cache(maxsize=1)
def _identifier_maps():
    table = _filter_table()
    canonical = {}
    sncosmo = {}
    legacy = {}

    for _, row in table.iterrows():
        svofps_id = str(row["svofps_id"])
        canonical[svofps_id] = svofps_id

        for alias in [str(row.get("sncosmo_name", ""))] + _split_aliases(row.get("sncosmo_aliases", "")):
            if not alias:
                continue
            if alias in sncosmo and sncosmo[alias] != svofps_id:
                raise ValueError(f"SNCosmo alias {alias!r} maps to multiple SVO FPS filters")
            sncosmo[alias] = svofps_id

        for alias in _split_aliases(row.get("legacy_aliases", "")):
            if alias in legacy and legacy[alias] != svofps_id:
                raise ValueError(f"Legacy filter alias {alias!r} maps to multiple SVO FPS filters")
            legacy[alias] = svofps_id

    return canonical, sncosmo, legacy


_USER_BANDPASSES: dict[str, FilterBandpass] = {}


@lru_cache(maxsize=1)
def _legacy_sncosmo_only() -> set[str]:
    if not LEGACY_SNCOSMO_FILTERS_PATH.exists():
        return set()
    return {
        line.strip()
        for line in LEGACY_SNCOSMO_FILTERS_PATH.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def clear_filter_cache() -> None:
    """Clear Redback's in-process filter table and transmission caches."""
    _filter_table.cache_clear()
    _identifier_maps.cache_clear()
    _legacy_sncosmo_only.cache_clear()
    _get_svo_transmission_arrays.cache_clear()
    _get_bandpass_cached.cache_clear()
    _get_sncosmo_bandpass_cached.cache_clear()


def _looks_like_svofps_id(identifier: str) -> bool:
    # SVO FPS identifiers have the general form ``facility/instrument.filter``.
    return "/" in identifier and "." in identifier.rsplit("/", 1)[-1]


def resolve_filter_id(identifier, *, warn_alias: bool = False, allow_remote: bool = True) -> str:
    """Resolve a filter identifier to its canonical SVO FPS ID.

    Canonical SVO FPS IDs are preferred.  Known SNCosmo identifiers and old
    Redback band aliases are accepted for backwards compatibility.  A string
    that already looks like an SVO FPS ID is accepted even if it is not in the
    bundled table, allowing any SVO filter to be queried at runtime.  Identifier
    matching is intentionally case-sensitive because historical aliases such as
    ``R`` and ``r`` can refer to different physical bandpasses.
    """
    if not isinstance(identifier, str):
        raise TypeError(f"Filter identifiers must be strings, got {type(identifier).__name__}")

    identifier = identifier.strip()

    # Runtime user filters intentionally remain outside the canonical SVO table.
    if identifier in _USER_BANDPASSES:
        return identifier

    canonical, sncosmo, legacy = _identifier_maps()

    if identifier in canonical:
        return canonical[identifier]
    if identifier in sncosmo:
        if warn_alias:
            warnings.warn(
                f"SNCosmo filter identifier {identifier!r} is deprecated in Redback; "
                f"use SVO FPS ID {sncosmo[identifier]!r} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return sncosmo[identifier]
    if identifier in legacy:
        if warn_alias:
            warnings.warn(
                f"Legacy Redback filter identifier {identifier!r} is deprecated; "
                f"use SVO FPS ID {legacy[identifier]!r} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return legacy[identifier]

    if allow_remote and _looks_like_svofps_id(identifier):
        return identifier

    if identifier in _legacy_sncosmo_only():
        # These filters have no known SVO counterpart.  Keep the identifier so
        # get_bandpass() can use the explicit SNCosmo compatibility fallback.
        return identifier

    raise KeyError(
        f"Filter {identifier!r} is not known. Use an SVO FPS filter ID "
        "(e.g. 'Palomar/ZTF.r') or a retained SNCosmo compatibility name."
    )


def canonicalize_filter_ids(filters, *, warn_alias: bool = False):
    """Return filter identifiers converted to canonical SVO FPS IDs.

    Scalar strings return a scalar string.  Array-like inputs preserve their
    shape and return an object ndarray.  Non-string values are passed through;
    this preserves Redback's frequency-as-band behavior for non-optical data.
    """
    if filters is None:
        return None
    if isinstance(filters, str):
        return resolve_filter_id(filters, warn_alias=warn_alias)

    values = np.asarray(filters, dtype=object)
    result = values.copy()
    for index, value in np.ndenumerate(values):
        if isinstance(value, str):
            result[index] = resolve_filter_id(value, warn_alias=warn_alias)
    return result


def _row_for_svofps_id(svofps_id: str):
    table = _filter_table()
    rows = table.loc[table["svofps_id"] == svofps_id]
    if len(rows) == 0:
        return None
    return rows.iloc[0]


def sncosmo_name_from_filter(identifier, *, required: bool = True) -> str | None:
    """Return the retained canonical SNCosmo name for a filter.

    This is a compatibility helper only; Redback does not use the returned name
    to source its default transmission curve.
    """
    svofps_id = resolve_filter_id(identifier, warn_alias=False)
    if svofps_id in _legacy_sncosmo_only():
        return svofps_id
    row = _row_for_svofps_id(svofps_id)
    if row is None:
        if required:
            raise KeyError(f"No SNCosmo compatibility name is registered for SVO FPS filter {svofps_id!r}")
        return None
    name = str(row.get("sncosmo_name", "")).strip()
    if not name:
        if required:
            raise KeyError(f"No SNCosmo compatibility name is registered for SVO FPS filter {svofps_id!r}")
        return None
    return name


def sncosmo_aliases_from_filter(identifier) -> list[str]:
    """Return all retained SNCosmo names for a canonical filter."""
    svofps_id = resolve_filter_id(identifier, warn_alias=False)
    if svofps_id in _legacy_sncosmo_only():
        return [svofps_id]
    row = _row_for_svofps_id(svofps_id)
    if row is None:
        return []
    names = []
    canonical = str(row.get("sncosmo_name", "")).strip()
    if canonical:
        names.append(canonical)
    names.extend(_split_aliases(row.get("sncosmo_aliases", "")))
    return names


def _svo_query_transmission(svofps_id: str, *, cache: bool = True):
    # Import lazily so importing redback does not itself trigger astroquery
    # initialization and to keep the filter identity utilities lightweight.
    from astroquery.svo_fps import SvoFps

    return SvoFps.get_transmission_data(svofps_id, cache=cache)


@lru_cache(maxsize=256)
def _get_svo_transmission_arrays(svofps_id: str) -> tuple[np.ndarray, np.ndarray]:
    transmission = _svo_query_transmission(svofps_id, cache=True)
    wave = np.asarray(transmission["Wavelength"], dtype=float)
    trans = np.asarray(transmission["Transmission"], dtype=float)
    return wave, trans


def get_transmission_data(identifier, *, cache: bool = True):
    """Retrieve an SVO FPS transmission table for a filter identifier."""
    svofps_id = resolve_filter_id(identifier, warn_alias=False)
    if svofps_id in _legacy_sncosmo_only():
        raise KeyError(
            f"Legacy SNCosmo filter {svofps_id!r} has no SVO FPS mapping and therefore no SVO transmission table"
        )
    return _svo_query_transmission(svofps_id, cache=cache)


def _sncosmo_fallback_bandpass(identifier: str) -> FilterBandpass:
    import sncosmo

    try:
        name = sncosmo_name_from_filter(identifier, required=False) or identifier
    except KeyError:
        name = identifier
    band = sncosmo.get_bandpass(name)
    return FilterBandpass(
        svofps_id=resolve_filter_id(identifier, allow_remote=True),
        wave=np.asarray(band.wave, dtype=float),
        trans=np.asarray(band.trans, dtype=float),
        source="sncosmo",
    )


@lru_cache(maxsize=256)
def _get_bandpass_cached(svofps_id: str, allow_sncosmo_fallback: bool) -> FilterBandpass:
    if svofps_id in _legacy_sncosmo_only():
        if not allow_sncosmo_fallback:
            raise KeyError(f"Filter {svofps_id!r} has no SVO FPS mapping")
        logger.warning(
            "Filter %s has no SVO FPS mapping; using its SNCosmo transmission as a legacy fallback.",
            svofps_id,
        )
        return _sncosmo_fallback_bandpass(svofps_id)

    try:
        wave, trans = _get_svo_transmission_arrays(svofps_id)
        return FilterBandpass(svofps_id=svofps_id, wave=wave, trans=trans, source="svo")
    except Exception as exc:
        if not allow_sncosmo_fallback:
            raise
        name = sncosmo_name_from_filter(svofps_id, required=False)
        if name is None:
            raise
        logger.warning(
            "Could not retrieve %s from SVO FPS (%s). Falling back to SNCosmo bandpass %s.",
            svofps_id,
            exc,
            name,
        )
        return _sncosmo_fallback_bandpass(svofps_id)


def get_bandpass(identifier, *, allow_sncosmo_fallback: bool = True) -> FilterBandpass:
    """Return a cached filter bandpass, sourcing the transmission from SVO FPS by default."""
    if isinstance(identifier, str) and identifier in _USER_BANDPASSES:
        return _USER_BANDPASSES[identifier]
    svofps_id = resolve_filter_id(identifier, warn_alias=False)
    return _get_bandpass_cached(svofps_id, allow_sncosmo_fallback)


@lru_cache(maxsize=256)
def _get_sncosmo_bandpass_cached(svofps_id: str, allow_sncosmo_fallback: bool):
    import sncosmo

    profile = _get_bandpass_cached(svofps_id, allow_sncosmo_fallback)
    return sncosmo.Bandpass(
        profile.wave,
        profile.trans,
        name=profile.svofps_id,
        wave_unit=u.angstrom,
    )


def get_sncosmo_bandpass(identifier, *, allow_sncosmo_fallback: bool = True):
    """Return an SNCosmo ``Bandpass`` object built from Redback's transmission curve.

    This function is the compatibility boundary for Redback code that still
    needs an SNCosmo object (for example the SALT2 interface).  Canonical
    filters are built from SVO FPS transmission data; runtime user filters are
    built from their registered local curves.  No SNCosmo registry lookup is
    used for canonical SVO filters.
    """
    if isinstance(identifier, str) and identifier in _USER_BANDPASSES:
        import sncosmo

        profile = _USER_BANDPASSES[identifier]
        return sncosmo.Bandpass(
            profile.wave,
            profile.trans,
            name=profile.svofps_id,
            wave_unit=u.angstrom,
        )
    svofps_id = resolve_filter_id(identifier, warn_alias=False)
    return _get_sncosmo_bandpass_cached(svofps_id, allow_sncosmo_fallback)


def sncosmo_bandmag(model, *, filters, magsys: str = "ab", time=None, phase=None):
    """Evaluate an SNCosmo model/source through Redback/SVO filter identifiers.

    Exactly one of ``time`` (for ``sncosmo.Model``) or ``phase`` (for an
    SNCosmo ``Source``) must be supplied.
    """
    if (time is None) == (phase is None):
        raise ValueError("Exactly one of time or phase must be provided")

    coordinate_name = "time" if time is not None else "phase"
    coordinate = time if time is not None else phase
    coordinate_array, filter_array = np.broadcast_arrays(coordinate, filters)
    return_scalar = coordinate_array.ndim == 0
    coordinate_array = np.atleast_1d(coordinate_array)
    filter_array = np.atleast_1d(filter_array)
    result = np.empty(coordinate_array.shape, dtype=float)

    for identifier in set(filter_array.ravel()):
        mask = filter_array == identifier
        bandpass = get_sncosmo_bandpass(identifier)
        result[mask] = model.bandmag(
            **{coordinate_name: coordinate_array[mask]},
            band=bandpass,
            magsys=magsys,
        )

    if return_scalar:
        return float(result[0])
    return result


def _effective_width_hz(effective_width_angstrom: float, wavelength_angstrom: float) -> float:
    wavelength_m = wavelength_angstrom * 1.0e-10
    effective_width_m = effective_width_angstrom * 1.0e-10
    return (_C_M_S / wavelength_m**2) * effective_width_m


def _reference_flux(wavelength_angstrom: float, effective_width_angstrom: float) -> float:
    # Preserve Redback's historical integrated-AB-flux convention.
    constant = 3631e-23 * _C_M_S * 1e10
    lo = wavelength_angstrom - effective_width_angstrom / 2.0
    hi = wavelength_angstrom + effective_width_angstrom / 2.0
    if lo <= 0:
        raise ValueError("Effective filter width extends to non-positive wavelength")
    return constant * (1.0 / lo - 1.0 / hi)


def _metadata_from_bandpass(profile: FilterBandpass) -> dict[str, float | str]:
    wavelength = profile.pivot_wavelength
    width_angstrom = profile.effective_width_angstrom
    return {
        "svofps_id": profile.svofps_id,
        "wavelength [Hz]": _C_M_S / (wavelength * 1.0e-10),
        "wavelength [Angstrom]": wavelength,
        "reference_flux": _reference_flux(wavelength, width_angstrom),
        "effective_width [Hz]": _effective_width_hz(width_angstrom, wavelength),
    }


def get_filter_metadata(identifier, *, refresh_from_svo: bool = False) -> dict:
    """Return metadata for a filter.

    Bundled filters use the local table by default.  Unknown valid SVO IDs, or
    calls with ``refresh_from_svo=True``, derive the numerical metadata from the
    SVO transmission curve.
    """
    svofps_id = resolve_filter_id(identifier, warn_alias=False)
    row = None if svofps_id in _legacy_sncosmo_only() else _row_for_svofps_id(svofps_id)
    if row is not None and not refresh_from_svo:
        return row.to_dict()

    profile = get_bandpass(svofps_id)
    metadata = _metadata_from_bandpass(profile)
    if row is not None:
        for key in ("color", "sncosmo_name", "sncosmo_aliases", "legacy_aliases", "label"):
            if key in row:
                metadata[key] = row[key]
    return metadata


def filter_values(filters, column: str) -> np.ndarray:
    """Return one metadata column for a scalar or sequence of filter IDs."""
    if filters is None:
        return np.array([])
    values = [filters] if isinstance(filters, str) else list(filters)
    result = []
    for identifier in values:
        metadata = get_filter_metadata(identifier)
        if column not in metadata:
            raise KeyError(f"Filter metadata does not contain column {column!r}")
        result.append(metadata[column])
    return np.asarray(result)


def add_to_database(
    svofps_id,
    wavelength,
    zeroflux,
    database,
    plot_label,
    effective_width,
    sncosmo_name="",
    sncosmo_aliases="",
    legacy_aliases="",
):
    """Add one canonical SVO FPS filter row to an Astropy table.

    ``wavelength`` is in metres and ``effective_width`` in Angstrom, matching
    the historical Redback helper API.
    """
    frequency = _C_M_S / wavelength
    effective_width_hz = _effective_width_hz(effective_width, wavelength * 1e10)
    database.add_row([
        svofps_id,
        frequency,
        wavelength * 1e10,
        "black",
        zeroflux,
        sncosmo_name,
        sncosmo_aliases,
        legacy_aliases,
        plot_label,
        effective_width_hz,
    ])


def _database_table():
    return ascii.read(FILTER_TABLE_PATH)


def _write_database(database) -> None:
    for column, fmt in {
        "wavelength [Hz]": ".05e",
        "wavelength [Angstrom]": ".05f",
        "reference_flux": ".05e",
        "effective_width [Hz]": ".05e",
    }.items():
        if column in database.colnames:
            database[column].info.format = fmt
    database.write(FILTER_TABLE_PATH, overwrite=True, format="csv")
    clear_filter_cache()


def _row_value(row, key, default=None):
    """Return ``key`` from an Astropy row/mapping, or ``default`` if absent."""
    if row is None:
        return default
    if hasattr(row, "colnames") and key in row.colnames:
        return row[key]
    if isinstance(row, Mapping) and key in row:
        return row[key]
    try:
        return row[key]
    except (KeyError, TypeError, IndexError):
        return default


def add_filter_svo(filter, label=None, plot_label=None, overwrite=False, sncosmo_name=None):
    """Add an SVO FPS filter to Redback's metadata table.

    Parameters
    ----------
    filter:
        Either an SVO FPS ID string or a row returned by
        ``SvoFps.get_filter_list``.  The row's ``filterID`` is always used as
        the canonical key.
    label:
        Deprecated historical Redback identifier.  If supplied, it is stored
        only as a legacy alias; it is never the primary key.
    plot_label:
        Human-readable plotting label.  Defaults to the SVO FPS ID.
    overwrite:
        Replace an existing row for the same SVO FPS ID.
    sncosmo_name:
        Optional SNCosmo compatibility name.
    """
    if isinstance(filter, str):
        svofps_id = filter
        filter_row = None
    else:
        svofps_id = str(filter["filterID"])
        filter_row = filter

    database = _database_table()
    mask = np.where(np.asarray(database["svofps_id"], dtype=str) == svofps_id)[0]
    if len(mask) and not overwrite:
        return

    existing = None
    if len(mask):
        existing = {name: database[name][mask[0]] for name in database.colnames}

    # Always source the actual curve from SVO when creating/replacing metadata.
    transmission = _svo_query_transmission(svofps_id, cache=True)
    profile = FilterBandpass(
        svofps_id=svofps_id,
        wave=np.asarray(transmission["Wavelength"], dtype=float),
        trans=np.asarray(transmission["Transmission"], dtype=float),
    )
    derived = _metadata_from_bandpass(profile)

    wavelength = float(_row_value(filter_row, "WavelengthRef", derived["wavelength [Angstrom]"]))
    width_angstrom = float(_row_value(filter_row, "WidthEff", profile.effective_width_angstrom))

    if len(mask):
        database.remove_rows(mask)

    existing_sncosmo_name = str(existing.get("sncosmo_name", "")) if existing else ""
    existing_sncosmo_aliases = str(existing.get("sncosmo_aliases", "")) if existing else ""
    existing_legacy_aliases = _split_aliases(existing.get("legacy_aliases", "")) if existing else []
    if label and label != svofps_id and label not in existing_legacy_aliases:
        existing_legacy_aliases.append(label)

    if plot_label is None:
        plot_label = str(existing.get("label", svofps_id)) if existing else svofps_id

    zeroflux = _reference_flux(wavelength, width_angstrom)
    add_to_database(
        svofps_id=svofps_id,
        wavelength=wavelength * 1e-10,
        zeroflux=zeroflux,
        database=database,
        plot_label=plot_label,
        effective_width=width_angstrom,
        sncosmo_name=sncosmo_name if sncosmo_name is not None else existing_sncosmo_name,
        sncosmo_aliases=existing_sncosmo_aliases,
        legacy_aliases=_ALIAS_SEPARATOR.join(existing_legacy_aliases),
    )
    _write_database(database)


def add_filter_user(file, label, plot_label=None, overwrite=False):
    """Register a user-supplied transmission curve for the current process.

    User curves do not have an SVO FPS ID and therefore are intentionally not
    written to the canonical SVO filter table.  The returned ``FilterBandpass``
    can be passed directly to low-level routines; callers that need persistent
    identity should publish/use the filter in SVO FPS.
    """
    if label in _USER_BANDPASSES and not overwrite:
        return _USER_BANDPASSES[label]

    transmission = ascii.read(file)
    transmission.rename_columns(list(transmission.keys()), ["Wavelength", "Transmission"])
    profile = FilterBandpass(
        svofps_id=label,
        wave=np.asarray(transmission["Wavelength"], dtype=float),
        trans=np.asarray(transmission["Transmission"], dtype=float),
        source="user",
    )
    # Store in the cached lookup under the user label without polluting filters.csv.
    _USER_BANDPASSES[label] = profile
    _get_bandpass_cached.cache_clear()
    _get_sncosmo_bandpass_cached.cache_clear()
    return profile


def add_to_sncosmo(label, transmission):
    """Deprecated compatibility helper that registers a curve in SNCosmo."""
    warnings.warn(
        "add_to_sncosmo is deprecated; Redback now sources filter curves from SVO FPS.",
        DeprecationWarning,
        stacklevel=2,
    )
    import sncosmo

    band = sncosmo.Bandpass(
        transmission["Wavelength"],
        transmission["Transmission"],
        name=label,
        wave_unit=u.angstrom,
    )
    sncosmo.register(band, label, force=True)


def add_common_filters(overwrite=False):
    """Ensure historically optional SVO filter sets are present in filters.csv."""
    from astroquery.svo_fps import SvoFps

    queries = [
        ("La Silla", "GROND", None),
        ("La Silla", "EFOSC", "Gunn"),
        ("Euclid", "VIS", None),
        ("Euclid", "NISP", None),
        ("Spitzer", "IRAC", None),
        ("WISE", None, None),
    ]
    for facility, instrument, description_contains in queries:
        print(f"{facility}/{instrument or 'all'} filters...")
        table = SvoFps.get_filter_list(facility=facility, instrument=instrument)
        if description_contains is not None and "Description" in table.colnames:
            table = table[[description_contains in str(x) for x in table["Description"]]]
        for row in table:
            add_filter_svo(row, overwrite=overwrite)
        print("done.\n")


def show_all_filters():
    """Return the canonical SVO FPS filter metadata table."""
    return ascii.read(FILTER_TABLE_PATH)


def add_effective_widths():
    """Refresh effective widths in ``filters.csv`` from SVO transmission curves."""
    database = _database_table()
    for ii, svofps_id in enumerate(database["svofps_id"]):
        try:
            profile = get_bandpass(str(svofps_id), allow_sncosmo_fallback=False)
            database["effective_width [Hz]"][ii] = _effective_width_hz(
                profile.effective_width_angstrom,
                profile.pivot_wavelength,
            )
        except Exception as exc:
            logger.warning("Failed to refresh SVO filter %s at index %s: %s", svofps_id, ii, exc)
    _write_database(database)


def refresh_filter_metadata_from_svo():
    """Refresh numerical metadata for every bundled filter from SVO FPS curves."""
    database = _database_table()
    for ii, svofps_id in enumerate(database["svofps_id"]):
        try:
            profile = get_bandpass(str(svofps_id), allow_sncosmo_fallback=False)
            metadata = _metadata_from_bandpass(profile)
            for column in (
                "wavelength [Hz]",
                "wavelength [Angstrom]",
                "reference_flux",
                "effective_width [Hz]",
            ):
                database[column][ii] = metadata[column]
        except Exception as exc:
            logger.warning("Failed to refresh SVO filter %s at index %s: %s", svofps_id, ii, exc)
    _write_database(database)
