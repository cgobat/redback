from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import redback.get_data.directory
from redback.spectral.dataset import SpectralDataset
from redback.transient.spectral import CountsSpectrumTransient


@dataclass
class FXT(CountsSpectrumTransient):
    """Fast X-ray transient count-spectrum wrapper."""

    dataset: SpectralDataset
    name: str = "fxt"
    instrument: Optional[str] = None
    energy_range: Optional[tuple[float, float]] = None

    def __post_init__(self):
        self._validate_dataset(self.dataset)
        super().__post_init__()
        self.directory_structure = redback.get_data.directory.open_access_directory_structure(
            transient=self.name, transient_type="fxt")
        if self.energy_range is not None:
            self.dataset.set_active_interval(*self.energy_range)

    @staticmethod
    def _validate_dataset(dataset: SpectralDataset):
        counts = np.asarray(dataset.counts, dtype=float)
        energy_edges_keV = np.asarray(dataset.energy_edges_keV, dtype=float)

        if counts.ndim != 1:
            raise ValueError("FXT dataset counts must be a one-dimensional array")
        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("FXT dataset counts must be finite and non-negative")
        if energy_edges_keV.ndim != 1 or len(energy_edges_keV) < 2:
            raise ValueError("FXT dataset energy_edges_keV must be a one-dimensional array with at least two edges")
        if not np.all(np.isfinite(energy_edges_keV)) or np.any(np.diff(energy_edges_keV) <= 0):
            raise ValueError("FXT dataset energy_edges_keV must be finite and strictly increasing")
        if not np.isfinite(float(dataset.exposure)) or float(dataset.exposure) <= 0:
            raise ValueError("FXT dataset exposure must be finite and positive")

        has_energy_bin_axis = len(counts) == len(energy_edges_keV) - 1
        has_response_channel_axis = dataset.rmf is not None and len(counts) == len(dataset.rmf.channel)
        if not has_energy_bin_axis and not has_response_channel_axis:
            raise ValueError("FXT dataset counts must match energy bins or RMF detector channels")

        if dataset.counts_bkg is not None:
            counts_bkg = np.asarray(dataset.counts_bkg, dtype=float)
            if counts_bkg.shape != counts.shape:
                raise ValueError("FXT dataset counts_bkg must have the same shape as counts")
            if not np.all(np.isfinite(counts_bkg)) or np.any(counts_bkg < 0):
                raise ValueError("FXT dataset counts_bkg must be finite and non-negative")
            if dataset.bkg_exposure is not None and (
                    not np.isfinite(float(dataset.bkg_exposure)) or float(dataset.bkg_exposure) <= 0):
                raise ValueError("FXT dataset bkg_exposure must be finite and positive")

        if dataset.quality is not None and len(dataset.quality) != len(counts):
            raise ValueError("FXT dataset quality must have the same length as counts")
        if dataset.grouping is not None and len(dataset.grouping) != len(counts):
            raise ValueError("FXT dataset grouping must have the same length as counts")

    @staticmethod
    def _validate_binned_spectrum(counts, exposure, energy_edges_keV, counts_bkg=None):
        counts = np.asarray(counts, dtype=float)
        energy_edges_keV = np.asarray(energy_edges_keV, dtype=float)
        exposure = float(exposure)

        if counts.ndim != 1:
            raise ValueError("counts must be a one-dimensional array")
        if energy_edges_keV.ndim != 1 or len(energy_edges_keV) < 2:
            raise ValueError("energy_edges_keV must be a one-dimensional array with at least two edges")
        if len(counts) != len(energy_edges_keV) - 1:
            raise ValueError("counts must have one value per energy bin")
        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("counts must be finite and non-negative")
        if not np.all(np.isfinite(energy_edges_keV)) or np.any(np.diff(energy_edges_keV) <= 0):
            raise ValueError("energy_edges_keV must be finite and strictly increasing")
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("exposure must be finite and positive")

        if counts_bkg is None:
            return counts, exposure, energy_edges_keV, None

        counts_bkg = np.asarray(counts_bkg, dtype=float)
        if counts_bkg.shape != counts.shape:
            raise ValueError("counts_bkg must have the same shape as counts")
        if not np.all(np.isfinite(counts_bkg)) or np.any(counts_bkg < 0):
            raise ValueError("counts_bkg must be finite and non-negative")
        return counts, exposure, energy_edges_keV, counts_bkg

    @classmethod
    def from_counts(
            cls, name: str, counts: np.ndarray, exposure: float, energy_edges_keV: np.ndarray,
            counts_bkg: np.ndarray = None, bkg_exposure: float = None, instrument: str = None,
            energy_range: tuple[float, float] = None, **kwargs) -> "FXT":
        """Build an FXT from counts in energy bins."""
        counts, exposure, energy_edges_keV, counts_bkg = cls._validate_binned_spectrum(
            counts=counts, exposure=exposure, energy_edges_keV=energy_edges_keV, counts_bkg=counts_bkg)
        if bkg_exposure is not None and (not np.isfinite(float(bkg_exposure)) or float(bkg_exposure) <= 0):
            raise ValueError("bkg_exposure must be finite and positive")
        dataset = SpectralDataset(
            counts=counts,
            exposure=exposure,
            energy_edges_keV=energy_edges_keV,
            counts_bkg=counts_bkg,
            bkg_exposure=bkg_exposure,
            name=name,
            **kwargs,
        )
        return cls(dataset=dataset, name=name, instrument=instrument, energy_range=energy_range)

    @classmethod
    def from_count_rate_density(
            cls, name: str, count_rate_density: np.ndarray, exposure: float, energy_edges_keV: np.ndarray,
            background_rate_density: np.ndarray = None, bkg_exposure: float = None, instrument: str = None,
            energy_range: tuple[float, float] = None, **kwargs) -> "FXT":
        """
        Build an FXT from count-rate density data in counts/s/keV.

        The values are converted back to counts using exposure and bin width so that
        Poisson spectral likelihoods and count-spectrum plotting remain consistent.
        """
        energy_edges_keV = np.asarray(energy_edges_keV, dtype=float)
        count_rate_density = np.asarray(count_rate_density, dtype=float)
        if count_rate_density.ndim != 1:
            raise ValueError("count_rate_density must be a one-dimensional array")
        if energy_edges_keV.ndim != 1 or len(energy_edges_keV) != len(count_rate_density) + 1:
            raise ValueError("energy_edges_keV must have one more edge than count_rate_density values")
        if not np.all(np.isfinite(count_rate_density)) or np.any(count_rate_density < 0):
            raise ValueError("count_rate_density must be finite and non-negative")
        widths = energy_edges_keV[1:] - energy_edges_keV[:-1]
        counts = count_rate_density * float(exposure) * widths
        counts_bkg = None
        background_exposure = bkg_exposure if bkg_exposure is not None else exposure
        if background_rate_density is not None:
            background_rate_density = np.asarray(background_rate_density, dtype=float)
            if background_rate_density.shape != count_rate_density.shape:
                raise ValueError("background_rate_density must have the same shape as count_rate_density")
            if not np.all(np.isfinite(background_rate_density)) or np.any(background_rate_density < 0):
                raise ValueError("background_rate_density must be finite and non-negative")
            counts_bkg = background_rate_density * float(background_exposure) * widths
        return cls.from_counts(
            name=name,
            counts=counts,
            exposure=exposure,
            energy_edges_keV=energy_edges_keV,
            counts_bkg=counts_bkg,
            bkg_exposure=background_exposure if counts_bkg is not None else None,
            instrument=instrument,
            energy_range=energy_range,
            **kwargs,
        )

    @classmethod
    def from_ogip(
            cls, pha: str, rmf: Optional[str] = None, arf: Optional[str] = None,
            bkg: Optional[str] = None, spectrum_index: Optional[int] = None,
            name: Optional[str] = None, energy_edges_keV=None, instrument: str = None,
            energy_range: tuple[float, float] = None) -> "FXT":
        dataset = SpectralDataset.from_ogip(
            pha=pha, rmf=rmf, arf=arf, bkg=bkg, spectrum_index=spectrum_index,
            name=name or "fxt", energy_edges_keV=energy_edges_keV)
        return cls(dataset=dataset, name=dataset.name, instrument=instrument, energy_range=energy_range)

    @classmethod
    def from_ogip_directory(
            cls, directory: str, pha: Optional[str] = None, bkg: Optional[str] = None,
            rmf: Optional[str] = None, arf: Optional[str] = None, spectrum_index: Optional[int] = None,
            name: Optional[str] = None, instrument: str = None,
            energy_range: tuple[float, float] = None) -> "FXT":
        dataset = SpectralDataset.from_ogip_directory(
            directory=directory, pha=pha, bkg=bkg, rmf=rmf, arf=arf,
            spectrum_index=spectrum_index, name=name or "fxt")
        return cls(dataset=dataset, name=dataset.name, instrument=instrument, energy_range=energy_range)

    @classmethod
    def from_simulator(cls, sim, time_bins, name: Optional[str] = None,
                       instrument: str = None, energy_range: tuple[float, float] = None) -> "FXT":
        dataset = SpectralDataset.from_simulator(sim=sim, time_bins=time_bins)
        dataset.name = name or "fxt"
        return cls(dataset=dataset, name=dataset.name, instrument=instrument, energy_range=energy_range)
