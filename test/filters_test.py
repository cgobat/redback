import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from astropy.table import Table

import redback.filters as filters


class TestFilterTable(unittest.TestCase):

    def test_svofps_id_is_primary_key(self):
        table = filters.show_all_filters()
        self.assertIn("svofps_id", table.colnames)
        self.assertNotIn("bands", table.colnames)
        self.assertIn("sncosmo_name", table.colnames)
        ids = np.asarray(table["svofps_id"], dtype=str)
        self.assertTrue(np.all(ids != ""))
        self.assertEqual(len(ids), len(np.unique(ids)))

    def test_resolve_canonical_svo_id(self):
        self.assertEqual(filters.resolve_filter_id("Palomar/ZTF.r"), "Palomar/ZTF.r")

    def test_resolve_sncosmo_alias(self):
        self.assertEqual(filters.resolve_filter_id("ztfr"), "Palomar/ZTF.r")

    def test_resolve_legacy_alias(self):
        self.assertEqual(filters.resolve_filter_id("r"), "APO/SDSS.r")

    def test_legacy_alias_resolution_is_case_sensitive(self):
        self.assertEqual(filters.resolve_filter_id("R"), "Generic/Bessell.R")
        self.assertEqual(filters.resolve_filter_id("r"), "APO/SDSS.r")

    def test_canonicalize_filter_ids_preserves_shape(self):
        values = np.array([["ztfg", "Palomar/ZTF.r"], ["lssti", "2MASS/2MASS.J"]])
        result = filters.canonicalize_filter_ids(values)
        expected = np.array([
            ["Palomar/ZTF.g", "Palomar/ZTF.r"],
            ["LSST/LSST.i", "2MASS/2MASS.J"],
        ], dtype=object)
        np.testing.assert_array_equal(result, expected)

    def test_unknown_svo_id_is_accepted_for_runtime_lookup(self):
        identifier = "SomeFacility/SomeInstrument.some_filter"
        self.assertEqual(filters.resolve_filter_id(identifier), identifier)


class TestSVOProvider(unittest.TestCase):

    def setUp(self):
        filters.clear_filter_cache()

    def tearDown(self):
        filters.clear_filter_cache()

    @staticmethod
    def transmission_table():
        table = Table()
        table["Wavelength"] = np.array([4000.0, 5000.0, 6000.0])
        table["Transmission"] = np.array([0.0, 1.0, 0.0])
        return table

    @mock.patch("astroquery.svo_fps.SvoFps.get_transmission_data")
    def test_get_bandpass_sources_curve_from_svo(self, mock_get_transmission):
        mock_get_transmission.return_value = self.transmission_table()

        bandpass = filters.get_bandpass("Palomar/ZTF.r", allow_sncosmo_fallback=False)

        mock_get_transmission.assert_called_once_with("Palomar/ZTF.r", cache=True)
        self.assertEqual(bandpass.svofps_id, "Palomar/ZTF.r")
        self.assertEqual(bandpass.source, "svo")
        np.testing.assert_array_equal(bandpass.wave, [4000.0, 5000.0, 6000.0])

    @mock.patch("astroquery.svo_fps.SvoFps.get_transmission_data")
    def test_sncosmo_alias_uses_corresponding_svo_curve(self, mock_get_transmission):
        mock_get_transmission.return_value = self.transmission_table()

        bandpass = filters.get_bandpass("ztfr", allow_sncosmo_fallback=False)

        mock_get_transmission.assert_called_once_with("Palomar/ZTF.r", cache=True)
        self.assertEqual(bandpass.svofps_id, "Palomar/ZTF.r")

    @mock.patch("astroquery.svo_fps.SvoFps.get_transmission_data")
    def test_bandpass_is_cached_in_process(self, mock_get_transmission):
        mock_get_transmission.return_value = self.transmission_table()

        first = filters.get_bandpass("Palomar/ZTF.r", allow_sncosmo_fallback=False)
        second = filters.get_bandpass("Palomar/ZTF.r", allow_sncosmo_fallback=False)

        self.assertIs(first, second)
        mock_get_transmission.assert_called_once()

    @mock.patch("astroquery.svo_fps.SvoFps.get_transmission_data")
    def test_get_sncosmo_bandpass_is_built_from_svo_curve(self, mock_get_transmission):
        mock_get_transmission.return_value = self.transmission_table()

        bandpass = filters.get_sncosmo_bandpass("Palomar/ZTF.r", allow_sncosmo_fallback=False)

        self.assertEqual(bandpass.name, "Palomar/ZTF.r")
        np.testing.assert_array_equal(bandpass.wave, [4000.0, 5000.0, 6000.0])
        mock_get_transmission.assert_called_once_with("Palomar/ZTF.r", cache=True)


class TestUserFilters(unittest.TestCase):

    def test_register_user_filter_without_adding_table_identity(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as handle:
            handle.write("4000 0.0\n5000 1.0\n6000 0.0\n")
            filename = handle.name

        try:
            profile = filters.add_filter_user(filename, "my-local-filter")
            self.assertIs(filters.get_bandpass("my-local-filter"), profile)
            self.assertEqual(filters.resolve_filter_id("my-local-filter"), "my-local-filter")
            metadata = filters.get_filter_metadata("my-local-filter")
            self.assertGreater(metadata["wavelength [Hz]"], 0.0)
        finally:
            filters._USER_BANDPASSES.pop("my-local-filter", None)
            os.unlink(filename)


class TestDatabaseHelpers(unittest.TestCase):

    def test_add_to_database_uses_svo_id_as_key(self):
        database = Table(
            names=[
                "svofps_id", "wavelength [Hz]", "wavelength [Angstrom]", "color",
                "reference_flux", "sncosmo_name", "sncosmo_aliases", "legacy_aliases",
                "label", "effective_width [Hz]",
            ],
            dtype=[str, float, float, str, float, str, str, str, str, float],
        )

        filters.add_to_database(
            svofps_id="Facility/Instrument.test",
            wavelength=5.5e-7,
            zeroflux=1e-10,
            database=database,
            plot_label="Test",
            effective_width=100.0,
            sncosmo_name="legacytest",
            legacy_aliases="old-test",
        )

        self.assertEqual(len(database), 1)
        self.assertEqual(database["svofps_id"][0], "Facility/Instrument.test")
        self.assertEqual(database["sncosmo_name"][0], "legacytest")
        self.assertEqual(database["legacy_aliases"][0], "old-test")


if __name__ == "__main__":
    unittest.main()
