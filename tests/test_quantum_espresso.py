"""
Tests for Quantum Espresso (QE) defect parsing and charge corrections using
``doped``.
"""

# TODO: Change beta to Angstrom
# TODO: Add one sxd beta=3 test
# TODO: Some significant duplicate code here, get Claude to reduce
# TODO: Cube files take up significant space, like LOCPOTs; should compress with gzip and ensure
#  parseable like this

import os
import unittest

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from monty.serialization import loadfn
from test_analysis import _create_dp_and_capture_warnings, check_DefectsParser
from test_utils import EXAMPLE_DIR, custom_mpl_image_compare, if_present_rm

from doped.corrections import get_kumagai_correction

mpl.use("Agg")  # don't show interactive plots if testing from CLI locally

BOHR_TO_ANGSTROM = 0.529177  # sxdefectalign outputs distances in Bohr

# MgO dielectric constant, used throughout
MGO_DIELECTRIC = 8.8963


def _load_sxdefectalign_vatoms(vatoms_path):
    """
    Load ``sxdefectalign`` ``vAtoms.dat`` and return a tuple of
    (distance (in Å), Vlr, Vdef - Vbulk, Vdef - Vbulk - Vlr).

    Potentials are sign-flipped (multiplied by -1) to match doped (VASP)
    convention (electron charge is negative), as sxdefectalign uses the
    opposite convention (electron charge is positive).
    """
    data = np.loadtxt(vatoms_path)
    distance = data[:, 0] * BOHR_TO_ANGSTROM  # Bohr -> Angstrom
    vlr = -data[:, 1]  # Vlr (sign-flipped)
    v_def_minus_bulk = -data[:, 2]  # Vdef - Vbulk (sign-flipped)
    v_def_minus_bulk_minus_vlr = -data[:, 3]  # Vdef - Vbulk - Vlr (sign-flipped)
    return distance, vlr, v_def_minus_bulk, v_def_minus_bulk_minus_vlr


class QEDefectsParserTestCase(unittest.TestCase):
    """
    Test QE defect parsing using pre-computed defect dicts from MgO examples.
    """

    # TODO: Will likely move much of this data to tests/data in the near future.
    # TODO: Add tutorial for defect generation and parsing with QE.

    @classmethod
    def setUpClass(cls):
        cls.MgO_QE_DIR = os.path.join(EXAMPLE_DIR, "MgO_qe")
        cls.MgO_VASP_DIR = os.path.join(EXAMPLE_DIR, "MgO/Defects/Pre_Calculated_Results")
        cls.MgO_VASP_ENCUT400_DIR = os.path.join(EXAMPLE_DIR, "MgO/ENCUT_400_Defects")

        # Load pre-computed defect dicts (much faster than re-parsing cube files)
        cls.qe_defect_dict = loadfn(os.path.join(cls.MgO_QE_DIR, "MgO_defect_dict.json.gz"))
        cls.qe_defect_dict_beta_1_2 = loadfn(
            os.path.join(cls.MgO_QE_DIR, "MgO_defect_dict_beta_1.2.json.gz")
        )
        cls.vasp_defect_dict = loadfn(os.path.join(cls.MgO_VASP_DIR, "MgO_defect_dict.json.gz"))
        cls.vasp_encut400_defect_dict = loadfn(
            os.path.join(cls.MgO_VASP_ENCUT400_DIR, "MgO_defect_dict.json.gz")
        )

    def tearDown(self):
        if_present_rm(os.path.join(self.MgO_QE_DIR, "MgO_defect_dict.json.gz.bak"))
        plt.close("all")

    # --- QE defect dict structure tests ---

    def test_qe_defect_dict_keys(self):
        """
        Test that the QE defect dict contains expected defect entries.
        """
        expected_keys = {"Mg_O_+1", "Mg_O_+2", "Mg_O_+3", "Mg_O_+4", "Mg_O_Unrelaxed_+1"}
        assert set(self.qe_defect_dict.keys()) == expected_keys

    def test_vasp_defect_dict_keys(self):
        """
        Test that the VASP defect dict contains expected defect entries.
        """
        expected_keys = {"Mg_O_+1", "Mg_O_+2", "Mg_O_+3", "Mg_O_+4", "Mg_O_0"}
        assert set(self.vasp_defect_dict.keys()) == expected_keys

    def test_qe_corrections_nonzero(self):
        """
        Test that all charged QE defects have non-zero corrections.
        """
        for name, entry in self.qe_defect_dict.items():
            assert sum(entry.corrections.values()) != 0, f"Zero correction for {name}"

    # --- QE correction value tests (with default beta=0.5) ---

    def test_qe_kumagai_correction_q1_unrelaxed(self):
        """
        Test eFNV correction for unrelaxed q=+1 QE defect.
        """
        entry = self.qe_defect_dict["Mg_O_Unrelaxed_+1"]
        assert np.isclose(
            entry.corrections["kumagai_charge_correction"], 0.3098, atol=2e-3
        ), f"Got {entry.corrections['kumagai_charge_correction']}"

    def test_qe_kumagai_correction_q1(self):
        """
        Test eFNV correction for q=+1 QE defect.
        """
        entry = self.qe_defect_dict["Mg_O_+1"]
        assert np.isclose(
            entry.corrections["kumagai_charge_correction"], 0.1934, atol=2e-3
        ), f"Got {entry.corrections['kumagai_charge_correction']}"

    def test_qe_kumagai_correction_q2(self):
        """
        Test eFNV correction for q=+2 QE defect.
        """
        entry = self.qe_defect_dict["Mg_O_+2"]
        assert np.isclose(
            entry.corrections["kumagai_charge_correction"], 0.7247, atol=2e-3
        ), f"Got {entry.corrections['kumagai_charge_correction']}"

    def test_qe_kumagai_correction_q3(self):
        """
        Test eFNV correction for q=+3 QE defect.
        """
        entry = self.qe_defect_dict["Mg_O_+3"]
        assert np.isclose(
            entry.corrections["kumagai_charge_correction"], 1.573, atol=2e-3
        ), f"Got {entry.corrections['kumagai_charge_correction']}"

    def test_qe_kumagai_correction_q4(self):
        """
        Test eFNV correction for q=+4 QE defect.
        """
        entry = self.qe_defect_dict["Mg_O_+4"]
        assert np.isclose(
            entry.corrections["kumagai_charge_correction"], 2.6007, atol=2e-3
        ), f"Got {entry.corrections['kumagai_charge_correction']}"

    # --- VASP correction value tests (reference values) ---

    def test_vasp_kumagai_correction_q1(self):
        """
        Test eFNV correction for q=+1 VASP defect.
        """
        entry = self.vasp_defect_dict["Mg_O_+1"]
        assert np.isclose(
            entry.corrections["kumagai_charge_correction"], 0.1988, atol=2e-3
        ), f"Got {entry.corrections['kumagai_charge_correction']}"

    def test_vasp_kumagai_correction_q2(self):
        """
        Test eFNV correction for q=+2 VASP defect.
        """
        entry = self.vasp_defect_dict["Mg_O_+2"]
        assert np.isclose(
            entry.corrections["kumagai_charge_correction"], 0.7233, atol=2e-3
        ), f"Got {entry.corrections['kumagai_charge_correction']}"

    def test_vasp_kumagai_correction_q3(self):
        """
        Test eFNV correction for q=+3 VASP defect.
        """
        entry = self.vasp_defect_dict["Mg_O_+3"]
        assert np.isclose(
            entry.corrections["kumagai_charge_correction"], 1.572, atol=2e-3
        ), f"Got {entry.corrections['kumagai_charge_correction']}"

    def test_vasp_kumagai_correction_q4(self):
        """
        Test eFNV correction for q=+4 VASP defect.
        """
        entry = self.vasp_defect_dict["Mg_O_+4"]
        assert np.isclose(
            entry.corrections["kumagai_charge_correction"], 2.5860, atol=2e-3
        ), f"Got {entry.corrections['kumagai_charge_correction']}"

    # --- QE vs VASP correction comparison (default beta=0.5) ---

    def test_qe_vs_vasp_correction_q1(self):
        """
        Test QE vs VASP eFNV correction agreement for q=+1.

        With beta=0.5 (default), QE and VASP corrections should agree within
        ~0.01 eV for MgO.
        """
        qe_corr = self.qe_defect_dict["Mg_O_+1"].corrections["kumagai_charge_correction"]
        vasp_corr = self.vasp_defect_dict["Mg_O_+1"].corrections["kumagai_charge_correction"]
        assert np.isclose(
            qe_corr, vasp_corr, atol=0.01
        ), f"QE ({qe_corr:.4f}) vs VASP ({vasp_corr:.4f}) differ by {abs(qe_corr - vasp_corr):.4f} eV"

    def test_qe_vs_vasp_correction_q2(self):
        """
        Test QE vs VASP eFNV correction agreement for q=+2.
        """
        qe_corr = self.qe_defect_dict["Mg_O_+2"].corrections["kumagai_charge_correction"]
        vasp_corr = self.vasp_defect_dict["Mg_O_+2"].corrections["kumagai_charge_correction"]
        assert np.isclose(
            qe_corr, vasp_corr, atol=0.01
        ), f"QE ({qe_corr:.4f}) vs VASP ({vasp_corr:.4f}) differ by {abs(qe_corr - vasp_corr):.4f} eV"

    def test_qe_vs_vasp_correction_q3(self):
        """
        Test QE vs VASP eFNV correction agreement for q=+3.
        """
        qe_corr = self.qe_defect_dict["Mg_O_+3"].corrections["kumagai_charge_correction"]
        vasp_corr = self.vasp_defect_dict["Mg_O_+3"].corrections["kumagai_charge_correction"]
        assert np.isclose(
            qe_corr, vasp_corr, atol=0.01
        ), f"QE ({qe_corr:.4f}) vs VASP ({vasp_corr:.4f}) differ by {abs(qe_corr - vasp_corr):.4f} eV"

    def test_qe_vs_vasp_correction_q4(self):
        """
        Test QE vs VASP eFNV correction agreement for q=+4.
        """
        qe_corr = self.qe_defect_dict["Mg_O_+4"].corrections["kumagai_charge_correction"]
        vasp_corr = self.vasp_defect_dict["Mg_O_+4"].corrections["kumagai_charge_correction"]
        assert np.isclose(
            qe_corr, vasp_corr, atol=0.02
        ), f"QE ({qe_corr:.4f}) vs VASP ({vasp_corr:.4f}) differ by {abs(qe_corr - vasp_corr):.4f} eV"

    def test_qe_vs_vasp_unrelaxed_correction(self):
        """
        Test QE vs VASP (ENCUT=400) eFNV correction for unrelaxed q=+1.

        The unrelaxed defect with matching ENCUT should show very close
        agreement between QE and VASP.
        """
        qe_corr = self.qe_defect_dict["Mg_O_Unrelaxed_+1"].corrections["kumagai_charge_correction"]
        vasp_corr = self.vasp_encut400_defect_dict["Mg_O_Unrelaxed_+1"].corrections[
            "kumagai_charge_correction"
        ]
        assert np.isclose(qe_corr, vasp_corr, atol=0.005), (
            f"QE ({qe_corr:.4f}) vs VASP ENCUT=400 ({vasp_corr:.4f}) differ by "
            f"{abs(qe_corr - vasp_corr):.4f} eV"
        )

    def test_qe_vs_vasp_correction_all_charges_summary(self):
        """
        Test that QE vs VASP average correction difference is small across all
        charge states.

        With default beta=0.5, the average difference should be <0.01 eV.
        """
        diffs = []
        for charge in ["+1", "+2", "+3", "+4"]:
            key = f"Mg_O_{charge}"
            qe_corr = self.qe_defect_dict[key].corrections["kumagai_charge_correction"]
            vasp_corr = self.vasp_defect_dict[key].corrections["kumagai_charge_correction"]
            diffs.append(abs(qe_corr - vasp_corr))

        avg_diff = np.mean(diffs)
        assert avg_diff < 0.01, (
            f"Average QE-VASP correction difference {avg_diff:.4f} eV exceeds 0.01 eV. "
            f"Per-charge diffs: {[f'{d:.4f}' for d in diffs]}"
        )

    # --- Correction error tests ---

    def test_qe_correction_errors_small(self):
        """
        Test that QE correction errors are reasonably small.
        """
        for name, entry in self.qe_defect_dict.items():
            error = entry.corrections_metadata.get("kumagai_charge_correction_error", 0)
            assert error < 0.02, f"Correction error {error:.4f} too large for {name}"

    def test_vasp_correction_errors_small(self):
        """
        Test that VASP correction errors are reasonably small.
        """
        for name, entry in self.vasp_defect_dict.items():
            if entry.corrections:  # skip neutral
                error = entry.corrections_metadata.get("kumagai_charge_correction_error", 0)
                assert error < 0.02, f"Correction error {error:.4f} too large for {name}"

    # --- Recalculated correction tests (using get_kumagai_correction) ---

    def test_recalculate_qe_kumagai_correction_q2(self):
        """
        Test that recalculating the correction from stored metadata matches.
        """
        entry = self.qe_defect_dict["Mg_O_+2"]
        corr = get_kumagai_correction(entry, dielectric=MGO_DIELECTRIC, verbose=False)
        assert np.isclose(
            corr.correction_energy, 0.7246, atol=2e-3
        ), f"Recalculated correction {corr.correction_energy:.4f} differs from expected"

    def test_recalculate_vasp_kumagai_correction_q2(self):
        """
        Test that recalculating the VASP correction from stored metadata
        matches.
        """
        entry = self.vasp_defect_dict["Mg_O_+2"]
        corr = get_kumagai_correction(entry, dielectric=MGO_DIELECTRIC, verbose=False)
        assert np.isclose(
            corr.correction_energy, 0.7233, atol=2e-3
        ), f"Recalculated correction {corr.correction_energy:.4f} differs from expected"

    # --- Beta = 1.2 correction tests ---
    # With a larger beta (1.2 Bohr), QE corrections deviate more from VASP
    # compared to the default beta=0.5. This tests that behaviour is as expected.

    def test_qe_beta_1_2_correction_q1(self):
        """
        Test eFNV correction for q=+1 QE defect with beta=1.2.
        """
        entry = self.qe_defect_dict_beta_1_2["Mg_O_+1"]
        assert np.isclose(
            entry.corrections["kumagai_charge_correction"], 0.1608, atol=2e-3
        ), f"Got {entry.corrections['kumagai_charge_correction']}"

    def test_qe_beta_1_2_correction_q2(self):
        """
        Test eFNV correction for q=+2 QE defect with beta=1.2.
        """
        entry = self.qe_defect_dict_beta_1_2["Mg_O_+2"]
        assert np.isclose(
            entry.corrections["kumagai_charge_correction"], 0.6567, atol=2e-3
        ), f"Got {entry.corrections['kumagai_charge_correction']}"

    def test_qe_beta_1_2_correction_q3(self):
        """
        Test eFNV correction for q=+3 QE defect with beta=1.2.
        """
        entry = self.qe_defect_dict_beta_1_2["Mg_O_+3"]
        assert np.isclose(
            entry.corrections["kumagai_charge_correction"], 1.4711, atol=2e-3
        ), f"Got {entry.corrections['kumagai_charge_correction']}"

    def test_qe_beta_1_2_correction_q4(self):
        """
        Test eFNV correction for q=+4 QE defect with beta=1.2.
        """
        entry = self.qe_defect_dict_beta_1_2["Mg_O_+4"]
        assert np.isclose(
            entry.corrections["kumagai_charge_correction"], 2.4662, atol=2e-3
        ), f"Got {entry.corrections['kumagai_charge_correction']}"

    def test_qe_beta_1_2_vs_vasp_larger_deviation(self):
        """
        Test that QE corrections with beta=1.2 deviate more from VASP than with
        the default beta=0.5.

        This validates that beta=0.5 is the better default choice.
        """
        diffs_beta_0_5 = []
        diffs_beta_1_2 = []
        for charge in ["+1", "+2", "+3", "+4"]:
            key = f"Mg_O_{charge}"
            vasp_corr = self.vasp_defect_dict[key].corrections["kumagai_charge_correction"]
            qe_corr_0_5 = self.qe_defect_dict[key].corrections["kumagai_charge_correction"]
            qe_corr_1_2 = self.qe_defect_dict_beta_1_2[key].corrections["kumagai_charge_correction"]
            diffs_beta_0_5.append(abs(qe_corr_0_5 - vasp_corr))
            diffs_beta_1_2.append(abs(qe_corr_1_2 - vasp_corr))

        avg_diff_0_5 = np.mean(diffs_beta_0_5)
        avg_diff_1_2 = np.mean(diffs_beta_1_2)
        assert avg_diff_1_2 > avg_diff_0_5, (
            f"beta=1.2 avg diff ({avg_diff_1_2:.4f}) should be larger than "
            f"beta=0.5 avg diff ({avg_diff_0_5:.4f})"
        )
        # beta=1.2 should have noticeably larger deviations (>0.05 eV average)
        assert avg_diff_1_2 > 0.05, f"beta=1.2 avg QE-VASP diff ({avg_diff_1_2:.4f}) unexpectedly small"


class QEDefectsParserFromScratchTestCase(unittest.TestCase):
    """
    Test QE ``DefectsParser`` parsing from scratch (not from pre-computed
    JSON), using ``_create_dp_and_capture_warnings`` and
    ``check_DefectsParser`` from ``test_analysis``.
    """

    @classmethod
    def setUpClass(cls):
        cls.MgO_QE_DIR = os.path.join(EXAMPLE_DIR, "MgO_qe")
        cls.pp_folder = os.path.join(EXAMPLE_DIR, "pp_folder")
        cls.bulk_path = os.path.join(cls.MgO_QE_DIR, "MgO_bulk")
        cls.qe_defect_dict = loadfn(os.path.join(cls.MgO_QE_DIR, "MgO_defect_dict.json.gz"))

    def tearDown(self):
        if_present_rm(os.path.join(self.MgO_QE_DIR, "MgO_defect_dict.json.gz.bak"))
        plt.close("all")

    def test_qe_defects_parser_from_scratch(self):
        """
        Test parsing QE MgO defects from scratch with ``DefectsParser``.
        """
        dp, w = _create_dp_and_capture_warnings(
            code="espresso",
            output_path=self.MgO_QE_DIR,
            dielectric=MGO_DIELECTRIC,
            bulk_path=self.bulk_path,
            pp_folder=self.pp_folder,
        )
        assert any(  # QE parsing should warn about projected magnetisation
            "Projected magnetisation not implemented for QE" in str(warn.message) for warn in w
        ), f"Expected projected magnetisation warning, got: {[str(x.message) for x in w]}"
        check_DefectsParser(dp, band_gap=7.8)

        with pytest.raises(ValueError) as exc:
            dp.get_defect_thermodynamics()
        assert (
            "No band gap value was supplied or able to be parsed from the defect entries "
            "(calculation_metadata attributes). Please specify the band gap value in the function input."
            in str(exc.value)
        ), f"Expected band gap error, got: {exc.value!s}"

    def test_qe_defects_parser_from_scratch_beta_1_2(self):
        """
        Test parsing QE MgO defects from scratch with ``DefectsParser`` and
        ``beta=1.2`` for the charge correction.
        """
        dp, w = _create_dp_and_capture_warnings(
            code="espresso",
            output_path=self.MgO_QE_DIR,
            dielectric=MGO_DIELECTRIC,
            bulk_path=self.bulk_path,
            pp_folder=self.pp_folder,
            beta=1.2,
            json_filename=os.path.join(self.MgO_QE_DIR, "MgO_defect_dict_beta_1.2.json.gz"),
        )
        assert any(  # QE parsing should warn about projected magnetisation
            "Projected magnetisation not implemented for QE" in str(warn.message) for warn in w
        ), f"Expected projected magnetisation warning, got: {[str(x.message) for x in w]}"
        check_DefectsParser(dp, band_gap=7.8)

        # check that the charge corrections are different from the default beta=0.5
        for name, entry in dp.defect_dict.items():
            assert (
                entry.corrections["kumagai_charge_correction"]
                != self.qe_defect_dict[name].corrections["kumagai_charge_correction"]
            ), f"Charge correction for {name} with beta=1.2 should be different from default beta=0.5"

        # specific charge correction values for beta=1.2 tested above

    @pytest.mark.xfail(
        reason="QE DefectsParser._parse_parsing_warnings not yet implemented",
        raises=AttributeError,
    )
    def test_qe_defects_parser_from_scratch_no_multiprocessing(self):
        """
        Test parsing QE MgO defects from scratch with ``DefectsParser``, with
        no multiprocessing.

        Currently xfail due to incomplete QE parsing code
        (``_parse_parsing_warnings`` not defined for espresso, fails with
        processes=1).  TODO: Fix this.
        """
        dp, w = _create_dp_and_capture_warnings(
            code="espresso",
            output_path=self.MgO_QE_DIR,
            dielectric=MGO_DIELECTRIC,
            bulk_path=self.bulk_path,
            pp_folder=self.pp_folder,
            processes=1,
        )
        check_DefectsParser(dp)
        assert any(  # QE parsing should warn about projected magnetisation
            "Projected magnetisation not implemented for QE" in str(warn.message) for warn in w
        ), f"Expected projected magnetisation warning, got: {[str(x.message) for x in w]}"

    def test_qe_defects_parser_skip_corrections(self):
        """
        Test QE DefectsParser with skip_corrections=True.
        """
        dp, _w = _create_dp_and_capture_warnings(
            code="espresso",
            output_path=self.MgO_QE_DIR,
            dielectric=MGO_DIELECTRIC,
            bulk_path=self.bulk_path,
            pp_folder=self.pp_folder,
            skip_corrections=True,
            json_filename=False,  # don't save JSONs with no corrections
        )
        check_DefectsParser(dp, skip_corrections=True, band_gap=7.8)

        for name, entry in dp.defect_dict.items():
            assert (  # with skip_corrections, charged defects should have no corrections
                sum(entry.corrections.values()) == 0
            ), f"Expected zero correction for {name} with skip_corrections=True"

    def test_qe_defects_parser_no_dielectric_warning(self):
        """
        Test that QE DefectsParser warns when no dielectric is provided.
        """
        dp, w = _create_dp_and_capture_warnings(
            code="espresso",
            output_path=self.MgO_QE_DIR,
            bulk_path=self.bulk_path,
            pp_folder=self.pp_folder,
            json_filename=False,  # don't save JSONs with no corrections
        )
        assert any(
            "The dielectric constant (`dielectric`) is needed to compute finite-size charge "
            "corrections, but none was provided" in str(warn.message)
            for warn in w
        ), f"Expected dielectric warning, got: {[str(x.message) for x in w]}"
        check_DefectsParser(dp, skip_corrections=True, band_gap=7.8)

    def test_check_defects_parser_on_loaded_qe_dict(self):
        """
        Test ``check_DefectsParser``-style checks on pre-loaded QE defect dict
        entries, validating structure and metadata.
        """
        qe_defect_dict = loadfn(os.path.join(self.MgO_QE_DIR, "MgO_defect_dict.json.gz"))

        for name, defect_entry in qe_defect_dict.items():
            print(f"Checking {name}")
            assert name == defect_entry.name
            assert sum(defect_entry.corrections.values()) != 0, f"Zero correction for {name}"
            assert defect_entry.get_ediff()
            assert defect_entry.calculation_metadata

            # check defect equivalent sites and multiplicity
            print(
                f"  multiplicity: {len(defect_entry.defect.equivalent_sites)} == "
                f"{defect_entry.defect.multiplicity}"
            )
            assert len(defect_entry.defect.equivalent_sites) == defect_entry.defect.multiplicity
            assert defect_entry.defect.site in defect_entry.defect.equivalent_sites


class QEvsVASPCorrectionPlottingTestCase(unittest.TestCase):
    """
    Test eFNV correction plotting for QE and VASP defects, including side-by-
    side comparison plots.
    """

    @classmethod
    def setUpClass(cls):
        cls.MgO_QE_DIR = os.path.join(EXAMPLE_DIR, "MgO_qe")
        cls.MgO_VASP_DIR = os.path.join(EXAMPLE_DIR, "MgO/Defects/Pre_Calculated_Results")
        cls.MgO_VASP_ENCUT400_DIR = os.path.join(EXAMPLE_DIR, "MgO/ENCUT_400_Defects")

        cls.qe_defect_dict = loadfn(os.path.join(cls.MgO_QE_DIR, "MgO_defect_dict.json.gz"))
        cls.qe_defect_dict_beta_1_2 = loadfn(
            os.path.join(cls.MgO_QE_DIR, "MgO_defect_dict_beta_1.2.json.gz")
        )
        cls.vasp_defect_dict = loadfn(os.path.join(cls.MgO_VASP_DIR, "MgO_defect_dict.json.gz"))
        cls.vasp_encut400_defect_dict = loadfn(
            os.path.join(cls.MgO_VASP_ENCUT400_DIR, "MgO_defect_dict.json.gz")
        )

    def tearDown(self):
        plt.close("all")

    @staticmethod
    def _make_side_by_side(fig_left, fig_right, figsize=(7.5, 3.5), subtitles=None):
        """
        Create a side-by-side comparison figure from two eFNV plot figures.

        Unifies axis limits and renders both figures as subplots. Optionally,
        provide `subtitles` as a tuple/list of strings for (left, right)
        subplots.
        """
        # unify axis limits
        axes_left = fig_left.get_axes()
        axes_right = fig_right.get_axes()
        n_ax = min(len(axes_left), len(axes_right))
        for i in range(n_ax):
            x0 = min(axes_left[i].get_xlim()[0], axes_right[i].get_xlim()[0])
            x1 = max(axes_left[i].get_xlim()[1], axes_right[i].get_xlim()[1])
            y0 = min(axes_left[i].get_ylim()[0], axes_right[i].get_ylim()[0])
            y1 = max(axes_left[i].get_ylim()[1], axes_right[i].get_ylim()[1])
            for ax in (axes_left[i], axes_right[i]):
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)

        axes_right[0].set_ylabel("")
        axes_right[0].set_yticklabels([])

        # render both into a new figure, tightly packing to reduce horizontal whitespace
        fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"wspace": 0})
        for idx, (ax, f) in enumerate(zip(axs, (fig_left, fig_right), strict=False)):
            f.canvas.draw()
            w, h = f.canvas.get_width_height()
            rgb = np.frombuffer(f.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]

            if idx == 1:  # For right subplot, trim 15% of the left side (as we've removed the ylabel)
                rgb = rgb[:, int(w * 0.15) :, :]

            ax.imshow(rgb)
            ax.axis("off")
            if subtitles and idx < len(subtitles) and subtitles[idx]:  # Set subtitles if provided
                ax.set_title(subtitles[idx], fontsize=16, pad=8)
        fig.subplots_adjust(wspace=0.05)
        return fig

    # --- Side-by-side QE vs VASP eFNV plots ---

    @custom_mpl_image_compare("QE_vs_VASP_Mg_O_+2_eFNV_side_by_side.png")
    def test_plot_side_by_side_efnv_q2(self):
        """
        Test side-by-side QE vs VASP eFNV plot for q=+2.
        """
        plt.clf()
        _corr_qe, fig_qe = self.qe_defect_dict["Mg_O_+2"].get_kumagai_correction(plot=True)
        _corr_vasp, fig_vasp = self.vasp_defect_dict["Mg_O_+2"].get_kumagai_correction(plot=True)
        return self._make_side_by_side(fig_qe, fig_vasp, subtitles=("QE", "VASP"))

    @custom_mpl_image_compare("QE_vs_VASP_Mg_O_+4_eFNV_side_by_side.png")
    def test_plot_side_by_side_efnv_q4(self):
        """
        Test side-by-side QE vs VASP eFNV plot for q=+4.
        """
        plt.clf()
        _corr_qe, fig_qe = self.qe_defect_dict["Mg_O_+4"].get_kumagai_correction(plot=True)
        _corr_vasp, fig_vasp = self.vasp_defect_dict["Mg_O_+4"].get_kumagai_correction(plot=True)
        return self._make_side_by_side(fig_qe, fig_vasp, subtitles=("QE", "VASP"))

    @custom_mpl_image_compare("QE_vs_VASP_ENCUT400_Unrelaxed_eFNV_side_by_side.png")
    def test_plot_side_by_side_efnv_unrelaxed(self):
        """
        Test side-by-side QE vs VASP (ENCUT=400) eFNV plot for unrelaxed q=+1.
        """
        plt.clf()
        _corr_qe, fig_qe = self.qe_defect_dict["Mg_O_Unrelaxed_+1"].get_kumagai_correction(plot=True)
        _corr_vasp, fig_vasp = self.vasp_encut400_defect_dict["Mg_O_Unrelaxed_+1"].get_kumagai_correction(
            plot=True
        )
        return self._make_side_by_side(fig_qe, fig_vasp, subtitles=("QE, Unrelaxed", "VASP, ENCUT=400"))

    # --- Beta = 1.2 QE eFNV plots ---
    @custom_mpl_image_compare("QE_beta0.5_vs_beta1.2_Mg_O_+2_eFNV_side_by_side.png")
    def test_plot_side_by_side_beta_0_5_vs_1_2_q2(self):
        """
        Test side-by-side QE beta=0.5 vs beta=1.2 eFNV plot for q=+2.

        Visually demonstrates how increased beta leads to noisier site
        potentials and larger correction errors.
        """
        plt.clf()
        _corr_0_5, fig_0_5 = self.qe_defect_dict["Mg_O_+2"].get_kumagai_correction(plot=True)
        _corr_1_2, fig_1_2 = self.qe_defect_dict_beta_1_2["Mg_O_+2"].get_kumagai_correction(plot=True)
        return self._make_side_by_side(fig_0_5, fig_1_2, subtitles=("beta=0.5", "beta=1.2"))

    @custom_mpl_image_compare("QE_beta1.2_vs_VASP_Mg_O_+4_eFNV_side_by_side.png")
    def test_plot_side_by_side_beta_1_2_vs_vasp_q4(self):
        """
        Test side-by-side QE beta=1.2 vs VASP eFNV plot for q=+4.

        Shows the larger QE-VASP discrepancy at beta=1.2 compared to the
        default beta=0.5.
        """
        plt.clf()
        _corr_qe, fig_qe = self.qe_defect_dict_beta_1_2["Mg_O_+4"].get_kumagai_correction(plot=True)
        _corr_vasp, fig_vasp = self.vasp_defect_dict["Mg_O_+4"].get_kumagai_correction(plot=True)
        return self._make_side_by_side(fig_qe, fig_vasp, subtitles=("QE, beta=1.2", "VASP"))


class SxdefectalignComparisonTestCase(unittest.TestCase):
    """
    Test doped charge corrections against ``sxdefectalign`` reference data for
    both ``QE`` and ``VASP``.

    The ``QE`` charge correction approach in ``doped`` should be fully
    equivalent (outside of small numerical differences) to the
    ``sxdefectalign`` atomic-sphere averaging approach, while the ``VASP`` eFNV
    correction approach differs slightly, by using a test charge of norm 1 over
    the ``RWIGS`` radii of the atomic sites.

    Note that the ``sxdefectalign`` data here is for ``beta=1.2``, unless
    otherwise specified.
    """

    @classmethod
    def setUpClass(cls):
        cls.MgO_QE_DIR = os.path.join(EXAMPLE_DIR, "MgO_qe")
        cls.MgO_VASP_DIR = os.path.join(EXAMPLE_DIR, "MgO/Defects/Pre_Calculated_Results")
        cls.MgO_VASP_ENCUT400_DIR = os.path.join(EXAMPLE_DIR, "MgO/ENCUT_400_Defects")
        cls.MgO_QE_sxd_dir = os.path.join(cls.MgO_QE_DIR, "sxdefectalign")

        cls.qe_defect_dict = loadfn(os.path.join(cls.MgO_QE_DIR, "MgO_defect_dict.json.gz"))
        cls.qe_defect_dict_beta_1_2 = loadfn(
            os.path.join(cls.MgO_QE_DIR, "MgO_defect_dict_beta_1.2.json.gz")
        )
        cls.vasp_defect_dict = loadfn(os.path.join(cls.MgO_VASP_DIR, "MgO_defect_dict.json.gz"))
        cls.vasp_encut400_defect_dict = loadfn(
            os.path.join(cls.MgO_VASP_ENCUT400_DIR, "MgO_defect_dict.json.gz")
        )

    def tearDown(self):
        plt.close("all")

    # --- QE-doped vs QE-sxdefectalign ---

    @staticmethod
    def _compare_doped_vs_sxdefectalign(entry, vatoms_path, dielectric, atol=0.01, match_expected=True):
        """
        Compare ``doped`` eFNV site potentials with ``sxdefectalign``
        ``vAtoms.dat`` data.

        ``doped`` and ``sxdefectalign`` may have slightly different site counts
        (``sxdefectalign`` may exclude the defect site or nearby sites, due to
        failed site-matching), so we compare using sites matched by distance
        rather than assuming 1:1 matching.
        """
        sx_dist, _sx_vlr, sx_v_diff, _sx_v_diff_lr = _load_sxdefectalign_vatoms(vatoms_path)

        corr = get_kumagai_correction(entry, dielectric=dielectric, verbose=False)
        efnv_data = corr.metadata["pydefect_ExtendedFnvCorrection"]

        doped_distances = np.array([float(s.distance) for s in efnv_data.sites])
        doped_potentials = np.array([float(s.potential) for s in efnv_data.sites])

        # site counts should be close (within a few sites)
        assert (
            abs(len(doped_distances) - len(sx_dist)) <= 5
        ), f"Site count mismatch too large: doped={len(doped_distances)}, sx={len(sx_dist)}"

        # compare far-field potentials (sites > 5 Angstrom from defect) where
        # both methods should agree well
        doped_far = doped_potentials[doped_distances > 5.0]
        sx_far = sx_v_diff[sx_dist > 5.0]

        # mean absolute potential in far-field region should be consistent
        assert (
            np.isclose(np.mean(np.abs(doped_far)), np.mean(np.abs(sx_far)), atol=atol) == match_expected
        ), (
            f"Far-field mean |V| mismatch: doped={np.mean(np.abs(doped_far)):.4f}, "
            f"sx={np.mean(np.abs(sx_far)):.4f}"
        )
        return doped_distances, doped_potentials, sx_dist, sx_v_diff

    def test_qe_doped_vs_qe_sxdefectalign_unrelaxed_potentials(self):
        """
        Compare QE-doped site potential differences with QE-sxdefectalign for
        unrelaxed q=+1.
        """
        vatoms_path = os.path.join(self.MgO_QE_sxd_dir, "Mg_O_Unrelaxed_+1/vAtoms.dat")
        assert os.path.exists(vatoms_path), f"sxdefectalign data not found: {vatoms_path}"
        entry = self.qe_defect_dict_beta_1_2["Mg_O_Unrelaxed_+1"]
        self._compare_doped_vs_sxdefectalign(entry, vatoms_path, MGO_DIELECTRIC, atol=0.001)

        # shouldn't match for beta = 0.5 default (sxd data from beta = 1.2)
        entry = self.qe_defect_dict["Mg_O_Unrelaxed_+1"]
        self._compare_doped_vs_sxdefectalign(
            entry, vatoms_path, MGO_DIELECTRIC, atol=0.001, match_expected=False
        )  # drop atol for this case, as beta = 0.5 also gives a reasonable match to beta = 1.2 in the
        # unrelaxed case

    def test_qe_doped_vs_qe_sxdefectalign_q1_potentials(self):
        """
        Compare QE-doped and QE-sxdefectalign site potentials for q=+1.
        """
        vatoms_path = os.path.join(self.MgO_QE_sxd_dir, "Mg_O_+1/vAtoms.dat")
        assert os.path.exists(vatoms_path), f"sxdefectalign data not found: {vatoms_path}"
        entry = self.qe_defect_dict_beta_1_2["Mg_O_+1"]
        self._compare_doped_vs_sxdefectalign(entry, vatoms_path, MGO_DIELECTRIC, atol=0.01)

        # shouldn't match for beta = 0.5 default (sxd data from beta = 1.2)
        entry = self.qe_defect_dict["Mg_O_+1"]
        self._compare_doped_vs_sxdefectalign(
            entry, vatoms_path, MGO_DIELECTRIC, atol=0.01, match_expected=False
        )

    def test_qe_doped_vs_qe_sxdefectalign_q2_potentials(self):
        """
        Compare QE-doped and QE-sxdefectalign site potentials for q=+2.
        """
        vatoms_path = os.path.join(self.MgO_QE_sxd_dir, "Mg_O_+2/vAtoms.dat")
        assert os.path.exists(vatoms_path)
        entry = self.qe_defect_dict_beta_1_2["Mg_O_+2"]
        self._compare_doped_vs_sxdefectalign(entry, vatoms_path, MGO_DIELECTRIC, atol=0.01)

        # shouldn't match for beta = 0.5 default (sxd data from beta = 1.2)
        entry = self.qe_defect_dict["Mg_O_+2"]
        self._compare_doped_vs_sxdefectalign(
            entry, vatoms_path, MGO_DIELECTRIC, atol=0.01, match_expected=False
        )

    def test_qe_doped_vs_qe_sxdefectalign_q3_potentials(self):
        """
        Compare QE-doped and QE-sxdefectalign site potentials for q=+3.
        """
        vatoms_path = os.path.join(self.MgO_QE_sxd_dir, "Mg_O_+3/vAtoms.dat")
        assert os.path.exists(vatoms_path)
        entry = self.qe_defect_dict_beta_1_2["Mg_O_+3"]
        self._compare_doped_vs_sxdefectalign(entry, vatoms_path, MGO_DIELECTRIC, atol=0.01)

        # shouldn't match for beta = 0.5 default (sxd data from beta = 1.2)
        entry = self.qe_defect_dict["Mg_O_+3"]
        self._compare_doped_vs_sxdefectalign(
            entry, vatoms_path, MGO_DIELECTRIC, atol=0.01, match_expected=False
        )

    # --- VASP-doped vs VASP-sxdefectalign ---

    def test_vasp_doped_vs_vasp_sxdefectalign_unrelaxed_potentials(self):
        """
        Compare VASP-doped site potentials with VASP-sxdefectalign for
        unrelaxed q=+1.
        """
        vatoms_path = os.path.join(self.MgO_VASP_ENCUT400_DIR, "Mg_O_Unrelaxed_+1/vAtoms.dat")
        assert os.path.exists(vatoms_path), f"sxdefectalign data not found: {vatoms_path}"
        entry = self.vasp_encut400_defect_dict["Mg_O_Unrelaxed_+1"]
        self._compare_doped_vs_sxdefectalign(entry, vatoms_path, MGO_DIELECTRIC, atol=0.01)

    # --- QE-sxdefectalign vs VASP-sxdefectalign ---

    @custom_mpl_image_compare("QE_sxd_vs_VASP_sxd_unrelaxed_potentials.png")
    def test_qe_sxd_vs_vasp_sxd_unrelaxed_potentials(self):
        """
        Compare ``QE``-``sxdefectalign`` and ``VASP``-``sxdefectalign`` site
        potentials for unrelaxed ``q=+1``.
        """
        qe_vatoms_path = os.path.join(self.MgO_QE_sxd_dir, "Mg_O_Unrelaxed_+1/vAtoms.dat")
        vasp_vatoms_path = os.path.join(self.MgO_VASP_ENCUT400_DIR, "Mg_O_Unrelaxed_+1/vAtoms.dat")
        assert os.path.exists(qe_vatoms_path)
        assert os.path.exists(vasp_vatoms_path)

        qe_dist, _qe_vlr, qe_v_diff, _qe_v_diff_lr = _load_sxdefectalign_vatoms(qe_vatoms_path)
        vasp_dist, _vasp_vlr, vasp_v_diff, _vasp_v_diff_lr = _load_sxdefectalign_vatoms(vasp_vatoms_path)

        # same number of sites
        assert len(qe_dist) == len(vasp_dist)

        # distances should match (same structure)
        assert np.allclose(qe_dist, vasp_dist, atol=0.01), "Site distances differ between QE and VASP"

        # potentials should be close for unrelaxed case
        # near-defect sites can differ by up to ~0.035 eV, but far-field
        # sites (>5 Angstrom) should agree more tightly
        far_mask = qe_dist > 5.0
        assert np.allclose(qe_v_diff[far_mask], vasp_v_diff[far_mask], atol=0.005), (
            f"Far-field max potential difference: "
            f"{np.max(np.abs(qe_v_diff[far_mask] - vasp_v_diff[far_mask])):.4f} eV"
        )
        # all sites should agree within 0.04 eV
        assert np.allclose(
            qe_v_diff, vasp_v_diff, atol=0.04
        ), f"Max potential difference: {np.max(np.abs(qe_v_diff - vasp_v_diff)):.4f} eV"

        qe_dist, _qe_vlr, qe_v_diff, _qe_v_diff_lr = _load_sxdefectalign_vatoms(qe_vatoms_path)
        vasp_dist, _vasp_vlr, vasp_v_diff, _vasp_v_diff_lr = _load_sxdefectalign_vatoms(vasp_vatoms_path)

        fig, ax = plt.subplots()
        ax.scatter(qe_dist, qe_v_diff, label=r"$V_{\mathrm{def}} - V_{\mathrm{bulk}}$ (QE)", s=10)
        ax.scatter(vasp_dist, vasp_v_diff, label=r"$V_{\mathrm{def}} - V_{\mathrm{bulk}}$ (VASP)", s=10)
        ax.set_xlabel(r"Distance from defect ($\mathrm{\AA}$)")
        ax.set_ylabel(r"Potential $\times$ ($-1$) (eV)")
        ax.set_title(r"sxdefectalign: QE vs VASP, Mg$_\mathrm{O}$ Unrelaxed $q$=+1")
        ax.legend(fontsize=8)
        return fig

    @custom_mpl_image_compare("QE_doped_vs_sxd_q1_potentials.png")
    def test_plot_qe_doped_vs_sxd_q1(self):
        """
        Plot ``doped`` eFNV correction with ``sxdefectalign`` data overlaid for
        ``QE`` ``q=+1``.
        """
        plt.clf()
        entry = self.qe_defect_dict_beta_1_2["Mg_O_+1"]
        _corr, fig = entry.get_kumagai_correction(plot=True)

        # overlay sxdefectalign data
        vatoms_path = os.path.join(self.MgO_QE_sxd_dir, "Mg_O_+1/vAtoms.dat")
        sx_dist, _sx_vlr, sx_v_diff, _sx_v_diff_lr = _load_sxdefectalign_vatoms(vatoms_path)

        ax = fig.gca()
        ax.scatter(
            sx_dist,
            sx_v_diff,
            label=r"$V_{\mathrm{def}} - V_{\mathrm{bulk}}$ (sxd)",
            s=10,
            color="black",
            zorder=5,
        )
        ax.legend(fontsize=8)
        return fig

    @custom_mpl_image_compare("QE_doped_vs_sxd_q2_potentials.png")
    def test_plot_qe_doped_vs_sxd_q2(self):
        """
        Plot ``doped`` eFNV correction with ``sxdefectalign`` data overlaid for
        ``QE`` ``q=+2``.
        """
        plt.clf()
        entry = self.qe_defect_dict_beta_1_2["Mg_O_+2"]
        _corr, fig = entry.get_kumagai_correction(plot=True)

        vatoms_path = os.path.join(self.MgO_QE_sxd_dir, "Mg_O_+2/vAtoms.dat")
        sx_dist, _sx_vlr, sx_v_diff, _sx_v_diff_lr = _load_sxdefectalign_vatoms(vatoms_path)

        ax = fig.gca()
        ax.scatter(
            sx_dist,
            sx_v_diff,
            label=r"$V_{\mathrm{def}} - V_{\mathrm{bulk}}$ (sxd)",
            s=10,
            color="black",
            zorder=5,
        )
        ax.legend(fontsize=8)
        return fig

    @custom_mpl_image_compare("QE_doped_vs_sxd_q3_potentials.png")
    def test_plot_qe_doped_vs_sxd_q3(self):
        """
        Plot ``doped`` eFNV correction with ``sxdefectalign`` data overlaid for
        ``QE`` ``q=+3``.
        """
        plt.clf()
        entry = self.qe_defect_dict_beta_1_2["Mg_O_+3"]
        _corr, fig = entry.get_kumagai_correction(plot=True)

        vatoms_path = os.path.join(self.MgO_QE_sxd_dir, "Mg_O_+3/vAtoms.dat")
        sx_dist, _sx_vlr, sx_v_diff, _sx_v_diff_lr = _load_sxdefectalign_vatoms(vatoms_path)

        ax = fig.gca()
        ax.scatter(
            sx_dist,
            sx_v_diff,
            label=r"$V_{\mathrm{def}} - V_{\mathrm{bulk}}$ (sxd)",
            s=10,
            color="black",
            zorder=5,
        )
        ax.legend(fontsize=8)
        return fig
