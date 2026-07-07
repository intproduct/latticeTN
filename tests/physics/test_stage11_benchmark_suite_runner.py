from __future__ import annotations

import json
import math
import subprocess
import sys
import uuid
from pathlib import Path


def test_quick_physics_benchmark_suite_generates_reports():
    outdir = Path(".tmp_stage11_benchmark_suite") / f"run_{uuid.uuid4().hex}"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_physics_benchmark_suite.py",
            "--suite",
            "quick",
            "--output-dir",
            str(outdir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary_json = outdir / "benchmark_summary.json"
    summary_csv = outdir / "benchmark_summary.csv"
    summary_md = outdir / "benchmark_summary.md"
    report = outdir / "PHYSICS_VALIDATION_REPORT.md"
    large_n_json = outdir / "large_n_evidence.json"
    large_n_md = outdir / "large_n_evidence.md"
    for path in [summary_json, summary_csv, summary_md, report, large_n_json, large_n_md]:
        assert path.exists(), path

    records = json.loads(summary_json.read_text(encoding="utf-8"))
    assert records
    assert all(record["pass"] for record in records)
    assert any(record["case_id"].startswith("hamiltonian_heisenberg") for record in records)
    assert any(record["suite"] == "observables" for record in records)
    assert any(record["suite"] == "fullstack" for record in records)
    assert any(record["suite"] == "policy" for record in records)
    assert any(record["suite"] == "large_n_ad" for record in records)
    assert any(record["suite"] == "literature" for record in records)
    case_ids = {record["case_id"] for record in records}
    assert "policy_cuda_quick_suite_cpu_only" in case_ids
    assert "policy_large_n_ad_runner_no_dense_or_classical_reference" in case_ids
    assert "large_n_ad_heisenberg_N20_chi8_no_reference" in case_ids
    assert "large_n_ad_heisenberg_N40_chi8_no_reference" in case_ids
    assert "large_n_ad_heisenberg_N80_chi8_no_reference" in case_ids
    assert "large_n_chi_table_heisenberg_N20_chi4_8_16_32_no_reference" in case_ids
    assert "large_n_chi_table_heisenberg_N40_chi4_8_16_32_no_reference" in case_ids
    assert "large_n_chi_table_heisenberg_N80_chi4_8_16_32_no_reference" in case_ids
    assert "large_n_ad_spinless_tv_N20_chi8_hard_sector_no_reference" in case_ids
    assert "large_n_ad_hubbard_N10_chi8_hard_sector_no_reference" in case_ids
    assert "large_n_ad_spinless_tv_N40_chi8_hard_sector_no_reference" in case_ids
    assert "large_n_ad_hubbard_N20_chi8_hard_sector_no_reference" in case_ids
    assert "large_n_ad_spinless_tv_N80_chi8_hard_sector_no_reference" in case_ids
    assert "large_n_ad_hubbard_N40_chi8_hard_sector_no_reference" in case_ids
    assert "large_n_chi_table_spinless_tv_N20_chi4_8_16_32_hard_sector_no_reference" in case_ids
    assert "large_n_chi_table_hubbard_N10_chi4_8_16_32_hard_sector_no_reference" in case_ids
    assert "large_n_chi_table_spinless_tv_N40_chi4_8_16_32_hard_sector_no_reference" in case_ids
    assert "large_n_chi_table_hubbard_N20_chi4_8_16_32_hard_sector_no_reference" in case_ids
    assert "large_n_chi_table_spinless_tv_N80_chi4_8_16_32_hard_sector_no_reference" in case_ids
    assert "large_n_chi_table_hubbard_N40_chi4_8_16_32_hard_sector_no_reference" in case_ids
    by_id = {record["case_id"]: record for record in records}
    heis_obs = by_id["large_n_ad_heisenberg_N20_chi8_no_reference"]["observables"]
    assert math.isfinite(heis_obs["local_Sz_mid"])
    assert math.isfinite(heis_obs["connected_SzSz_midbond"])
    assert heis_obs["entanglement_entropy_midbond"] >= 0.0
    assert math.isfinite(heis_obs["entanglement_entropy_midbond"])
    heis_n40_obs = by_id["large_n_ad_heisenberg_N40_chi8_no_reference"]["observables"]
    assert heis_n40_obs["final_energy"] < heis_n40_obs["initial_energy"]
    assert heis_n40_obs["max_bond"] <= 8
    assert math.isfinite(heis_n40_obs["local_Sz_mid"])
    assert math.isfinite(heis_n40_obs["connected_SzSz_midbond"])
    assert heis_n40_obs["entanglement_entropy_midbond"] >= 0.0
    assert math.isfinite(heis_n40_obs["entanglement_entropy_midbond"])
    assert heis_n40_obs["ed_used"] is False
    assert heis_n40_obs["dense_hamiltonian_built"] is False
    heis_n80_obs = by_id["large_n_ad_heisenberg_N80_chi8_no_reference"]["observables"]
    assert heis_n80_obs["final_energy"] < heis_n80_obs["initial_energy"]
    assert heis_n80_obs["max_bond"] <= 8
    assert math.isfinite(heis_n80_obs["local_Sz_mid"])
    assert math.isfinite(heis_n80_obs["connected_SzSz_midbond"])
    assert heis_n80_obs["entanglement_entropy_midbond"] >= 0.0
    assert math.isfinite(heis_n80_obs["entanglement_entropy_midbond"])
    assert heis_n80_obs["ed_used"] is False
    assert heis_n80_obs["dense_hamiltonian_built"] is False
    chi_table_obs = by_id["large_n_chi_table_heisenberg_N20_chi4_8_16_32_no_reference"]["observables"]
    assert chi_table_obs["energy_decreases_each_chi"] is True
    assert chi_table_obs["higher_chi_energy_not_worse"] is True
    assert chi_table_obs["ed_used"] is False
    assert chi_table_obs["classical_dmrg_used"] is False
    assert chi_table_obs["lanczos_used"] is False
    assert chi_table_obs["dense_hamiltonian_built"] is False
    chi_rows = chi_table_obs["chi_table"]
    assert [row["chi"] for row in chi_rows] == [4, 8, 16, 32]
    assert all(row["max_bond"] <= row["chi"] for row in chi_rows)
    assert all(math.isfinite(row["local_Sz_mid"]) for row in chi_rows)
    assert all(math.isfinite(row["connected_SzSz_midbond"]) for row in chi_rows)
    assert all(row["entanglement_entropy_midbond"] >= 0.0 for row in chi_rows)
    assert all(math.isfinite(row["entanglement_entropy_midbond"]) for row in chi_rows)
    chi_table_n40_obs = by_id["large_n_chi_table_heisenberg_N40_chi4_8_16_32_no_reference"]["observables"]
    assert chi_table_n40_obs["energy_decreases_each_chi"] is True
    assert chi_table_n40_obs["higher_chi_energy_not_worse"] is True
    assert [row["chi"] for row in chi_table_n40_obs["chi_table"]] == [4, 8, 16, 32]
    assert all(row["max_bond"] <= row["chi"] for row in chi_table_n40_obs["chi_table"])
    assert all(math.isfinite(row["local_Sz_mid"]) for row in chi_table_n40_obs["chi_table"])
    assert all(math.isfinite(row["connected_SzSz_midbond"]) for row in chi_table_n40_obs["chi_table"])
    assert all(row["entanglement_entropy_midbond"] >= 0.0 for row in chi_table_n40_obs["chi_table"])
    assert all(math.isfinite(row["entanglement_entropy_midbond"]) for row in chi_table_n40_obs["chi_table"])
    chi_table_n80_obs = by_id["large_n_chi_table_heisenberg_N80_chi4_8_16_32_no_reference"]["observables"]
    assert chi_table_n80_obs["energy_decreases_each_chi"] is True
    assert chi_table_n80_obs["higher_chi_energy_not_worse"] is True
    assert [row["chi"] for row in chi_table_n80_obs["chi_table"]] == [4, 8, 16, 32]
    assert all(row["max_bond"] <= row["chi"] for row in chi_table_n80_obs["chi_table"])
    assert all(math.isfinite(row["local_Sz_mid"]) for row in chi_table_n80_obs["chi_table"])
    assert all(math.isfinite(row["connected_SzSz_midbond"]) for row in chi_table_n80_obs["chi_table"])
    assert all(row["entanglement_entropy_midbond"] >= 0.0 for row in chi_table_n80_obs["chi_table"])
    assert all(math.isfinite(row["entanglement_entropy_midbond"]) for row in chi_table_n80_obs["chi_table"])
    spinless_obs = by_id["large_n_ad_spinless_tv_N20_chi8_hard_sector_no_reference"]["observables"]
    assert spinless_obs["additive_observables"]["abs_error"] < 1e-10
    assert math.isfinite(spinless_obs["local_density_mid"])
    assert 0.0 <= spinless_obs["local_density_mid"] <= 1.0
    hubbard_obs = by_id["large_n_ad_hubbard_N10_chi8_hard_sector_no_reference"]["observables"]
    assert hubbard_obs["additive_observables"]["n_up_abs_error"] < 1e-10
    assert hubbard_obs["additive_observables"]["n_down_abs_error"] < 1e-10
    assert math.isfinite(hubbard_obs["local_density_mid"])
    assert math.isfinite(hubbard_obs["double_occupancy_mid"])
    assert math.isfinite(hubbard_obs["local_sz_mid"])
    assert 0.0 <= hubbard_obs["local_density_mid"] <= 2.0
    assert 0.0 <= hubbard_obs["double_occupancy_mid"] <= 1.0
    assert abs(hubbard_obs["local_sz_mid"]) <= 0.5
    spinless_n40_obs = by_id["large_n_ad_spinless_tv_N40_chi8_hard_sector_no_reference"]["observables"]
    assert spinless_n40_obs["additive_observables"]["abs_error"] < 1e-10
    assert math.isfinite(spinless_n40_obs["local_density_mid"])
    assert 0.0 <= spinless_n40_obs["local_density_mid"] <= 1.0
    assert spinless_n40_obs["diagnostics"]["dense_hamiltonian_built"] is False
    hubbard_n20_obs = by_id["large_n_ad_hubbard_N20_chi8_hard_sector_no_reference"]["observables"]
    assert hubbard_n20_obs["additive_observables"]["n_up_abs_error"] < 1e-10
    assert hubbard_n20_obs["additive_observables"]["n_down_abs_error"] < 1e-10
    assert math.isfinite(hubbard_n20_obs["local_density_mid"])
    assert math.isfinite(hubbard_n20_obs["double_occupancy_mid"])
    assert math.isfinite(hubbard_n20_obs["local_sz_mid"])
    assert 0.0 <= hubbard_n20_obs["local_density_mid"] <= 2.0
    assert 0.0 <= hubbard_n20_obs["double_occupancy_mid"] <= 1.0
    assert abs(hubbard_n20_obs["local_sz_mid"]) <= 0.5
    assert hubbard_n20_obs["diagnostics"]["dense_hamiltonian_built"] is False
    spinless_n80_obs = by_id["large_n_ad_spinless_tv_N80_chi8_hard_sector_no_reference"]["observables"]
    assert spinless_n80_obs["additive_observables"]["abs_error"] < 1e-10
    assert math.isfinite(spinless_n80_obs["local_density_mid"])
    assert 0.0 <= spinless_n80_obs["local_density_mid"] <= 1.0
    assert spinless_n80_obs["diagnostics"]["dense_hamiltonian_built"] is False
    hubbard_n40_obs = by_id["large_n_ad_hubbard_N40_chi8_hard_sector_no_reference"]["observables"]
    assert hubbard_n40_obs["additive_observables"]["n_up_abs_error"] < 1e-10
    assert hubbard_n40_obs["additive_observables"]["n_down_abs_error"] < 1e-10
    assert math.isfinite(hubbard_n40_obs["local_density_mid"])
    assert math.isfinite(hubbard_n40_obs["double_occupancy_mid"])
    assert math.isfinite(hubbard_n40_obs["local_sz_mid"])
    assert 0.0 <= hubbard_n40_obs["local_density_mid"] <= 2.0
    assert 0.0 <= hubbard_n40_obs["double_occupancy_mid"] <= 1.0
    assert abs(hubbard_n40_obs["local_sz_mid"]) <= 0.5
    assert hubbard_n40_obs["diagnostics"]["dense_hamiltonian_built"] is False
    for case_id in [
        "large_n_chi_table_spinless_tv_N20_chi4_8_16_32_hard_sector_no_reference",
        "large_n_chi_table_hubbard_N10_chi4_8_16_32_hard_sector_no_reference",
        "large_n_chi_table_spinless_tv_N40_chi4_8_16_32_hard_sector_no_reference",
        "large_n_chi_table_hubbard_N20_chi4_8_16_32_hard_sector_no_reference",
        "large_n_chi_table_spinless_tv_N80_chi4_8_16_32_hard_sector_no_reference",
        "large_n_chi_table_hubbard_N40_chi4_8_16_32_hard_sector_no_reference",
    ]:
        obs = by_id[case_id]["observables"]
        assert obs["sector_clean_each_chi"] is True
        assert obs["ed_used"] is False
        assert obs["classical_dmrg_used"] is False
        assert obs["lanczos_used"] is False
        assert obs["dense_hamiltonian_built"] is False
        rows = obs["chi_table"]
        assert [row["chi"] for row in rows] == [4, 8, 16, 32]
        assert all(math.isfinite(row["final_energy"]) for row in rows)
        assert all(row["max_bond"] <= row["chi"] for row in rows)
        assert all(row["diagnostics"]["max_forbidden_abs"] < 1e-12 for row in rows)
        assert all(row["diagnostics"]["max_forbidden_grad_abs"] < 1e-12 for row in rows)
        assert all(math.isfinite(row["local_density_mid"]) for row in rows)
        if "_spinless_tv_" in case_id:
            assert all(0.0 <= row["local_density_mid"] <= 1.0 for row in rows)
        if "_hubbard_" in case_id:
            assert all(math.isfinite(row["double_occupancy_mid"]) for row in rows)
            assert all(math.isfinite(row["local_sz_mid"]) for row in rows)
            assert all(0.0 <= row["local_density_mid"] <= 2.0 for row in rows)
            assert all(0.0 <= row["double_occupancy_mid"] <= 1.0 for row in rows)
            assert all(abs(row["local_sz_mid"]) <= 0.5 for row in rows)
    assert "fullstack_heisenberg_ad_vs_ed_N4" in case_ids
    assert "fullstack_heisenberg_dmrg_vs_ed_N4" in case_ids
    assert "fullstack_heisenberg_ad_vs_dmrg_N4" in case_ids
    assert "fullstack_tfi_ad_vs_ed_N4" in case_ids
    assert "fullstack_tfi_dmrg_vs_ed_N4" in case_ids
    assert "fullstack_tfi_ad_vs_dmrg_N4" in case_ids
    assert "fullstack_spinless_tv_ad_vs_ed_N4" in case_ids
    assert "fullstack_spinless_tv_dmrg_vs_ed_N4" in case_ids
    assert "fullstack_spinless_tv_ad_vs_dmrg_N4" in case_ids
    assert "fullstack_hubbard_ad_hard_sector_vs_ed_N2" in case_ids
    assert "fullstack_hubbard_dmrg_vs_ed_N2" in case_ids
    assert "fullstack_hubbard_ad_vs_dmrg_N2" in case_ids
    assert "trend_heisenberg_bethe_finite_obc" in case_ids
    assert "trend_tfi_transverse_magnetization" in case_ids
    assert "trend_spinless_free_fermion_limit" in case_ids
    assert "trend_spinless_free_fermion_large_n_observables" in case_ids
    assert "trend_hubbard_double_occupancy_large_u" in case_ids
    assert "trend_hubbard_free_fermion_large_n_observables" in case_ids
    spinless_large_n = by_id["trend_spinless_free_fermion_large_n_observables"]["observables"]
    assert [row["N"] for row in spinless_large_n["rows"]] == [40, 80]
    assert spinless_large_n["energy_distance_to_thermodynamic"][1] < spinless_large_n["energy_distance_to_thermodynamic"][0]
    assert all(err < 0.02 for err in spinless_large_n["density_mid_abs_error_from_half_filling"])
    assert all(corr > 0.0 for corr in spinless_large_n["connected_density_midbond_abs"])
    hubbard_large_n = by_id["trend_hubbard_free_fermion_large_n_observables"]["observables"]
    assert [row["N"] for row in hubbard_large_n["rows"]] == [40, 80]
    assert hubbard_large_n["energy_distance_to_thermodynamic"][1] < hubbard_large_n["energy_distance_to_thermodynamic"][0]
    assert all(err < 0.02 for err in hubbard_large_n["density_mid_abs_error_from_half_filling"])
    assert all(err < 0.02 for err in hubbard_large_n["double_occupancy_mid_abs_error_from_quarter"])
    assert all(corr > 0.0 for corr in hubbard_large_n["connected_density_midbond_abs"])

    report_text = report.read_text(encoding="utf-8")
    assert "Hamiltonian audit" in report_text
    assert "Ground-state audit" in report_text
    assert "Sector audit" in report_text
    assert "Observable audit" in report_text
    assert "Literature audit" in report_text
    assert "Policy audit" in report_text
    assert "Large-N AD audit" in report_text
    assert "OVERALL STATUS: PASS" in report_text
    large_n_payload = json.loads(large_n_json.read_text(encoding="utf-8"))
    assert large_n_payload["status"] == "REVIEW REQUIRED"
    evidence_ids = {record["case_id"] for record in large_n_payload["evidence_records"]}
    assert "large_n_chi_table_heisenberg_N20_chi4_8_16_32_no_reference" in evidence_ids
    assert "large_n_ad_heisenberg_N40_chi8_no_reference" in evidence_ids
    assert "large_n_chi_table_heisenberg_N40_chi4_8_16_32_no_reference" in evidence_ids
    assert "large_n_ad_heisenberg_N80_chi8_no_reference" in evidence_ids
    assert "large_n_chi_table_heisenberg_N80_chi4_8_16_32_no_reference" in evidence_ids
    assert "large_n_chi_table_spinless_tv_N20_chi4_8_16_32_hard_sector_no_reference" in evidence_ids
    assert "large_n_chi_table_hubbard_N10_chi4_8_16_32_hard_sector_no_reference" in evidence_ids
    assert "large_n_ad_spinless_tv_N40_chi8_hard_sector_no_reference" in evidence_ids
    assert "large_n_ad_hubbard_N20_chi8_hard_sector_no_reference" in evidence_ids
    assert "large_n_chi_table_spinless_tv_N40_chi4_8_16_32_hard_sector_no_reference" in evidence_ids
    assert "large_n_chi_table_hubbard_N20_chi4_8_16_32_hard_sector_no_reference" in evidence_ids
    assert "large_n_ad_spinless_tv_N80_chi8_hard_sector_no_reference" in evidence_ids
    assert "large_n_ad_hubbard_N40_chi8_hard_sector_no_reference" in evidence_ids
    assert "large_n_chi_table_spinless_tv_N80_chi4_8_16_32_hard_sector_no_reference" in evidence_ids
    assert "large_n_chi_table_hubbard_N40_chi4_8_16_32_hard_sector_no_reference" in evidence_ids
    assert "trend_spinless_free_fermion_large_n_observables" in evidence_ids
    assert "trend_hubbard_free_fermion_large_n_observables" in evidence_ids
    review_items = {item["item"] for item in large_n_payload["review_required"]}
    assert "Heisenberg N=40/80 interacting AD chi-convergence" in review_items
    assert "Hubbard N=20/40 interacting AD chi-convergence" in review_items
    assert "interacting large-N observable/correlation literature trends" in review_items
    large_n_text = large_n_md.read_text(encoding="utf-8")
    assert "Large-N Evidence Audit" in large_n_text
    assert "REVIEW REQUIRED" in large_n_text


def test_reference_registry_contains_stage11_metadata():
    registry = json.loads(open("benchmarks/references/reference_registry.json", encoding="utf-8").read())
    ids = {entry["id"] for entry in registry}
    assert "heisenberg_bethe_energy" in ids
    assert "spinless_free_fermion_open_chain" in ids
    assert "hubbard_free_fermion_limit" in ids
    for entry in registry:
        assert entry["model"]
        assert entry["observable"]
        assert entry["reference_type"]
        assert entry["citation"]

