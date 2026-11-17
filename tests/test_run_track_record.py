"""Runner unit contour (CHECKABLE_FINDINGS.md: F-14, F-15).

Pins the v5 record: subprocess runs of the committed commands must
reproduce the committed result artifacts byte-identically (deterministic
seeds), and the partial-run semantics must hold:
  --case X          -> ONLY case X; results/track_record.csv untouched;
  --cases a,b       -> exactly a and b; CSV untouched;
  --cases all       -> full battery; CSV rewritten.
"""
import re
import subprocess
import sys
from pathlib import Path

from src.data_generator import load_or_generate
from src.metrics import block_permute_target
from src.run_track_record import REGISTRY

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
N = len(REGISTRY)


def run_runner(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.run_track_record", *args],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )


def case_ids(stdout: str) -> list[str]:
    """Case result lines look like '[defect_01_look_ahead] FAIL/DATA ...'.

    NC lines ('[NC-NULL seed=1]') and the '[partial run]' note do not match
    (they contain spaces inside the brackets).
    """
    out = []
    for line in stdout.splitlines():
        m = re.match(r"^\[([a-z0-9_]+)\]", line)
        if m:
            out.append(m.group(1))
    return out


def test_case_flag_runs_only_the_named_case():
    csv_before = (RESULTS / "track_record.csv").read_bytes()
    r = run_runner("--case", "defect_01_look_ahead")
    assert r.returncode == 0, r.stdout + r.stderr
    assert case_ids(r.stdout) == ["defect_01_look_ahead"]
    assert f"1/{N}" in r.stdout                     # exactly one of the registry
    assert "[partial run]" in r.stdout              # honest partial-run note
    assert (RESULTS / "track_record.csv").read_bytes() == csv_before


def test_cases_list_runs_only_the_named_subset():
    csv_before = (RESULTS / "track_record.csv").read_bytes()
    r = run_runner("--cases", "defect_01_look_ahead,defect_02_dst_timezone")
    assert r.returncode == 0, r.stdout + r.stderr
    assert case_ids(r.stdout) == ["defect_01_look_ahead", "defect_02_dst_timezone"]
    assert f"2/{N}" in r.stdout
    assert (RESULTS / "track_record.csv").read_bytes() == csv_before


def test_case_extends_an_explicit_cases_list():
    r = run_runner("--cases", "defect_01_look_ahead",
                   "--case", "clean_01_linear_forecast_features")
    assert r.returncode == 0, r.stdout + r.stderr
    assert case_ids(r.stdout) == ["defect_01_look_ahead",
                                  "clean_01_linear_forecast_features"]


def test_unknown_case_is_rejected():
    r = run_runner("--case", "no_such_case")
    assert r.returncode == 2
    assert "unknown cases" in r.stdout


def test_empty_selection_is_rejected():
    r = run_runner()
    assert r.returncode == 2


def test_null_control_runs_on_daily_dataset_and_reproduces():
    # F-15: NC must permute the DAILY dataset; the committed
    # results/null_control.csv must reproduce byte-identically. If the NC
    # loop ever picked the wrong dataset (v4 bug: the last processed case's
    # frame), the MAEs would not match.
    nc_before = (RESULTS / "null_control.csv").read_bytes()
    r = run_runner("--case", "defect_01_look_ahead", "--with-null-control")
    assert r.returncode == 0, r.stdout + r.stderr
    nc_lines = [l for l in r.stdout.splitlines() if l.startswith("[NC-NULL seed=")]
    assert len(nc_lines) == 5
    assert "[NC-NULL] false PASS: 0/5 (N=5)" in r.stdout
    assert (RESULTS / "null_control.csv").read_bytes() == nc_before


def test_block_permute_target_preserves_marginals_and_blocks():
    df = load_or_generate()
    perm = block_permute_target(df, block=7, seed=1)
    assert perm.index.equals(df.index)
    assert perm.columns.tolist() == df.columns.tolist()
    assert sorted(perm["load"]) == sorted(df["load"])            # marginals kept
    n = (len(df) // 7) * 7
    blocks_before = sorted(tuple(df["load"].to_numpy()[i:i + 7]) for i in range(0, n, 7))
    blocks_after = sorted(tuple(perm["load"].to_numpy()[i:i + 7]) for i in range(0, n, 7))
    assert blocks_before == blocks_after                         # blocks kept whole
    assert not (perm["load"].to_numpy() == df["load"].to_numpy()).all()  # alignment broken
