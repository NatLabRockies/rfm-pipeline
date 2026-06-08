"""Tests for hpc_cascade helpers (used by both rfm and bsm orchestrators)."""

from __future__ import annotations

from rfm_pipeline.hpc_cascade import inject_dependency_flag, parse_reduce_job_id


class TestParseReduceJobId:
    def test_single_marker(self) -> None:
        assert parse_reduce_job_id("foo\nRFM_HPC_SUBMIT_REDUCE_JOB_ID=12345\nbar") == 12345

    def test_returns_last_when_multiple(self) -> None:
        # When rfm-hpc-submit prints the marker more than once (e.g. sparse
        # re-submit), the last marker reflects the actual reduce job.
        stdout = (
            "RFM_HPC_SUBMIT_REDUCE_JOB_ID=100\n"
            "intermediate output\n"
            "RFM_HPC_SUBMIT_REDUCE_JOB_ID=200\n"
        )
        assert parse_reduce_job_id(stdout) == 200

    def test_no_marker(self) -> None:
        assert parse_reduce_job_id("no marker here\n") is None

    def test_malformed_marker_returns_none_not_overwrites(self) -> None:
        # Malformed line is skipped, not silently treated as an int.
        assert parse_reduce_job_id("RFM_HPC_SUBMIT_REDUCE_JOB_ID=not_an_int\n") is None

    def test_malformed_then_good(self) -> None:
        stdout = "RFM_HPC_SUBMIT_REDUCE_JOB_ID=not_an_int\nRFM_HPC_SUBMIT_REDUCE_JOB_ID=42\n"
        assert parse_reduce_job_id(stdout) == 42

    def test_marker_with_whitespace(self) -> None:
        assert parse_reduce_job_id("  RFM_HPC_SUBMIT_REDUCE_JOB_ID=7  \n") == 7

    def test_empty(self) -> None:
        assert parse_reduce_job_id("") is None


class TestInjectDependencyFlag:
    def test_appends_flag(self) -> None:
        cmd = "pixi run rfm-hpc-submit --config foo.yml --stage interaction_discovery"
        out = inject_dependency_flag(cmd, 42)
        assert out.endswith("--depends-on-job-id 42")
        assert cmd in out

    def test_idempotent_when_already_present(self) -> None:
        cmd = "pixi run rfm-hpc-submit --depends-on-job-id 99 --stage foo"
        assert inject_dependency_flag(cmd, 42) == cmd

    def test_passthrough_for_non_submit_command(self) -> None:
        cmd = "pixi run python tools/run_manuscript_pipeline.py cfg.yml"
        assert inject_dependency_flag(cmd, 42) == cmd
