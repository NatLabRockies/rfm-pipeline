"""P0-S06: Frozen-config provenance stamp precedes test evaluation (F2).

Tests:
- freeze_then_evaluate succeeds when stamp matches config.
- evaluate_without_freeze raises FrozenConfigRequiredError.
- config drift after freeze is detected (ConfigDriftError).
- FrozenConfigStamp records a hash and a timestamp.
"""

from __future__ import annotations

import pytest

from rfm_pipeline.artifacts import (
    ConfigDriftError,
    FrozenConfigRequiredError,
    FrozenConfigStamp,
    freeze_config,
    require_frozen_stamp,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def resolved_config() -> dict:
    """Minimal synthetic resolved-config object."""
    return {
        "model": "rfm",
        "n_permutations": 500,
        "threshold": 0.05,
        "seed": 42,
        "features": ["x1", "x2", "x3"],
        "outputs": ["y1"],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFreezeConfigStamp:
    def test_P0_S06_stamp_has_hash_and_timestamp(self, resolved_config):
        """freeze_config returns a FrozenConfigStamp with non-empty fields."""
        stamp = freeze_config(resolved_config)
        assert isinstance(stamp, FrozenConfigStamp)
        assert len(stamp.config_hash) == 64  # SHA-256 hex
        assert stamp.frozen_at  # non-empty ISO timestamp

    def test_P0_S06_same_config_same_hash(self, resolved_config):
        """Identical configs produce identical hashes."""
        s1 = freeze_config(resolved_config)
        s2 = freeze_config(resolved_config)
        assert s1.config_hash == s2.config_hash

    def test_P0_S06_different_config_different_hash(self, resolved_config):
        """Mutated config produces a different hash."""
        s1 = freeze_config(resolved_config)
        mutated = {**resolved_config, "threshold": 0.99}
        s2 = freeze_config(mutated)
        assert s1.config_hash != s2.config_hash


class TestFreezeEvaluateSucceeds:
    def test_P0_S06_freeze_then_evaluate_passes(self, resolved_config):
        """Calling require_frozen_stamp with a matching stamp does not raise."""
        stamp = freeze_config(resolved_config)
        # Should not raise
        require_frozen_stamp(resolved_config, stamp)

    def test_P0_S06_stamp_frozen_at_is_iso_utc(self, resolved_config):
        """frozen_at parses as an ISO-8601 UTC datetime."""
        import datetime

        stamp = freeze_config(resolved_config)
        dt = datetime.datetime.fromisoformat(stamp.frozen_at)
        assert dt.tzinfo is not None  # timezone-aware


class TestEvaluateWithoutFreezeRaises:
    def test_P0_S06_no_stamp_raises(self, resolved_config):
        """require_frozen_stamp with stamp=None raises FrozenConfigRequiredError."""
        with pytest.raises(FrozenConfigRequiredError):
            require_frozen_stamp(resolved_config, None)

    def test_P0_S06_error_message_is_informative(self, resolved_config):
        """The error message mentions freeze_config."""
        with pytest.raises(FrozenConfigRequiredError, match="freeze_config"):
            require_frozen_stamp(resolved_config, None)


class TestConfigDriftDetection:
    def test_P0_S06_drifted_config_raises(self, resolved_config):
        """Passing a mutated config with an old stamp raises ConfigDriftError."""
        stamp = freeze_config(resolved_config)
        drifted = {**resolved_config, "threshold": 0.99}
        with pytest.raises(ConfigDriftError):
            require_frozen_stamp(drifted, stamp)

    def test_P0_S06_drift_error_shows_hashes(self, resolved_config):
        """ConfigDriftError message includes both the current and frozen hash."""
        stamp = freeze_config(resolved_config)
        drifted = {**resolved_config, "seed": 999}
        with pytest.raises(ConfigDriftError, match="drift"):
            require_frozen_stamp(drifted, stamp)

    def test_P0_S06_undrifted_config_passes_after_refreeze(self, resolved_config):
        """After an intentional change, re-freezing produces a valid new stamp."""
        stamp_old = freeze_config(resolved_config)
        drifted = {**resolved_config, "threshold": 0.01}

        # Old stamp fails on drifted config
        with pytest.raises(ConfigDriftError):
            require_frozen_stamp(drifted, stamp_old)

        # Re-freeze allows evaluation
        stamp_new = freeze_config(drifted)
        require_frozen_stamp(drifted, stamp_new)  # should not raise
