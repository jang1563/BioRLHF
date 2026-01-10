"""Tests for ground truth data module."""

import pytest
from biorlhf.data.ground_truth import (
    STRESSOR_EFFECTS,
    KMP_EFFECTS,
    INTERACTIONS,
    TISSUE_TYPES,
    OXPHOS_PATTERNS,
)


class TestStressorEffects:
    """Tests for STRESSOR_EFFECTS data."""

    def test_all_tissues_present(self):
        """Verify all four tissues are in the dataset."""
        expected_tissues = {"Heart", "Hippocampus", "Liver", "Soleus"}
        assert set(STRESSOR_EFFECTS.keys()) == expected_tissues

    def test_all_conditions_present(self):
        """Verify all stressor conditions are present for each tissue."""
        expected_conditions = {"HU", "IR", "HU_IR"}
        for tissue, effects in STRESSOR_EFFECTS.items():
            assert set(effects.keys()) == expected_conditions, f"Missing conditions for {tissue}"

    def test_deg_counts_are_positive(self):
        """Verify all DEG counts are non-negative integers."""
        for tissue, effects in STRESSOR_EFFECTS.items():
            for condition, count in effects.items():
                assert isinstance(count, int), f"DEG count for {tissue}/{condition} should be int"
                assert count >= 0, f"DEG count for {tissue}/{condition} should be non-negative"

    def test_known_values(self):
        """Verify specific known values from the experimental data."""
        # Soleus is most HU-sensitive
        assert STRESSOR_EFFECTS["Soleus"]["HU"] == 6425
        # Hippocampus is most IR-sensitive
        assert STRESSOR_EFFECTS["Hippocampus"]["IR"] == 5477
        # Heart has minimal HU response
        assert STRESSOR_EFFECTS["Heart"]["HU"] == 165


class TestKMPEffects:
    """Tests for KMP_EFFECTS data."""

    def test_all_tissues_present(self):
        """Verify all four tissues are in the dataset."""
        expected_tissues = {"Heart", "Hippocampus", "Liver", "Soleus"}
        assert set(KMP_EFFECTS.keys()) == expected_tissues

    def test_all_conditions_present(self):
        """Verify all KMP conditions are present for each tissue."""
        expected_conditions = {"baseline", "in_HU", "in_IR", "in_HU_IR"}
        for tissue, effects in KMP_EFFECTS.items():
            assert set(effects.keys()) == expected_conditions, f"Missing conditions for {tissue}"

    def test_stress_activated_patterns(self):
        """Verify stress-activated tissues show increased response under stress."""
        # Heart should show stress-activated pattern
        assert KMP_EFFECTS["Heart"]["in_HU_IR"] > KMP_EFFECTS["Heart"]["baseline"]
        # Soleus should show stress-activated pattern
        assert KMP_EFFECTS["Soleus"]["in_HU_IR"] > KMP_EFFECTS["Soleus"]["baseline"]

    def test_stress_blocked_patterns(self):
        """Verify stress-blocked tissues show decreased response under stress."""
        # Hippocampus should show stress-blocked pattern
        assert KMP_EFFECTS["Hippocampus"]["in_HU_IR"] < KMP_EFFECTS["Hippocampus"]["baseline"]


class TestInteractions:
    """Tests for INTERACTIONS data."""

    def test_all_tissues_present(self):
        """Verify all four tissues are in the dataset."""
        expected_tissues = {"Heart", "Hippocampus", "Liver", "Soleus"}
        assert set(INTERACTIONS.keys()) == expected_tissues

    def test_all_interaction_types_present(self):
        """Verify all interaction types are present for each tissue."""
        expected_interactions = {"HU_x_IR", "KMP_x_HU", "KMP_x_IR"}
        for tissue, effects in INTERACTIONS.items():
            assert set(effects.keys()) == expected_interactions, f"Missing interactions for {tissue}"

    def test_soleus_kmp_hu_interaction(self):
        """Verify the notable KMP x HU interaction in soleus."""
        # This is the largest interaction effect
        assert INTERACTIONS["Soleus"]["KMP_x_HU"] == 8484


class TestTissueTypes:
    """Tests for TISSUE_TYPES classification."""

    def test_all_tissues_classified(self):
        """Verify all tissues have a type classification."""
        expected_tissues = {"Heart", "Hippocampus", "Liver", "Soleus"}
        assert set(TISSUE_TYPES.keys()) == expected_tissues

    def test_type_classifications(self):
        """Verify correct tissue type classifications."""
        assert "Type A" in TISSUE_TYPES["Heart"]
        assert "Type A" in TISSUE_TYPES["Soleus"]
        assert "Type B" in TISSUE_TYPES["Hippocampus"]
        assert "Type C" in TISSUE_TYPES["Liver"]


class TestOXPHOSPatterns:
    """Tests for OXPHOS_PATTERNS data."""

    def test_all_tissues_present(self):
        """Verify all tissues have OXPHOS data."""
        expected_tissues = {"Heart", "Hippocampus", "Liver", "Soleus"}
        assert set(OXPHOS_PATTERNS.keys()) == expected_tissues

    def test_pattern_fields_present(self):
        """Verify all expected fields are present."""
        expected_fields = {"stress_NES", "KMP_NES", "pattern"}
        for tissue, data in OXPHOS_PATTERNS.items():
            assert set(data.keys()) == expected_fields, f"Missing fields for {tissue}"

    def test_rescue_patterns(self):
        """Verify tissues with RESCUE pattern."""
        assert OXPHOS_PATTERNS["Heart"]["pattern"] == "RESCUE"
        assert OXPHOS_PATTERNS["Soleus"]["pattern"] == "RESCUE"

    def test_suppression_pattern(self):
        """Verify liver has SUPPRESSION pattern."""
        assert OXPHOS_PATTERNS["Liver"]["pattern"] == "SUPPRESSION"

    def test_nes_values_numeric(self):
        """Verify NES values are numeric."""
        for tissue, data in OXPHOS_PATTERNS.items():
            assert isinstance(data["stress_NES"], (int, float))
            assert isinstance(data["KMP_NES"], (int, float))
