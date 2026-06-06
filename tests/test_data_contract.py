"""Tests for DataContract — field validation and batch validation."""

import pytest

from genesys.contract.data_contract import ContractField, DataContract, ContractViolationError


@pytest.fixture()
def contract():
    return DataContract(
        model_name="test_model",
        version="1.0.0",
        fields=[
            ContractField("x1", float, required=True, min_value=0.0, max_value=100.0),
            ContractField("x2", float, required=True, min_value=0.0, max_value=100.0),
            ContractField("label", str, required=False),
        ],
    )


# ---------------------------------------------------------------------------
# Single-record validation
# ---------------------------------------------------------------------------

class TestValidateSingleRecord:
    def test_valid_record_returns_no_violations(self, contract):
        assert contract.validate({"x1": 10.0, "x2": 20.0}) == []

    def test_optional_field_can_be_absent(self, contract):
        assert contract.validate({"x1": 1.0, "x2": 2.0}) == []

    def test_missing_required_field_is_violation(self, contract):
        violations = contract.validate({"x1": 1.0})
        assert any("x2" in v for v in violations)

    def test_wrong_type_is_violation(self, contract):
        violations = contract.validate({"x1": "not_a_float", "x2": 5.0})
        assert any("x1" in v for v in violations)

    def test_below_min_is_violation(self, contract):
        violations = contract.validate({"x1": -1.0, "x2": 5.0})
        assert any("below minimum" in v for v in violations)

    def test_above_max_is_violation(self, contract):
        violations = contract.validate({"x1": 200.0, "x2": 5.0})
        assert any("above maximum" in v for v in violations)

    def test_exactly_at_boundary_is_valid(self, contract):
        assert contract.validate({"x1": 0.0, "x2": 100.0}) == []

    def test_multiple_violations_in_one_record(self, contract):
        violations = contract.validate({"x1": -1.0, "x2": 200.0})
        assert len(violations) >= 2


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

class TestValidateBatch:
    def test_all_valid_returns_empty_dict(self, contract):
        records = [{"x1": float(i), "x2": float(i)} for i in range(5)]
        assert contract.validate_batch(records) == {}

    def test_invalid_rows_reported_by_index(self, contract):
        records = [
            {"x1": 1.0, "x2": 2.0},          # row 0 — valid
            {"x1": -999.0, "x2": 2.0},        # row 1 — invalid
            {"x1": 3.0, "x2": 4.0},           # row 2 — valid
        ]
        result = contract.validate_batch(records)
        assert 1 in result
        assert 0 not in result
        assert 2 not in result

    def test_primary_key_uniqueness(self):
        pk_contract = DataContract(
            model_name="pk_model",
            fields=[
                ContractField("id", int, required=True),
                ContractField("val", float, required=True),
            ],
            primary_key=["id"],
        )
        records = [
            {"id": 1, "val": 1.0},
            {"id": 1, "val": 2.0},  # duplicate key
            {"id": 2, "val": 3.0},
        ]
        result = pk_contract.validate_batch(records)
        # At least one of the duplicate rows should be flagged
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# LinearRegression contract (real model contract)
# ---------------------------------------------------------------------------

class TestLinearRegressionContract:
    def test_valid_record_passes(self):
        from sample.ml_model.ml_sample_model1 import LinearRegression
        contract = LinearRegression.contract
        violations = contract.validate({"x1": 5.0, "x2": 5.0, "x3": 5.0, "x4": 5.0})
        assert violations == []

    def test_missing_field_fails(self):
        from sample.ml_model.ml_sample_model1 import LinearRegression
        contract = LinearRegression.contract
        violations = contract.validate({"x1": 5.0, "x2": 5.0, "x3": 5.0})
        assert any("x4" in v for v in violations)
