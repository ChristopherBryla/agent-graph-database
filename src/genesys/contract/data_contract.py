from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Field definition — model builders declare one per input
# ---------------------------------------------------------------------------

@dataclass
class ContractField:
    name: str
    dtype: type                        # e.g. float, int, str
    required: bool = True
    min_value: float | None = None     # numeric range check (optional)
    max_value: float | None = None
    description: str = ""


# ---------------------------------------------------------------------------
# Contract — the full set of fields a model expects
# ---------------------------------------------------------------------------

@dataclass
class DataContract:
    model_name: str
    version: str = "1.0.0"
    fields: list[ContractField] = field(default_factory=list)
    # One or more field names that together form the unique row identifier.
    # When set, validate_batch will report any duplicate key combinations.
    primary_key: list[str] = field(default_factory=list)

    def validate(self, record: dict[str, Any]) -> list[str]:
        """Validate a single data record against this contract.

        Returns a list of violation messages. Empty list means the record
        is valid. Raises nothing — callers decide how to handle violations.
        """
        violations: list[str] = []

        for f in self.fields:
            if f.name not in record:
                if f.required:
                    violations.append(f"required field '{f.name}' is missing")
                continue

            value = record[f.name]

            if not isinstance(value, f.dtype):
                violations.append(
                    f"field '{f.name}' expected {f.dtype.__name__}, "
                    f"got {type(value).__name__} (value={value!r})"
                )
                continue  # skip range check if type is wrong

            if f.min_value is not None and value < f.min_value:
                violations.append(
                    f"field '{f.name}' value {value} is below minimum {f.min_value}"
                )
            if f.max_value is not None and value > f.max_value:
                violations.append(
                    f"field '{f.name}' value {value} is above maximum {f.max_value}"
                )

        return violations

    def validate_batch(
        self, records: list[dict[str, Any]]
    ) -> dict[int, list[str]]:
        """Validate a list of records. Returns {row_index: [violations]}
        for any rows that fail. Rows that pass are not included."""
        result: dict[int, list[str]] = {
            i: v
            for i, record in enumerate(records)
            if (v := self.validate(record))
        }

        # Primary key uniqueness check — only runs when a PK is declared.
        if self.primary_key:
            seen: dict[tuple, int] = {}  # key_value -> first row index
            for i, record in enumerate(records):
                key_value = tuple(record.get(k) for k in self.primary_key)
                if key_value in seen:
                    key_repr = ", ".join(
                        f"{k}={v!r}" for k, v in zip(self.primary_key, key_value)
                    )
                    msg = (
                        f"primary key ({key_repr}) duplicates row {seen[key_value]}"
                    )
                    result.setdefault(i, []).append(msg)
                else:
                    seen[key_value] = i

        return result

    def summary(self) -> str:
        pk_set = set(self.primary_key)
        lines = [f"DataContract: {self.model_name} v{self.version}"]
        if self.primary_key:
            lines.append(f"  primary key: ({', '.join(self.primary_key)})")
        for f in self.fields:
            req = "PK" if f.name in pk_set else ("required" if f.required else "optional")
            bounds = ""
            if f.min_value is not None or f.max_value is not None:
                bounds = f" [{f.min_value} .. {f.max_value}]"
            desc = f"  # {f.description}" if f.description else ""
            lines.append(
                f"  {f.name}: {f.dtype.__name__} ({req}){bounds}{desc}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Raised when a contract check fails during pipeline execution
# ---------------------------------------------------------------------------

class ContractViolationError(Exception):
    def __init__(self, model_name: str, violations: list[str]) -> None:
        self.model_name = model_name
        self.violations = violations
        joined = "; ".join(violations)
        super().__init__(f"contract violation in '{model_name}': {joined}")
