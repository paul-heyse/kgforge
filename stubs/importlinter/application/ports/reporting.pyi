from __future__ import annotations

from collections.abc import Iterator

class InvalidContractOptions(Exception):  # noqa: N818 - library uses Options suffix
    errors: dict[str, str]

class Contract:
    name: str
    contract_options: dict[str, object]

class ContractCheck:
    kept: bool
    metadata: dict[str, object]
    warnings: list[str]

    def __init__(
        self,
        kept: bool,
        metadata: dict[str, object] | None = ...,
        warnings: list[str] | None = ...,
    ) -> None: ...

class Report:
    contracts: list[Contract]
    broken_count: int
    kept_count: int
    warnings_count: int
    module_count: int
    import_count: int
    graph_building_duration: int
    contains_failures: bool
    could_not_run: bool
    invalid_contract_options: dict[str, InvalidContractOptions]

    def add_contract_check(
        self, contract: Contract, contract_check: ContractCheck, duration: int
    ) -> None: ...
    def add_invalid_contract_options(
        self, contract_name: str, exception: InvalidContractOptions
    ) -> None: ...
    def get_contracts_and_checks(self) -> Iterator[tuple[Contract, ContractCheck]]: ...
    def get_duration(self, contract: Contract) -> int: ...
