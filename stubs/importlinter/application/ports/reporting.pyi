from __future__ import annotations

from typing import Dict, Iterator, List, Tuple


class InvalidContractOptions(Exception):
    errors: Dict[str, str]


class Contract:
    name: str
    contract_options: Dict[str, object]


class ContractCheck:
    kept: bool
    metadata: Dict[str, object]
    warnings: List[str]

    def __init__(
        self,
        kept: bool,
        metadata: Dict[str, object] | None = ...,
        warnings: List[str] | None = ...,
    ) -> None: ...


class Report:
    contracts: List[Contract]
    broken_count: int
    kept_count: int
    warnings_count: int
    module_count: int
    import_count: int
    graph_building_duration: int
    contains_failures: bool
    could_not_run: bool
    invalid_contract_options: Dict[str, InvalidContractOptions]

    def add_contract_check(
        self, contract: Contract, contract_check: ContractCheck, duration: int
    ) -> None: ...

    def add_invalid_contract_options(
        self, contract_name: str, exception: InvalidContractOptions
    ) -> None: ...

    def get_contracts_and_checks(self) -> Iterator[Tuple[Contract, ContractCheck]]: ...

    def get_duration(self, contract: Contract) -> int: ...

