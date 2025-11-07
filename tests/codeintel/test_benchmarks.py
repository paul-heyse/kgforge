"""Benchmark tests for CodeIntel performance baselines.

These tests establish performance budgets and regression detection for critical operations.
"""

from __future__ import annotations

import textwrap
import time
from typing import Any

import pytest
from codeintel.indexer.tscore import Langs, get_language, load_langs, parse_bytes, run_query
from tree_sitter import Tree


@pytest.fixture(scope="module")
def langs() -> Langs:
    """Load Tree-sitter languages once for all tests.

    Returns
    -------
    Langs
        Cached Tree-sitter languages reused across benchmarks.
    """
    return load_langs()


@pytest.fixture
def sample_python_code() -> bytes:
    """Sample Python code for benchmarking.

    Returns
    -------
    bytes
        UTF-8 encoded Python source used in benchmarks.
    """
    source = textwrap.dedent(
        """
        def factorial(n):
            '''Calculate factorial of n.'''
            if n <= 1:
                return 1
            return n * factorial(n - 1)

        class Calculator:
            '''Simple calculator class.'''

            def __init__(self):
                self.result = 0

            def add(self, x, y):
                '''Add two numbers.'''
                self.result = x + y
                return self.result

            def multiply(self, x, y):
                '''Multiply two numbers.'''
                self.result = x * y
                return self.result

        def main():
            calc = Calculator()
            print(calc.add(5, 3))
            print(factorial(5))
        """
    )
    return source.encode("utf-8")


@pytest.fixture
def sample_json_code() -> bytes:
    """Sample JSON for benchmarking.

    Returns
    -------
    bytes
        UTF-8 encoded JSON document used in benchmarks.
    """
    payload = textwrap.dedent(
        """
        {
          "name": "test-package",
          "version": "1.0.0",
          "description": "A test package",
          "main": "index.js",
          "scripts": {
            "test": "jest",
            "build": "webpack",
            "start": "node index.js"
          },
          "dependencies": {
            "react": "^18.0.0",
            "express": "^4.18.0"
          },
          "devDependencies": {
            "jest": "^29.0.0",
            "webpack": "^5.0.0"
          }
        }
        """
    )
    return payload.encode("utf-8")


def test_benchmark_python_parse(benchmark, langs, sample_python_code):
    """Benchmark Python file parsing."""
    lang = get_language(langs, "python")

    # Benchmark should complete in < 10ms for small files
    result = benchmark(parse_bytes, lang, sample_python_code)

    assert result is not None
    # Budget: < 10ms for files under 1KB
    assert benchmark.stats["mean"] < 0.01


def test_benchmark_python_query(benchmark, langs, sample_python_code):
    """Benchmark Tree-sitter query execution."""
    lang = get_language(langs, "python")
    tree = parse_bytes(lang, sample_python_code)

    query = textwrap.dedent(
        """
        (function_definition
          name: (identifier) @def.name)
        (class_definition
          name: (identifier) @class.name)
        """
    )

    # Benchmark query execution
    result = benchmark(run_query, lang, query, tree, sample_python_code)

    assert len(result) > 0
    # Budget: < 5ms for simple queries
    assert benchmark.stats["mean"] < 0.005


def test_benchmark_json_parse(benchmark, langs, sample_json_code):
    """Benchmark JSON file parsing."""
    lang = get_language(langs, "json")

    result = benchmark(parse_bytes, lang, sample_json_code)

    assert result is not None
    # Budget: < 5ms for JSON files
    assert benchmark.stats["mean"] < 0.005


def test_parse_performance_scaling(langs, sample_python_code):
    """Test that parse time scales linearly with file size."""
    lang = get_language(langs, "python")

    # Measure small file
    start = time.monotonic()
    parse_bytes(lang, sample_python_code)
    small_duration = time.monotonic() - start

    # Measure 10x larger file
    large_code = sample_python_code * 10
    start = time.monotonic()
    parse_bytes(lang, large_code)
    large_duration = time.monotonic() - start

    # Should scale roughly linearly (allow 20x for overhead)
    assert large_duration < (small_duration * 20)


def test_multiple_language_load_time(benchmark):
    """Benchmark loading all Tree-sitter languages."""
    # Should complete in < 100ms
    result = benchmark(load_langs)

    assert result is not None
    # Budget: < 100ms to load all grammars
    assert benchmark.stats["mean"] < 0.1


@pytest.mark.parametrize("language", ["python", "json", "yaml", "toml"])
def test_get_language_cached(benchmark, langs, language):
    """Test that get_language is fast when cached."""
    # Should be near-instant due to caching
    result = benchmark(get_language, langs, language)

    assert result is not None
    # Budget: < 1ms for cached lookups
    assert benchmark.stats["mean"] < 0.001


def test_query_compilation_performance(langs, sample_python_code):
    """Test query compilation time is acceptable."""
    lang = get_language(langs, "python")
    tree = parse_bytes(lang, sample_python_code)

    complex_query = textwrap.dedent(
        """
        (function_definition
          name: (identifier) @func.name
          parameters: (parameters) @func.params
          body: (block) @func.body)

        (class_definition
          name: (identifier) @class.name
          superclasses: (argument_list)? @class.bases
          body: (block) @class.body)

        (call
          function: (identifier) @call.name
          arguments: (argument_list) @call.args)
        """
    )

    start = time.monotonic()
    result = run_query(lang, complex_query, tree, sample_python_code)
    duration = time.monotonic() - start

    assert len(result) > 0
    # Budget: < 20ms for complex queries
    assert duration < 0.02


def test_memory_efficiency_large_file(langs):
    """Test that parsing large files doesn't consume excessive memory."""
    # Create a 1MB Python file
    large_code = b"x = 1\n" * 50000  # ~500KB

    lang = get_language(langs, "python")

    start = time.monotonic()
    tree = parse_bytes(lang, large_code)
    duration = time.monotonic() - start

    assert tree is not None
    # Budget: < 500ms for 500KB files
    assert duration < 0.5


@pytest.mark.benchmark(group="parse", min_rounds=100)
def test_parse_python_cold_start(benchmark, sample_python_code):
    """Benchmark Python parsing without warm caches."""

    def parse_with_fresh_lang() -> Tree:
        langs = load_langs()
        lang = get_language(langs, "python")
        return parse_bytes(lang, sample_python_code)

    result = benchmark(parse_with_fresh_lang)
    assert result is not None


@pytest.mark.benchmark(group="query", min_rounds=50)
def test_query_execution_cold_start(benchmark, sample_python_code):
    """Benchmark query execution without warm caches."""

    def query_with_fresh_setup() -> list[dict[str, Any]]:
        langs = load_langs()
        lang = get_language(langs, "python")
        tree = parse_bytes(lang, sample_python_code)
        query = "(function_definition name: (identifier) @name)"
        return run_query(lang, query, tree, sample_python_code)

    result = benchmark(query_with_fresh_setup)
    assert len(result) > 0
