"""Shards keep blocks of a module together and balance collected item counts."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import conftest
import pytest


def _items():
    return [SimpleNamespace(path=Path(f"test_{module}.py"), nodeid=f"{module}::{index}",
                            keywords={})
            for module, size in zip("abcdefghijk", [21, 19, 17, 13, 11, 9, 7, 5, 3, 2, 1],
                                    strict=True)
            for index in range(size)]


def _collect(monkeypatch, shard, items):
    monkeypatch.setenv("YAP_STUDIO_TEST_SHARD", shard)
    config = Mock()
    selected = items.copy()
    conftest.pytest_collection_modifyitems(config=config, items=selected)
    rejected = config.hook.pytest_deselected.call_args.kwargs["items"]
    assert {item.nodeid for item in selected + rejected} == {item.nodeid for item in items}
    assert not {item.nodeid for item in selected} & {item.nodeid for item in rejected}
    return selected


def test_blocks_land_in_exactly_one_balanced_stable_shard(monkeypatch):
    items = _items()
    owners = {}
    sizes = []
    for shard in range(1, 6):
        selected = _collect(monkeypatch, f"{shard}/5", items)
        reversed_selection = _collect(monkeypatch, f"{shard}/5", list(reversed(items)))
        assert {item.nodeid for item in selected} == {
            item.nodeid for item in reversed_selection
        }
        sizes.append(len(selected))
        for module in {item.path for item in selected}:
            assert module not in owners
            owners[module] = shard
            assert sum(item.path == module for item in selected) == sum(
                item.path == module for item in items
            )
    assert set(owners) == {item.path for item in items}
    assert sum(sizes) == len(items)
    assert max(sizes) - min(sizes) <= 2
    assert owners[Path("test_a.py")] == 1


def test_big_module_splits_into_blocks_across_shards(monkeypatch):
    big = [SimpleNamespace(path=Path("test_big.py"), nodeid=f"big::{index:03d}", keywords={})
           for index in range(conftest._SHARD_CHUNK * 2 + 5)]
    shards = [{item.nodeid for item in _collect(monkeypatch, f"{shard}/3", big)}
              for shard in (1, 2, 3)]
    assert sorted(len(shard) for shard in shards) == [5, conftest._SHARD_CHUNK,
                                                      conftest._SHARD_CHUNK]
    ordered = [item.nodeid for item in big]
    for shard in shards:
        positions = sorted(ordered.index(nodeid) for nodeid in shard)
        assert positions == list(range(positions[0], positions[0] + len(positions)))


def test_equal_modules_break_ties_by_path(monkeypatch):
    items = [SimpleNamespace(path=Path(f"test_{name}.py"), nodeid=name, keywords={})
             for name in "dcba"]
    assert [item.nodeid for item in _collect(monkeypatch, "1/2", items)] == ["c", "a"]


def test_unset_shard_keeps_every_item(monkeypatch):
    monkeypatch.delenv("YAP_STUDIO_TEST_SHARD", raising=False)
    items = _items()
    selected = items.copy()
    config = Mock()
    conftest.pytest_collection_modifyitems(config=config, items=selected)
    assert selected == items
    config.hook.pytest_deselected.assert_not_called()


def test_nested_pytest_does_not_inherit_the_outer_shard():
    assert "YAP_STUDIO_TEST_SHARD" not in os.environ


@pytest.mark.parametrize("value", ["", "bad", "1", "1/2/3", "1.0/2", "0/5", "6/5",
                                       "1/0", "-1/5", "1/-5", " 1/5", "1/5\n"])
def test_bad_shard_is_usage_error(monkeypatch, value):
    monkeypatch.setenv("YAP_STUDIO_TEST_SHARD", value)
    with pytest.raises(pytest.UsageError, match="YAP_STUDIO_TEST_SHARD"):
        conftest.pytest_collection_modifyitems(config=Mock(), items=[])


def test_test_timeout_names_the_slow_test(monkeypatch) -> None:
    import signal
    import time
    from types import SimpleNamespace

    import conftest

    monkeypatch.setenv("YAP_STUDIO_TEST_TIMEOUT", "1")
    calls = conftest.pytest_runtest_call(SimpleNamespace(nodeid="tests/x.py::test_slow"))
    next(calls)
    with pytest.raises(TimeoutError, match=r"tests/x.py::test_slow ran longer than 1 s"):
        time.sleep(3)
    calls.close()
    assert signal.alarm(0) == 0
