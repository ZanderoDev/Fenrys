import pytest

from fenrys.ipc import ExecutorClient, ExecutorError


def _client(executor):
    return ExecutorClient(executor.sock)


def test_write_returns_unified_diff(executor):
    client = _client(executor)
    target = executor.home / "a.txt"
    target.write_text("one\ntwo\n")
    result = client.execute("file.write", {"path": str(target), "content": "one\nTWO\n"}, mode="coding")
    diff = result["result"]["diff"]
    assert "-two" in diff
    assert "+TWO" in diff
    assert result["result"]["added"] == 1
    assert result["result"]["removed"] == 1
    assert result["rollback_available"] is True


def test_create_diff_has_no_removals(executor):
    client = _client(executor)
    target = executor.home / "new.txt"
    result = client.execute("file.write", {"path": str(target), "content": "hello\n"}, mode="coding")
    assert result["result"]["created"] is True
    assert result["result"]["removed"] == 0
    assert "+++ b/" in result["result"]["diff"]


def test_edit_replaces_and_diffs(executor):
    client = _client(executor)
    target = executor.home / "edit.txt"
    target.write_text("alpha\nbeta\n")
    result = client.execute(
        "file.edit",
        {"path": str(target), "edits": [{"search": "beta", "replace": "BETA"}]},
        mode="coding",
    )
    assert result["result"]["changed"] is True
    assert result["result"]["edits_applied"] == 1
    assert "+BETA" in result["result"]["diff"]
    assert target.read_text() == "alpha\nBETA\n"


def test_edit_is_whitespace_tolerant(executor):
    client = _client(executor)
    target = executor.home / "ws.txt"
    target.write_text("def f():\n    return 1   \n")
    result = client.execute(
        "file.edit",
        {"path": str(target), "edits": [{"search": "    return 1", "replace": "    return 2"}]},
        mode="coding",
    )
    assert result["result"]["changed"] is True
    assert "return 2" in target.read_text()


def test_edit_fails_closed_when_text_missing(executor):
    client = _client(executor)
    target = executor.home / "miss.txt"
    target.write_text("content\n")
    with pytest.raises(ExecutorError) as excinfo:
        client.execute(
            "file.edit",
            {"path": str(target), "edits": [{"search": "nonexistent", "replace": "x"}]},
            mode="coding",
        )
    assert excinfo.value.code == "action_failed"
    assert target.read_text() == "content\n"


def test_delete_returns_diff_to_dev_null(executor):
    client = _client(executor)
    target = executor.home / "gone.txt"
    target.write_text("bye\n")
    result = client.execute("file.delete", {"path": str(target)}, mode="coding")
    assert "+++ /dev/null" in result["result"]["diff"]
    assert "-bye" in result["result"]["diff"]
    assert not target.exists()


def test_delete_of_system_path_is_gated(executor):
    client = _client(executor)
    with pytest.raises(ExecutorError) as excinfo:
        client.execute("file.delete", {"path": "/etc/hostname"}, mode="coding")
    assert excinfo.value.code == "approval_required"


def test_diff_between_two_files(executor):
    client = _client(executor)
    left = executor.home / "l.txt"
    right = executor.home / "r.txt"
    left.write_text("a\nb\n")
    right.write_text("a\nc\n")
    result = client.execute("file.diff", {"path": str(left), "other_path": str(right)}, mode="coding")
    assert "+c" in result["result"]["diff"]
    assert "-b" in result["result"]["diff"]


def test_changes_are_logged(executor):
    client = _client(executor)
    target = executor.home / "logged.txt"
    target.write_text("one\n")
    client.execute("file.edit", {"path": str(target), "edits": [{"search": "one", "replace": "two"}]}, mode="coding")
    changes = client.list_changes()
    assert changes
    assert changes[-1]["path"] == str(target)
    assert changes[-1]["added"] == 1
    assert changes[-1]["removed"] == 1


def test_edit_ambiguous_exact_match_is_rejected(executor):
    client = _client(executor)
    target = executor.home / "ambig.txt"
    target.write_text("return 1\nreturn 1\n")
    with pytest.raises(ExecutorError) as excinfo:
        client.execute(
            "file.edit",
            {"path": str(target), "edits": [{"search": "return 1", "replace": "return 2"}]},
            mode="coding",
        )
    assert excinfo.value.code == "action_failed"
    assert "lines 1, 2" in str(excinfo.value)
    assert target.read_text() == "return 1\nreturn 1\n"


def test_edit_ambiguous_fuzzy_match_is_rejected(executor):
    client = _client(executor)
    target = executor.home / "ambig_ws.txt"
    target.write_text("def f():\n    return 1  \ndef g():\n    return 1  \n")
    with pytest.raises(ExecutorError):
        client.execute(
            "file.edit",
            {"path": str(target), "edits": [{"search": "    return 1", "replace": "    return 2"}]},
            mode="coding",
        )
    assert target.read_text() == "def f():\n    return 1  \ndef g():\n    return 1  \n"


def test_edit_ambiguous_batch_is_atomic(executor):
    client = _client(executor)
    target = executor.home / "batch.txt"
    target.write_text("one\ntwo\ntwo\n")
    with pytest.raises(ExecutorError):
        client.execute(
            "file.edit",
            {
                "path": str(target),
                "edits": [
                    {"search": "one", "replace": "ONE"},
                    {"search": "two", "replace": "TWO"},
                ],
            },
            mode="coding",
        )
    assert target.read_text() == "one\ntwo\ntwo\n"


def test_edit_single_match_still_applies(executor):
    client = _client(executor)
    target = executor.home / "single.txt"
    target.write_text("alpha\nbeta\ngamma\n")
    result = client.execute(
        "file.edit",
        {"path": str(target), "edits": [{"search": "beta", "replace": "BETA"}]},
        mode="coding",
    )
    assert result["result"]["changed"] is True
    assert target.read_text() == "alpha\nBETA\ngamma\n"


def test_edit_replace_all_still_replaces_everywhere(executor):
    client = _client(executor)
    target = executor.home / "all.txt"
    target.write_text("x\nx\nx\n")
    result = client.execute(
        "file.edit",
        {"path": str(target), "edits": [{"search": "x", "replace": "y", "replace_all": True}]},
        mode="coding",
    )
    assert target.read_text() == "y\ny\ny\n"
