"""Typed-action catalog: the only operations the executor can perform."""

from __future__ import annotations

import difflib
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fenrys_common import (
    Risk,
    Zone,
    action_fingerprint,
    classify_path,
    decide,
    is_blocked_shell,
    risk_for,
)

ALLOWED_DEV_BINARIES = frozenset(
    {
        "python",
        "python3",
        "pytest",
        "pip",
        "pip3",
        "uv",
        "node",
        "npm",
        "pnpm",
        "yarn",
        "cargo",
        "rustc",
        "go",
        "make",
        "gcc",
        "g++",
        "clang",
        "cmake",
        "ninja",
        "git",
        "tree-sitter",
    }
)

DEFAULT_TIMEOUT = 120


@dataclass
class ActionContext:
    home: str
    user_roots: tuple[str, ...]
    mode: str = "agent"
    critical_services: frozenset[str] = field(default_factory=frozenset)
    timeout: int = DEFAULT_TIMEOUT
    run_as_uid: int | None = None
    run_as_gid: int | None = None


@dataclass
class ActionSpec:
    action_type: str
    description: str
    target_of: Callable[[dict[str, Any]], str | None]
    zone_of: Callable[[dict[str, Any], ActionContext], Zone]
    handler: Callable[[dict[str, Any], ActionContext], dict[str, Any]]
    force_block: Callable[[dict[str, Any]], bool] | None = None


def _path_target(args: dict[str, Any]) -> str | None:
    return args.get("path") or args.get("cwd")


def _path_zone(args: dict[str, Any], ctx: ActionContext) -> Zone:
    target = _path_target(args)
    if not target:
        return Zone.SYSTEM_CORE
    return classify_path(target, ctx.user_roots)


def _system_zone(_args: dict[str, Any], _ctx: ActionContext) -> Zone:
    return Zone.SYSTEM_CORE


def _require_absolute(path: str) -> Path:
    expanded = os.path.expanduser(path)
    if not os.path.isabs(expanded):
        raise ValueError(f"target must be absolute: {path!r}")
    return Path(expanded)


def unified_diff(path: str, old: str, new: str | None, old_exists: bool = True) -> str:
    """Hermes-style ``a/path`` -> ``b/path`` unified diff; ``new=None`` means deleted."""
    fromfile = f"a/{path}" if old_exists else "/dev/null"
    tofile = "/dev/null" if new is None else f"b/{path}"
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            [] if new is None else new.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def diff_counts(diff: str) -> tuple[int, int]:
    added = removed = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".fenrys-tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _fuzzy_replace(text: str, search: str, replace: str, replace_all: bool) -> tuple[str, bool]:
    """Whitespace-tolerant replacement so the model need not match trailing spaces."""
    lines = text.splitlines(keepends=True)
    needle = [line.rstrip() for line in search.splitlines()]
    span = len(needle)
    matches: list[int] = []
    for index in range(len(lines) - span + 1):
        window = [line.rstrip() for line in lines[index : index + span]]
        if window == needle:
            matches.append(index)
    if not matches:
        return text, False
    if len(matches) > 1 and not replace_all:
        locations = ", ".join(str(index + 1) for index in matches)
        raise ValueError(
            f"search text matches {len(matches)} locations (lines {locations}); "
            "add more context to the search text or set replace_all"
        )
    tail = replace if replace.endswith("\n") or replace == "" else replace + "\n"
    targets = matches if replace_all else matches[:1]
    for index in reversed(targets):
        lines[index : index + span] = [tail]
    return "".join(lines), True


def _preexec(ctx: ActionContext):
    if os.getuid() != 0 or ctx.run_as_uid is None:
        return None
    uid = ctx.run_as_uid
    gid = ctx.run_as_gid
    if gid is None:
        try:
            import pwd

            gid = pwd.getpwuid(uid).pw_gid
        except KeyError:
            gid = uid

    def _drop() -> None:
        os.setgid(gid)
        try:
            os.setgroups([])
        except OSError:
            pass
        os.setuid(uid)

    return _drop


def _child_env(ctx: ActionContext) -> dict[str, str]:
    env = dict(os.environ)
    if ctx.run_as_uid is not None:
        try:
            import pwd

            env["HOME"] = pwd.getpwuid(ctx.run_as_uid).pw_dir
        except KeyError:
            pass
    return env


def _run(argv: list[str], ctx: ActionContext, cwd: str | None = None) -> dict[str, Any]:
    proc = subprocess.run(
        argv,
        cwd=os.path.expanduser(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=ctx.timeout,
        preexec_fn=_preexec(ctx),
        env=_child_env(ctx),
    )
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _handle_file_read(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    path = _require_absolute(str(args["path"]))
    max_bytes = int(args.get("max_bytes", 200_000))
    data = path.read_bytes()[:max_bytes]
    return {"path": str(path), "content": data.decode("utf-8", errors="replace"), "bytes": len(data)}


def _handle_file_list(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    path = _require_absolute(str(args["path"]))
    entries = []
    for name in sorted(os.listdir(path)):
        full = path / name
        entries.append({"name": name, "dir": full.is_dir(), "size": full.stat().st_size if full.exists() else 0})
    return {"path": str(path), "entries": entries}


def _handle_file_write(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    path = _require_absolute(str(args["path"]))
    content = str(args.get("content", ""))
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    _atomic_write(path, content)
    diff = unified_diff(str(path), previous or "", content, old_exists=previous is not None)
    added, removed = diff_counts(diff)
    return {
        "path": str(path),
        "created": previous is None,
        "bytes": len(content.encode()),
        "diff": diff,
        "added": added,
        "removed": removed,
        "previous": previous,
    }


def _handle_file_edit(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    path = _require_absolute(str(args["path"]))
    edits = args.get("edits") or []
    if not isinstance(edits, list) or not edits:
        raise ValueError("edits must be a non-empty list of {search, replace}")
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist — use fs_write to create it")
    original = current = path.read_text(encoding="utf-8")
    applied: list[int] = []
    for index, edit in enumerate(edits, start=1):
        search = str(edit.get("search", ""))
        replace = str(edit.get("replace", ""))
        if not search:
            raise ValueError(f"edit #{index}: search text is required")
        replace_all = bool(edit.get("replace_all"))
        occurrences = current.count(search)
        if occurrences > 1 and not replace_all:
            starts = []
            pos = current.find(search)
            while pos != -1:
                starts.append(current.count("\n", 0, pos) + 1)
                pos = current.find(search, pos + 1)
            locations = ", ".join(str(line) for line in starts)
            raise ValueError(
                f"edit #{index}: search text matches {occurrences} locations (lines {locations}); "
                "add more context to the search text or set replace_all"
            )
        if occurrences >= 1:
            current = current.replace(search, replace) if replace_all else current.replace(search, replace, 1)
            applied.append(index)
            continue
        try:
            current, ok = _fuzzy_replace(current, search, replace, replace_all)
        except ValueError as exc:
            raise ValueError(f"edit #{index}: {exc}") from exc
        if ok:
            applied.append(index)
            continue
        if replace and replace in current:
            applied.append(index)
            continue
        raise ValueError(f"edit #{index}: search text not found in {path} (fuzzy match also failed)")
    diff = unified_diff(str(path), original, current)
    added, removed = diff_counts(diff)
    if current != original:
        _atomic_write(path, current)
    return {
        "path": str(path),
        "changed": current != original,
        "edits_applied": len(applied),
        "diff": diff,
        "added": added,
        "removed": removed,
        "previous": original,
    }


def _handle_file_delete(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    path = _require_absolute(str(args["path"]))
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if path.is_dir():
        raise IsADirectoryError(f"{path} is a directory")
    content = path.read_text(encoding="utf-8")
    path.unlink()
    diff = unified_diff(str(path), content, None)
    added, removed = diff_counts(diff)
    return {
        "path": str(path),
        "deleted": True,
        "bytes": len(content.encode()),
        "diff": diff,
        "added": added,
        "removed": removed,
        "previous": content,
    }


def _handle_file_diff(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    left = _require_absolute(str(args["path"]))
    right = _require_absolute(str(args["other_path"]))
    if not left.exists() or not right.exists():
        raise FileNotFoundError("both paths must exist")
    old = left.read_text(encoding="utf-8")
    new = right.read_text(encoding="utf-8")
    diff = unified_diff(str(left), old, new)
    added, removed = diff_counts(diff)
    return {"path": str(left), "other_path": str(right), "diff": diff, "added": added, "removed": removed}


def _handle_service_status(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    unit = str(args["unit"])
    return _run(["systemctl", "is-active", unit], ctx)


def _handle_pkg_query(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    name = str(args["name"])
    return _run(["dpkg-query", "-W", "-f=${Status} ${Version}\n", name], ctx)


def _handle_pkg_install(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    packages = [str(p) for p in args.get("packages", [])]
    if not packages:
        raise ValueError("packages must be a non-empty list")
    return _run(["apt-get", "install", "-y", *packages], ctx)


def _handle_service_control(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    action = str(args.get("action", "restart"))
    if action not in {"start", "stop", "restart", "reload", "enable", "disable"}:
        raise ValueError(f"unsupported service action: {action!r}")
    unit = str(args["unit"])
    return _run(["systemctl", action, unit], ctx)


def _handle_dev_run(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    argv = [str(a) for a in args.get("argv", [])]
    if not argv:
        raise ValueError("argv must be a non-empty list")
    binary = os.path.basename(argv[0])
    if binary not in ALLOWED_DEV_BINARIES:
        raise ValueError(f"binary not allowed for dev.run: {binary!r}")
    return _run(argv, ctx, cwd=args.get("cwd"))


def _handle_dev_shell(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    command = str(args.get("command", ""))
    if not command:
        raise ValueError("command must be non-empty")
    return _run(["/bin/sh", "-c", command], ctx, cwd=args.get("cwd"))


def _handle_shell_exec(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    command = str(args.get("command", ""))
    if not command:
        raise ValueError("command must be non-empty")
    return _run(["/bin/sh", "-c", command], ctx, cwd=args.get("cwd"))


def _shell_blocked(args: dict[str, Any]) -> bool:
    return is_blocked_shell(str(args.get("command", "")))


REGISTRY: dict[str, ActionSpec] = {
    "file.read": ActionSpec("file.read", "Read a file", _path_target, _path_zone, _handle_file_read),
    "file.list": ActionSpec("file.list", "List a directory", _path_target, _path_zone, _handle_file_list),
    "file.write": ActionSpec("file.write", "Write a file", _path_target, _path_zone, _handle_file_write),
    "file.edit": ActionSpec("file.edit", "Edit a file in place", _path_target, _path_zone, _handle_file_edit),
    "file.delete": ActionSpec("file.delete", "Delete a file", _path_target, _path_zone, _handle_file_delete),
    "file.diff": ActionSpec(
        "file.diff",
        "Compare two files",
        lambda a: str(a.get("path", "")),
        _path_zone,
        _handle_file_diff,
    ),
    "service.status": ActionSpec("service.status", "Query a systemd unit", lambda a: str(a.get("unit", "")), _system_zone, _handle_service_status),
    "pkg.query": ActionSpec("pkg.query", "Query an installed package", lambda a: str(a.get("name", "")), _system_zone, _handle_pkg_query),
    "pkg.install": ActionSpec("pkg.install", "Install packages", lambda a: ", ".join(a.get("packages", [])), _system_zone, _handle_pkg_install),
    "service.control": ActionSpec("service.control", "Control a systemd unit", lambda a: str(a.get("unit", "")), _system_zone, _handle_service_control),
    "dev.run": ActionSpec("dev.run", "Run an allowlisted dev command in a workspace", _path_target, _path_zone, _handle_dev_run),
    "dev.shell": ActionSpec(
        "dev.shell",
        "Run a shell command as an unprivileged user inside a workspace",
        _path_target,
        _path_zone,
        _handle_dev_shell,
        _shell_blocked,
    ),
    "shell.exec": ActionSpec(
        "shell.exec",
        "Escape hatch: run a shell command (root, High)",
        _path_target,
        _system_zone,
        _handle_shell_exec,
        _shell_blocked,
    ),
}


def preview(action_type: str, args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    spec = REGISTRY.get(action_type)
    if spec is None:
        return {
            "action_type": action_type,
            "supported": False,
            "fingerprint": action_fingerprint(action_type, args, None),
            "zone": Zone.SYSTEM_CORE.value,
            "risk": Risk.CRITICAL.label,
            "decision": "deny",
            "description": "unknown action",
            "blocked": True,
        }
    target = spec.target_of(args)
    zone = spec.zone_of(args, ctx)
    blocked = bool(spec.force_block and spec.force_block(args))
    risk = risk_for(action_type, zone, args, ctx.critical_services)
    return {
        "action_type": action_type,
        "supported": True,
        "fingerprint": action_fingerprint(action_type, args, target),
        "target": target,
        "zone": zone.value,
        "risk": risk.label,
        "decision": decide(ctx.mode, risk, zone, blocked),
        "description": spec.description,
        "blocked": blocked,
    }


def execute(action_type: str, args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    preview_result = preview(action_type, args, ctx)
    if preview_result["decision"] != "auto":
        raise PermissionError(preview_result["decision"])
    spec = REGISTRY[action_type]
    result = spec.handler(args, ctx)
    return {"preview": preview_result, "result": result}


__all__ = ["ActionContext", "ActionSpec", "REGISTRY", "ALLOWED_DEV_BINARIES", "preview", "execute"]
