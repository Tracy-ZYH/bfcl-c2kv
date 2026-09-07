#!/usr/bin/env python3
"""Resolve the training/serving contract attached to a C2KV checkpoint.

The profile is deliberately stricter than ``config.json``.  Hugging Face model
config records the gist modules, but it does not record the history packing,
training ratios, tool placement, or (for older forks) which projection handles
the query after gist KV.  Those fields must come from a profile written by the
trainer, real legacy run artifacts, or an explicitly selected reference
profile.  Missing fields are never filled from current benchmark defaults.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as _datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


PROFILE_FILENAME = "c2kv_checkpoint_profile.json"
SCHEMA_VERSION = 1
REFERENCE_PROFILES = ("checkpoint-1088",)
_SECRET_KEYS = {
    "token", "password", "secret", "api_key", "apikey", "access_token",
    "auth_token", "hub_token", "push_to_hub_token",
}


class ProfileError(ValueError):
    """A checkpoint's execution contract is missing or contradictory."""


def _json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProfileError(f"expected a JSON object in {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_bytes(payload.encode("utf-8"))


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
    raise ProfileError(f"{field} must be a boolean, got {value!r}")


def _as_positive_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"{field} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ProfileError(f"{field} must be positive, got {parsed}")
    return parsed


def _as_ratios(value: Any, field: str = "compression_ratios") -> list[int]:
    if isinstance(value, str):
        items: Iterable[Any] = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = value
    elif value is None:
        return []
    else:
        items = [value]
    ratios: list[int] = []
    for item in items:
        if isinstance(item, str) and not item.strip():
            continue
        ratio = _as_positive_int(item, field)
        if ratio not in ratios:
            ratios.append(ratio)
    return ratios


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            lowered = name.lower()
            if lowered in _SECRET_KEYS or lowered.endswith(("_password", "_secret", "_api_key", "_access_token")):
                result[name] = "<redacted>" if item not in (None, "") else item
            else:
                result[name] = _jsonable(item)
        return result
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    return str(value)


def _redacted_argv(argv: Sequence[str]) -> list[str]:
    result: list[str] = []
    redact_next = False
    for argument in argv:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        flag, separator, value = argument.partition("=")
        key = flag.lstrip("-").replace("-", "_").lower()
        sensitive = key in _SECRET_KEYS or key.endswith(("_password", "_secret", "_api_key", "_access_token"))
        if flag.startswith("--") and sensitive:
            result.append(flag + "=<redacted>" if separator else flag)
            redact_next = not bool(separator)
        else:
            result.append(argument)
    return result


def _git_identity(repo_root: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.SubprocessError):
        return {"missing": ["git_commit", "git_dirty"]}
    identity: Dict[str, Any] = {
        "git_commit": commit,
        "git_dirty": bool(status),
    }
    if status:
        try:
            tracked = subprocess.run(
                ["git", "-C", str(repo_root), "diff", "--binary", "HEAD"],
                check=True,
                capture_output=True,
                timeout=15,
            ).stdout
            identity["git_tracked_diff_sha256"] = _sha256_bytes(tracked)
            identity["git_status_sha256"] = _sha256_bytes(status.encode("utf-8"))
            identity["untracked_paths"] = [
                line[3:] for line in status.splitlines() if line.startswith("?? ")
            ]
        except (OSError, subprocess.SubprocessError):
            identity["missing"] = ["git_tracked_diff_sha256"]
    return identity


def _artifact(path: Path, label: str) -> Dict[str, Any]:
    return {
        "kind": label,
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
    }


def _find_parent(checkpoint: Path, name: str) -> Optional[Path]:
    current = checkpoint if checkpoint.is_dir() else checkpoint.parent
    # Profiles belong beside a run/checkpoint, never at an arbitrary filesystem
    # ancestor.  A bound prevents a stray repo- or home-level artifact from
    # being attached to an unrelated checkpoint.
    for parent in (current, *list(current.parents)[:3]):
        candidate = parent / name
        if candidate.is_file():
            return candidate
    return None


def _pick_consistent(
    field: str,
    candidates: Sequence[tuple[Any, str]],
    *,
    normalize=lambda value: value,
) -> tuple[Any, list[str]]:
    present = [(normalize(value), source) for value, source in candidates if value is not None]
    if not present:
        return None, []
    first = present[0][0]
    conflicts = [(value, source) for value, source in present[1:] if value != first]
    if conflicts:
        details = ", ".join(f"{source}={value!r}" for value, source in present)
        raise ProfileError(f"conflicting {field}: {details}")
    return first, [source for _value, source in present]


def _serving_compatibility(training: Mapping[str, Any], gist_param: Any) -> tuple[bool, str]:
    if training.get("doc_mode") != "history_only":
        return False, "SGLang history serving requires a history_only checkpoint"
    if training.get("tools_in_system") is not True:
        return False, "SGLang history serving presents tools in the system prefix"
    if str(gist_param or "") != "qkv":
        return False, "the pinned SGLang server supports the lowercase qkv profile only"
    return True, "training and serving both use tools-in-system plus compressed turn history"


def _validate_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise ProfileError(
            f"unsupported checkpoint profile schema {profile.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    training = profile.get("training")
    serving = profile.get("serving")
    model = profile.get("model")
    if not isinstance(training, Mapping) or not isinstance(serving, Mapping) or not isinstance(model, Mapping):
        raise ProfileError("checkpoint profile requires model, training, and serving objects")
    if training.get("doc_mode") not in {"joint", "tool_only", "history_only", "alternate", None}:
        raise ProfileError(f"unsupported training.doc_mode {training.get('doc_mode')!r}")
    if training.get("tools_in_system") is not None:
        _as_bool(training["tools_in_system"], "training.tools_in_system")
    if serving.get("query_projection") not in {"base", "gist", None}:
        raise ProfileError(f"unsupported serving.query_projection {serving.get('query_projection')!r}")
    if serving.get("doc_packing") not in {"turn", "message", None}:
        raise ProfileError(f"unsupported serving.doc_packing {serving.get('doc_packing')!r}")
    for key in ("max_doc_length", "max_doc_num"):
        if serving.get(key) is not None:
            _as_positive_int(serving[key], f"serving.{key}")
    _as_ratios(training.get("compression_ratios"))


def build_training_profile(
    *,
    output_dir: str | Path,
    repo_root: str | Path,
    model_args: Any,
    training_args: Any,
    data_args: Any,
    model_config: Any,
    argv: Optional[Sequence[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Build the self-description written beside newly trained checkpoints."""
    model_values = _jsonable(model_args)
    train_values = _jsonable(
        training_args.to_dict() if hasattr(training_args, "to_dict") else training_args
    )
    data_values = _jsonable(data_args)
    config_values = _jsonable(
        model_config.to_dict() if hasattr(model_config, "to_dict") else model_config
    )
    env = os.environ if environ is None else environ
    configured_ratios = env.get("C2KV_GIST_TRAIN_RATIOS")
    # models.gist_utils._sample_dynamic_gist_ratio uses this fallback when
    # unset/empty. Record that supported training regime instead of failing
    # a previously valid launch just because profile writing was enabled.
    gist_type = str(config_values.get("gist_type") or model_values.get("gist_type") or "")
    if gist_type.startswith(("interleave-", "anchor-")):
        ratios = [_as_positive_int(gist_type.split("-", 1)[1], "gist_type ratio")]
        sampling_ratios = ratios[:]
        ratio_source = "model.config.gist_type fixed ratio; dynamic environment is unused"
    elif gist_type == "dynamic-interleave":
        ratios = _as_ratios(configured_ratios) or [2, 4, 8]
        sampling_ratios = ([int(item.strip()) for item in configured_ratios.split(",")
                           if item.strip()] if configured_ratios else []) or [2, 4, 8]
        ratio_source = ("C2KV_GIST_TRAIN_RATIOS (duplicates preserve sampling weights)"
                        if configured_ratios else "models.gist_utils._sample_dynamic_gist_ratio default")
    else:
        ratios, sampling_ratios = [], []
        ratio_source = "no global interleave ratio for this gist_type"
    gist_param = config_values.get("gist_param") or model_values.get("gist_param")
    query_projection = "gist" if "q" in str(gist_param or "").lower() else "base"
    training = {
        "doc_mode": data_values.get("doc_mode"),
        "tools_in_system": _as_bool(data_values.get("tools_in_system"), "tools_in_system"),
        "doc_packing": "turn",
        "max_doc_length": _as_positive_int(data_values.get("max_doc_length"), "max_doc_length"),
        "max_doc_num": _as_positive_int(data_values.get("max_doc_num"), "max_doc_num"),
        "compression_ratios": ratios,
        "compression_ratio_sampling": sampling_ratios,
        "history_selection": data_values.get("history_selection"),
        "hybrid_tail_choices": data_values.get("hybrid_tail_choices"),
        "resolved_args": {
            "model": model_values,
            "training": train_values,
            "data": data_values,
        },
    }
    compatible, reason = _serving_compatibility(training, gist_param)
    manifest_path = Path(output_dir) / "train_manifest_used.json"
    artifacts = [_artifact(manifest_path, "train_manifest")] if manifest_path.is_file() else []
    initial_model = model_values.get("model_name_or_path")
    if initial_model:
        base_config = Path(str(initial_model)).expanduser() / "config.json"
        if base_config.is_file():
            artifacts.append(_artifact(base_config, "initial_model_config"))
    for field, label in (
        ("split_manifest_file", "split_manifest"),
        ("example_order_file", "example_order"),
    ):
        value = data_values.get(field)
        if value and Path(str(value)).expanduser().is_file():
            artifacts.append(_artifact(Path(str(value)).expanduser(), label))
    profile: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "profile_kind": "as_trained",
        "created_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "model": {
            "initial_model": initial_model,
            "gist_param": gist_param,
            "gist_type": config_values.get("gist_type"),
            "gist_overlap": config_values.get("gist_overlap"),
            "gist_residual_type": config_values.get("gist_residual_type"),
        },
        "training": training,
        "serving": {
            "compatible": compatible,
            "compatibility_reason": reason,
            "query_projection": query_projection,
            "doc_packing": "turn",
            "max_doc_length": training["max_doc_length"],
            "max_doc_num": training["max_doc_num"],
            "compression_ratios": ratios,
        },
        "evaluation_surfaces": {
            "training_dev": {
                "purpose": "checkpoint selection in the training dialect",
                "compatible": True,
            },
            "serving_e2e": {
                "purpose": "OpenAI-compatible SGLang benchmark evaluation",
                "compatible": compatible,
                "reason": reason,
            },
        },
        "provenance": {
            "claim": "resolved by the training process; applies to checkpoints under output_dir",
            "output_dir": str(Path(output_dir).resolve()),
            "command_line": _redacted_argv(list(sys.argv if argv is None else argv)),
            "source": _git_identity(Path(repo_root)),
            "artifacts": artifacts,
            "field_sources": {
                "training": "resolved HfArgumentParser dataclasses and runtime environment",
                "training.compression_ratio_sampling": ratio_source,
                "serving.query_projection": "training source contract: use_gist selects lowercase gist q",
            },
        },
        # Large model/data inputs are not hashed by the trainer.  Record that
        # gap rather than turning an on-disk path into an identity claim.
        "missing": ["initial_model_weights_identity", "dataset_content_identity"],
    }
    _validate_profile(profile)
    unsigned = copy.deepcopy(profile)
    profile["profile_fingerprint"] = _canonical_sha256(unsigned)
    return profile


def write_training_profile(**kwargs: Any) -> Path:
    output_dir = Path(kwargs["output_dir"])
    profile = build_training_profile(**kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PROFILE_FILENAME
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _reference_1088(checkpoint: Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    if checkpoint.name != "checkpoint-1088":
        raise ProfileError(
            "reference profile checkpoint-1088 may only be attached to a path named checkpoint-1088"
        )
    if config.get("gist_param") != "qkv":
        raise ProfileError(
            "checkpoint-1088 reference expects config.json gist_param='qkv'"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_kind": "reference",
        "reference_name": "checkpoint-1088",
        "model": {
            "gist_param": config.get("gist_param"),
            "gist_type": config.get("gist_type"),
            "gist_overlap": config.get("gist_overlap"),
            "gist_residual_type": config.get("gist_residual_type"),
        },
        "training": {
            "doc_mode": "history_only",
            "tools_in_system": True,
            "doc_packing": "turn",
            "max_doc_length": 512,
            "max_doc_num": 12,
            "compression_ratios": [4, 8, 16],
        },
        "serving": {
            "compatible": True,
            "compatibility_reason": "explicit checkpoint-1088 reference serving contract",
            "query_projection": "base",
            "doc_packing": "turn",
            "max_doc_length": 512,
            "max_doc_num": 12,
            "compression_ratios": [4, 8, 16],
        },
        "evaluation_surfaces": {
            "training_dev": {
                "purpose": "historical local checkpoint evaluation",
                "compatible": True,
            },
            "serving_e2e": {
                "purpose": "reference integration smoke and benchmark evaluation",
                "compatible": True,
            },
        },
        "provenance": {
            "claim": "explicit reference profile; reconstructed/inferred, not an as-trained record",
            "field_sources": {
                "serving.query_projection": "reference reconstruction for original lowercase qkv semantics",
                "training.doc_geometry": "reference reconstruction; individual source args were not archived",
                "training.compression_ratios": "reference launcher defaults; not archived as resolved run state",
            },
        },
        "missing": [
            "exact_training_git_commit",
            "training_git_dirty",
            "resolved_training_doc_geometry",
            "resolved_training_compression_ratios",
        ],
    }


def _legacy_profile(
    checkpoint: Path,
    config: Mapping[str, Any],
    run_config_path: Optional[Path],
    manifest_path: Optional[Path],
) -> Dict[str, Any]:
    if run_config_path is None and manifest_path is None:
        raise ProfileError(
            f"no {PROFILE_FILENAME}, run_config.json, or train_manifest_used.json found above {checkpoint}; "
            "select --reference-profile checkpoint-1088 only for that reference checkpoint"
        )
    run_config = _json(run_config_path) if run_config_path else {}
    manifest = _json(manifest_path) if manifest_path else {}
    doc_mode, doc_sources = _pick_consistent(
        "doc_mode",
        [(run_config.get("doc_mode"), "run_config.json"), (manifest.get("doc_mode"), "train_manifest_used.json")],
    )
    tools, tool_sources = _pick_consistent(
        "tools_in_system",
        [(run_config.get("tools_in_system"), "run_config.json"), (manifest.get("tools_in_system"), "train_manifest_used.json")],
        normalize=lambda value: _as_bool(value, "tools_in_system"),
    )
    max_doc_length = (
        _as_positive_int(run_config["max_doc_length"], "max_doc_length")
        if run_config.get("max_doc_length") is not None else None
    )
    max_doc_num = (
        _as_positive_int(run_config["max_doc_num"], "max_doc_num")
        if run_config.get("max_doc_num") is not None else None
    )
    ratios = _as_ratios(run_config.get("ratios"))
    missing = [
        name
        for name, value in (
            ("training.doc_mode", doc_mode),
            ("training.tools_in_system", tools),
            ("training.max_doc_length", max_doc_length),
            ("training.max_doc_num", max_doc_num),
            ("training.compression_ratios", ratios or None),
            ("serving.query_projection", None),
        )
        if value is None
    ]
    training = {
        "doc_mode": doc_mode,
        "tools_in_system": tools,
        "doc_packing": "turn" if doc_mode is not None else None,
        "max_doc_length": max_doc_length,
        "max_doc_num": max_doc_num,
        "compression_ratios": ratios,
        "history_selection": None,
        "hybrid_tail_choices": run_config.get("hybrid_tail_choices", manifest.get("hybrid_tail_choices")),
    }
    compatible, reason = _serving_compatibility(training, config.get("gist_param"))
    artifacts = []
    if run_config_path:
        artifacts.append(_artifact(run_config_path, "run_config"))
    if manifest_path:
        artifacts.append(_artifact(manifest_path, "train_manifest"))
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_kind": "legacy_artifacts",
        "model": {
            "gist_param": config.get("gist_param"),
            "gist_type": config.get("gist_type"),
            "gist_overlap": config.get("gist_overlap"),
            "gist_residual_type": config.get("gist_residual_type"),
        },
        "training": training,
        "serving": {
            "compatible": compatible,
            "compatibility_reason": reason,
            "query_projection": None,
            "doc_packing": training["doc_packing"],
            "max_doc_length": max_doc_length,
            "max_doc_num": max_doc_num,
            "compression_ratios": ratios,
        },
        "evaluation_surfaces": {
            "training_dev": {"purpose": "evaluation in the archived training dialect", "compatible": True},
            "serving_e2e": {"purpose": "SGLang benchmark evaluation", "compatible": compatible, "reason": reason},
        },
        "provenance": {
            "claim": "reconstructed from archived run artifacts; exact training source identity is unknown",
            "artifacts": artifacts,
            "field_sources": {
                "training.doc_mode": doc_sources,
                "training.tools_in_system": tool_sources,
                "training.doc_geometry": "run_config.json" if run_config_path else None,
                "training.compression_ratios": "run_config.json" if run_config_path else None,
            },
        },
        "missing": missing + ["exact_training_git_commit", "training_git_dirty"],
    }


def _apply_query_projection(profile: Dict[str, Any], query_projection: Optional[str]) -> None:
    recorded = profile["serving"].get("query_projection")
    if query_projection is None:
        if recorded is None:
            raise ProfileError(
                "query projection is absent from legacy artifacts; pass --query-projection base|gist "
                "as an explicit operator declaration"
            )
        return
    if query_projection not in {"base", "gist"}:
        raise ProfileError(f"query projection must be base or gist, got {query_projection!r}")
    if recorded is not None and recorded != query_projection:
        raise ProfileError(
            f"query projection override {query_projection!r} conflicts with profile value {recorded!r}"
        )
    if recorded is None:
        profile["serving"]["query_projection"] = query_projection
        profile["provenance"].setdefault("field_sources", {})[
            "serving.query_projection"
        ] = "explicit operator override; legacy artifacts did not record this field"
        profile["missing"] = [
            field for field in profile.get("missing", []) if field != "serving.query_projection"
        ]


def resolve_checkpoint_profile(
    checkpoint: str | Path,
    *,
    profile_path: str | Path | None = None,
    run_config_path: str | Path | None = None,
    train_manifest_path: str | Path | None = None,
    reference_profile: Optional[str] = None,
    query_projection: Optional[str] = None,
    require_serving_e2e: bool = False,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    config_path = checkpoint_path / "config.json"
    if not config_path.is_file():
        raise ProfileError(f"checkpoint config.json not found: {config_path}")
    config = _json(config_path)
    explicit_modes = sum(bool(item) for item in (profile_path, reference_profile))
    if explicit_modes > 1:
        raise ProfileError("--profile and --reference-profile are mutually exclusive")

    discovered_profile = (
        Path(profile_path).resolve()
        if profile_path
        else (None if reference_profile else _find_parent(checkpoint_path, PROFILE_FILENAME))
    )
    loaded_resolved = False
    if discovered_profile is not None:
        profile = _json(discovered_profile)
        loaded_resolved = isinstance(profile.get("checkpoint"), Mapping)
        if loaded_resolved:
            embedded_path = Path(str(profile["checkpoint"].get("path", ""))).resolve()
            if embedded_path != checkpoint_path:
                raise ProfileError(
                    f"resolved profile checkpoint {embedded_path} does not match {checkpoint_path}"
                )
            if profile["checkpoint"].get("config_sha256") != _file_sha256(config_path):
                raise ProfileError("checkpoint config.json changed since this profile was resolved")
            recorded_fingerprint = profile.get("profile_fingerprint")
            unsigned_loaded = copy.deepcopy(profile)
            unsigned_loaded.pop("profile_fingerprint", None)
            actual_fingerprint = _canonical_sha256(unsigned_loaded)
            if recorded_fingerprint != actual_fingerprint:
                raise ProfileError(
                    f"resolved profile fingerprint mismatch in {discovered_profile}"
                )
        else:
            profile.setdefault("provenance", {}).setdefault("artifacts", []).append(
                _artifact(discovered_profile, "checkpoint_profile")
            )
    elif reference_profile:
        if reference_profile != "checkpoint-1088":
            raise ProfileError(f"unknown reference profile {reference_profile!r}")
        profile = _reference_1088(checkpoint_path, config)
    else:
        run_path = Path(run_config_path).resolve() if run_config_path else _find_parent(checkpoint_path, "run_config.json")
        manifest_path = (
            Path(train_manifest_path).resolve()
            if train_manifest_path else _find_parent(checkpoint_path, "train_manifest_used.json")
        )
        profile = _legacy_profile(checkpoint_path, config, run_path, manifest_path)

    for key in ("gist_param", "gist_type", "gist_overlap", "gist_residual_type"):
        recorded = profile.get("model", {}).get(key)
        if recorded is not None and recorded != config.get(key):
            raise ProfileError(f"checkpoint {key}={config.get(key)!r} conflicts with profile {recorded!r}")
    _apply_query_projection(profile, query_projection)
    profile["checkpoint"] = {
        "path": str(checkpoint_path),
        "name": checkpoint_path.name,
        "config_sha256": _file_sha256(config_path),
    }
    _validate_profile(profile)
    required = {
        "training.doc_mode": profile["training"].get("doc_mode"),
        "training.tools_in_system": profile["training"].get("tools_in_system"),
        "training.compression_ratios": profile["training"].get("compression_ratios"),
        "serving.query_projection": profile["serving"].get("query_projection"),
        "serving.doc_packing": profile["serving"].get("doc_packing"),
        "serving.max_doc_length": profile["serving"].get("max_doc_length"),
        "serving.max_doc_num": profile["serving"].get("max_doc_num"),
    }
    absent = [name for name, value in required.items() if value is None or value == []]
    if absent:
        raise ProfileError("checkpoint profile lacks required execution fields: " + ", ".join(absent))
    if require_serving_e2e and not profile["serving"].get("compatible"):
        raise ProfileError(profile["serving"].get("compatibility_reason") or "profile is not serving-compatible")
    if not loaded_resolved:
        unsigned = copy.deepcopy(profile)
        unsigned.pop("profile_fingerprint", None)
        profile["profile_fingerprint"] = _canonical_sha256(unsigned)
    return profile


def profile_run_args(profile: Mapping[str, Any]) -> list[str]:
    """Arguments that make ``benchmarks/run.py`` match the checkpoint grid."""
    serving = profile["serving"]
    return [
        "--doc-packing", str(serving["doc_packing"]),
        "--max-doc-length", str(serving["max_doc_length"]),
        "--max-doc-num", str(serving["max_doc_num"]),
    ]


def write_resolved_profile(profile: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def write_shell_exports(profile: Mapping[str, Any], path: str | Path) -> Path:
    serving = profile["serving"]
    training = profile["training"]
    values = {
        "C2KV_PROFILE_FINGERPRINT": profile["profile_fingerprint"],
        "C2KV_PROFILE_KIND": profile["profile_kind"],
        "C2KV_QUERY_PROJ": serving["query_projection"],
        "C2KV_DOC_PACKING": serving["doc_packing"],
        "C2KV_MAX_DOC_LENGTH": serving["max_doc_length"],
        "C2KV_MAX_DOC_NUM": serving["max_doc_num"],
        "C2KV_TRAIN_DOC_MODE": training["doc_mode"],
        "C2KV_TOOLS_IN_SYSTEM": str(training["tools_in_system"]).lower(),
        "C2KV_TRAIN_RATIOS": ",".join(str(item) for item in training["compression_ratios"]),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(f"export {key}={shlex.quote(str(value))}\n" for key, value in values.items()),
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--reference-profile", choices=REFERENCE_PROFILES)
    parser.add_argument("--query-projection", choices=("base", "gist"))
    parser.add_argument("--require-serving-e2e", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shell-out", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        profile = resolve_checkpoint_profile(
            args.checkpoint,
            profile_path=args.profile,
            run_config_path=args.run_config,
            train_manifest_path=args.train_manifest,
            reference_profile=args.reference_profile,
            query_projection=args.query_projection,
            require_serving_e2e=args.require_serving_e2e,
        )
    except ProfileError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    write_resolved_profile(profile, args.out)
    if args.shell_out:
        write_shell_exports(profile, args.shell_out)
    print(json.dumps({
        "profile": str(args.out.resolve()),
        "profile_kind": profile["profile_kind"],
        "profile_fingerprint": profile["profile_fingerprint"],
        "serving": profile["serving"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
