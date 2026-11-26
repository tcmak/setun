"""Per-op tensor trace container (STRC v1) and first-divergence debugger (Task 3.5)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from setun.ref.format import _fnv1a_64

TRACE_MAGIC = b"STRC"
TRACE_VERSION = 1


def _canonical_le_array(tensor: np.ndarray) -> tuple[np.ndarray, str]:
    arr = np.asarray(tensor)
    dt = np.dtype(arr.dtype)
    le_dt = dt.newbyteorder("<") if dt.itemsize > 1 else dt
    contig = np.ascontiguousarray(arr, dtype=le_dt)
    return contig, le_dt.str


@dataclass(frozen=True)
class TraceEntry:
    """Single traced operation output with shape, dtype, and 64-bit FNV-1a digest."""

    seq: int
    step: int
    op_id: str
    op_name: str
    shape: tuple[int, ...]
    dtype: str
    fnv1a64: str
    tensor: np.ndarray


@dataclass(frozen=True)
class TraceDiffResult:
    """Structured comparison result between two trace sequences."""

    matched: bool
    reason: str
    total_compared: int
    len_a: int
    len_b: int
    seq: int | None = None
    step: int | None = None
    op_id: str | None = None
    op_name: str | None = None
    op_id_b: str | None = None
    shape_a: tuple[int, ...] | None = None
    shape_b: tuple[int, ...] | None = None
    dtype_a: str | None = None
    dtype_b: str | None = None
    hash_a: str | None = None
    hash_b: str | None = None
    first_flat_index: int | None = None
    first_multi_index: tuple[int, ...] | None = None
    val_a: float | int | None = None
    val_b: float | int | None = None
    max_abs_diff: float | None = None
    num_mismatched: int = 0
    prev_ops: tuple[str, ...] = ()
    next_ops: tuple[str, ...] = ()


class TraceRecorder:
    """Callable TraceHook recording per-step op outputs with FNV-1a-64 hashes."""

    def __init__(self, metadata: dict[str, object] | None = None) -> None:
        self.metadata: dict[str, object] = dict(metadata) if metadata is not None else {}
        self.entries: list[TraceEntry] = []
        self._step: int = 0
        self._started: bool = False

    def __call__(self, op_name: str, tensor: np.ndarray) -> None:
        if self._started and op_name == "embed":
            self._step += 1
        self._started = True

        contig, dtype_str = _canonical_le_array(tensor)
        raw = contig.tobytes()
        digest = f"{_fnv1a_64(raw):016x}"
        seq = len(self.entries)
        op_id = f"step{self._step}.{op_name}"
        self.entries.append(
            TraceEntry(
                seq=seq,
                step=self._step,
                op_id=op_id,
                op_name=op_name,
                shape=tuple(int(d) for d in contig.shape),
                dtype=dtype_str,
                fnv1a64=digest,
                tensor=contig.copy(),
            )
        )


def serialize_trace(
    entries: Sequence[TraceEntry],
    *,
    metadata: dict[str, object] | None = None,
) -> bytes:
    """Serialize TraceEntry records into a deterministic STRC v1 binary container."""
    meta_obj = dict(metadata) if metadata is not None else {}
    index_records: list[dict[str, object]] = []
    chunks: list[bytes] = []
    offset = 0

    for entry in entries:
        contig, dtype_str = _canonical_le_array(entry.tensor)
        raw = contig.tobytes()
        digest = f"{_fnv1a_64(raw):016x}"
        index_records.append(
            {
                "seq": entry.seq,
                "step": entry.step,
                "op_id": entry.op_id,
                "op_name": entry.op_name,
                "shape": list(entry.shape),
                "dtype": dtype_str,
                "fnv1a64": digest,
                "offset": offset,
                "nbytes": len(raw),
            }
        )
        chunks.append(raw)
        offset += len(raw)

    header_bytes = json.dumps(
        {"version": TRACE_VERSION, "metadata": meta_obj, "entries": index_records},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return TRACE_MAGIC + len(header_bytes).to_bytes(4, "little") + header_bytes + b"".join(chunks)


def deserialize_trace(blob: bytes) -> tuple[list[TraceEntry], dict[str, object]]:
    """Validate and decode an STRC v1 trace buffer, verifying every FNV-1a-64 hash."""
    if len(blob) < 8:
        raise ValueError("[setun.runtime.trace] STRC trace buffer is truncated (< 8 bytes).")
    if blob[:4] != TRACE_MAGIC:
        raise ValueError(
            f"[setun.runtime.trace] Invalid trace magic {blob[:4]!r}; expected b'STRC'."
        )

    header_len = int.from_bytes(blob[4:8], "little")
    if 8 + header_len > len(blob):
        raise ValueError("[setun.runtime.trace] STRC header length exceeds buffer size.")

    header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    if header.get("version") != TRACE_VERSION:
        raise ValueError(
            f"[setun.runtime.trace] Unsupported STRC version {header.get('version')!r}; "
            f"expected {TRACE_VERSION}."
        )

    payload = memoryview(blob)[8 + header_len :]
    entries: list[TraceEntry] = []
    for rec in header["entries"]:
        offset = int(rec["offset"])
        nbytes = int(rec["nbytes"])
        if offset < 0 or (offset + nbytes) > len(payload):
            raise ValueError(
                f"[setun.runtime.trace] STRC tensor payload out of bounds for {rec['op_id']!r}."
            )
        raw = bytes(payload[offset : offset + nbytes])
        actual_hash = f"{_fnv1a_64(raw):016x}"
        expected_hash = str(rec["fnv1a64"])
        if actual_hash != expected_hash:
            raise ValueError(
                f"[setun.runtime.trace] STRC hash mismatch on {rec['op_id']!r}: "
                f"expected {expected_hash}, got {actual_hash}."
            )
        shape = tuple(int(d) for d in rec["shape"])
        dtype_str = str(rec["dtype"])
        arr = np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape).copy()
        entries.append(
            TraceEntry(
                seq=int(rec["seq"]),
                step=int(rec["step"]),
                op_id=str(rec["op_id"]),
                op_name=str(rec["op_name"]),
                shape=shape,
                dtype=dtype_str,
                fnv1a64=expected_hash,
                tensor=arr,
            )
        )

    return entries, dict(header.get("metadata", {}))


def save_trace(
    path: Path | str,
    entries: Sequence[TraceEntry],
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    """Write TraceEntry records to `path` in STRC v1 format."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(serialize_trace(entries, metadata=metadata))


def load_trace(source: Path | str | bytes) -> tuple[list[TraceEntry], dict[str, object]]:
    """Load an STRC v1 trace from a file path or raw bytes."""
    blob = source if isinstance(source, bytes) else Path(source).read_bytes()
    return deserialize_trace(blob)


def diff_traces(
    entries_a: Sequence[TraceEntry],
    entries_b: Sequence[TraceEntry],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
    context: int = 3,
) -> TraceDiffResult:
    """Locate the first structural or element-wise divergence between two trace sequences."""
    n_common = min(len(entries_a), len(entries_b))

    for idx in range(n_common):
        ea = entries_a[idx]
        eb = entries_b[idx]
        prev_ops = tuple(e.op_id for e in entries_a[max(0, idx - context) : idx])
        next_ops = tuple(e.op_id for e in entries_a[idx + 1 : idx + 1 + context])

        if ea.op_id != eb.op_id:
            return TraceDiffResult(
                matched=False,
                reason="op_id_mismatch",
                total_compared=idx,
                len_a=len(entries_a),
                len_b=len(entries_b),
                seq=idx,
                step=ea.step,
                op_id=ea.op_id,
                op_name=ea.op_name,
                op_id_b=eb.op_id,
                prev_ops=prev_ops,
                next_ops=next_ops,
            )

        if ea.shape != eb.shape:
            return TraceDiffResult(
                matched=False,
                reason="shape_mismatch",
                total_compared=idx,
                len_a=len(entries_a),
                len_b=len(entries_b),
                seq=idx,
                step=ea.step,
                op_id=ea.op_id,
                op_name=ea.op_name,
                shape_a=ea.shape,
                shape_b=eb.shape,
                prev_ops=prev_ops,
                next_ops=next_ops,
            )

        if ea.dtype != eb.dtype:
            return TraceDiffResult(
                matched=False,
                reason="dtype_mismatch",
                total_compared=idx,
                len_a=len(entries_a),
                len_b=len(entries_b),
                seq=idx,
                step=ea.step,
                op_id=ea.op_id,
                op_name=ea.op_name,
                dtype_a=ea.dtype,
                dtype_b=eb.dtype,
                prev_ops=prev_ops,
                next_ops=next_ops,
            )

        arr_a, _ = _canonical_le_array(ea.tensor)
        arr_b, _ = _canonical_le_array(eb.tensor)
        if atol == 0.0 and rtol == 0.0:
            bits_dt = np.dtype(f"<u{arr_a.dtype.itemsize}")
            diff_mask = arr_a.view(bits_dt) != arr_b.view(bits_dt)
        else:
            diff_mask = ~np.isclose(arr_a, arr_b, atol=atol, rtol=rtol, equal_nan=True)

        if bool(np.any(diff_mask)):
            flat_mask = diff_mask.reshape(-1)
            first_flat = int(np.flatnonzero(flat_mask)[0])
            multi_idx = tuple(int(i) for i in np.unravel_index(first_flat, ea.shape))
            num_bad = int(np.count_nonzero(flat_mask))
            raw_a = arr_a.reshape(-1)[first_flat].item()
            raw_b = arr_b.reshape(-1)[first_flat].item()
            abs_diffs = np.abs(arr_a.astype(np.float64) - arr_b.astype(np.float64))
            max_diff = float(np.max(abs_diffs))
            return TraceDiffResult(
                matched=False,
                reason="value_mismatch",
                total_compared=idx,
                len_a=len(entries_a),
                len_b=len(entries_b),
                seq=idx,
                step=ea.step,
                op_id=ea.op_id,
                op_name=ea.op_name,
                shape_a=ea.shape,
                shape_b=eb.shape,
                dtype_a=ea.dtype,
                dtype_b=eb.dtype,
                hash_a=ea.fnv1a64,
                hash_b=eb.fnv1a64,
                first_flat_index=first_flat,
                first_multi_index=multi_idx,
                val_a=raw_a,
                val_b=raw_b,
                max_abs_diff=max_diff,
                num_mismatched=num_bad,
                prev_ops=prev_ops,
                next_ops=next_ops,
            )

    if len(entries_a) != len(entries_b):
        return TraceDiffResult(
            matched=False,
            reason="length_mismatch",
            total_compared=n_common,
            len_a=len(entries_a),
            len_b=len(entries_b),
        )

    return TraceDiffResult(
        matched=True,
        reason="ok",
        total_compared=n_common,
        len_a=len(entries_a),
        len_b=len(entries_b),
    )


def format_diff_report(res: TraceDiffResult) -> str:
    """Format a human-readable first-divergence report for CLI output."""
    if res.matched:
        return f"[setun diff] MATCH: {res.total_compared} traced ops verified identical."

    lines = [f"[setun diff] MISMATCH ({res.reason}):"]
    if res.reason == "value_mismatch":
        lines.append(
            f"  first_op : {res.op_id} (seq={res.seq}, step={res.step}, "
            f"shape={res.shape_a}, dtype={res.dtype_a})"
        )
        lines.append(f"  hashes   : A={res.hash_a} != B={res.hash_b}")
        lines.append(
            f"  location : index={res.first_multi_index} (flat={res.first_flat_index}, "
            f"mismatched_elements={res.num_mismatched})"
        )
        lines.append(
            f"  values   : A={res.val_a!r} vs B={res.val_b!r} (max_abs_diff={res.max_abs_diff:.6g})"
        )
    else:
        lines.append(
            f"  detail   : seq={res.seq} op_a={res.op_id!r} op_b={res.op_id_b!r} "
            f"len=({res.len_a}, {res.len_b}) shape=({res.shape_a}, {res.shape_b}) "
            f"dtype=({res.dtype_a}, {res.dtype_b})"
        )

    if res.prev_ops:
        lines.append(f"  prev_ops : {' -> '.join(res.prev_ops)}")
    if res.next_ops:
        lines.append(f"  next_ops : {' -> '.join(res.next_ops)}")
    return "\n".join(lines)
