"""Minimal Python UART/DMA/Etherbone hardware test harness for Setun (Task 3.3a).

Strictly scoped to driving hardware tests and simulation/board transport (`AccelDriver`).
No host language compiler, freestanding runtime, or model-serving code belongs in this package.
"""

from setun.runtime.driver import (
    VALID_BACKENDS,
    AccelDriver,
    AccelHardwareError,
    AsyncAccelDriver,
    CocotbWishboneBackend,
    CSRLayout,
    EtherboneBackend,
    PerfMetrics,
    RefBusBackend,
    add_backend_argument,
    open_backend,
)

__all__ = [
    "VALID_BACKENDS",
    "AccelDriver",
    "AccelHardwareError",
    "AsyncAccelDriver",
    "CSRLayout",
    "CocotbWishboneBackend",
    "EtherboneBackend",
    "PerfMetrics",
    "RefBusBackend",
    "add_backend_argument",
    "open_backend",
    "HybridDecodeSummary",
    "HybridM0Runner",
    "open_m0_backend",
    "run_m0_greedy",
    "TraceDiffResult",
    "TraceEntry",
    "TraceRecorder",
    "deserialize_trace",
    "diff_traces",
    "format_diff_report",
    "load_trace",
    "save_trace",
    "serialize_trace",
]

from setun.runtime.m0_runner import (
    HybridDecodeSummary,
    HybridM0Runner,
    open_m0_backend,
    run_m0_greedy,
)
from setun.runtime.trace import (
    TraceDiffResult,
    TraceEntry,
    TraceRecorder,
    deserialize_trace,
    diff_traces,
    format_diff_report,
    load_trace,
    save_trace,
    serialize_trace,
)
