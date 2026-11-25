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
]
