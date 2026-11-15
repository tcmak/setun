"""Minimal LiteX CSR peripheral stub for SimSoC bring-up (Task 0.4)."""

from __future__ import annotations

from litex.soc.interconnect.csr import AutoCSR, CSRStatus, CSRStorage
from migen import Module

SETUN_STUB_ID = 0x53455455  # ASCII "SETU"
SETUN_STUB_VERSION = 0x00010000  # v1.0.0


class AccelStub(Module, AutoCSR):
    """Minimal CSR peripheral stub with ID, VERSION, SCRATCH, and COUNTER CSRs."""

    def __init__(
        self,
        device_id: int = SETUN_STUB_ID,
        version: int = SETUN_STUB_VERSION,
    ) -> None:
        self.id = CSRStatus(32, reset=device_id, name="id", description="Setun magic ID")
        self.version = CSRStatus(
            32,
            reset=version,
            name="version",
            description="Stub peripheral version",
        )
        self.scratch = CSRStorage(
            32,
            reset=0,
            name="scratch",
            description="Read/write 32-bit scratch register",
        )
        self.counter = CSRStatus(
            32,
            reset=0,
            name="counter",
            description="Free-running sys_clk cycle counter",
        )

        # Increment free-running counter on every clock tick
        self.sync += self.counter.status.eq(self.counter.status + 1)
