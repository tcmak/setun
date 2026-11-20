"""LiteX SimSoC target for Setun (Tasks 0.3 & 0.4)."""

from __future__ import annotations

import argparse
from pathlib import Path

from litex.build.generic_platform import Pins, Subsignal
from litex.build.sim import SimPlatform
from litex.build.sim.config import SimConfig
from litex.soc.integration.builder import Builder, builder_argdict, builder_args
from litex.soc.integration.soc_core import SoCCore, soc_core_argdict, soc_core_args
from litex.soc.interconnect import wishbone
from migen.genlib.io import CRG

from setun.accelerators.stub import AccelStub

_io = [
    ("sys_clk", 0, Pins(1)),
    ("sys_rst", 0, Pins(1)),
    ("serial", 0, Subsignal("tx", Pins(1)), Subsignal("rx", Pins(1))),
]


class SetunSimPlatform(SimPlatform):
    def __init__(self) -> None:
        super().__init__("sim", _io)


class SetunSimSoC(SoCCore):
    def __init__(
        self,
        with_accel_stub: bool = False,
        expose_host_wb: bool = False,
        use_external_crg: bool = True,
        **kwargs,
    ) -> None:
        platform = SetunSimPlatform()
        kwargs.pop("uart_name", None)
        kwargs.setdefault("csr_data_width", 32)

        super().__init__(
            platform=platform,
            clk_freq=int(1e6),
            ident="Setun Simulated SoC",
            uart_name="serial",
            **kwargs,
        )
        if use_external_crg:
            self.crg = CRG(platform.request("sys_clk"), platform.request("sys_rst"))

        if with_accel_stub:
            self.accel_stub = AccelStub()

        if expose_host_wb:
            self.host_wb = wishbone.Interface(data_width=32, adr_width=30)
            self.bus.add_master(name="host_wb", master=self.host_wb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Setun LiteX SimSoC Target")
    builder_args(parser)
    soc_core_args(parser)
    parser.add_argument(
        "--with-accel-stub",
        action="store_true",
        help="Instantiate AccelStub CSR peripheral (Task 0.4)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Launch interactive Verilator binary after building",
    )

    args = parser.parse_args()

    soc_kwargs = soc_core_argdict(args)
    builder_kwargs = builder_argdict(args)

    soc_kwargs["cpu_type"] = None
    soc_kwargs["integrated_rom_size"] = 0
    soc_kwargs["integrated_main_ram_size"] = 0x10000
    builder_kwargs["compile_software"] = False

    if not builder_kwargs.get("csr_json"):
        out_dir = Path(builder_kwargs.get("output_dir") or "build/sim")
        builder_kwargs["csr_json"] = str(out_dir / "csr.json")

    soc = SetunSimSoC(with_accel_stub=args.with_accel_stub, **soc_kwargs)

    sim_config = SimConfig()
    sim_config.add_clocker("sys_clk", freq_hz=soc.clk_freq)
    sim_config.add_module("serial2console", "serial")

    builder = Builder(soc, **builder_kwargs)
    builder.build(sim_config=sim_config, run=args.run)
    print(
        "[sim-smoke] Verilator SoC gateware built successfully in",
        builder.output_dir,
    )


if __name__ == "__main__":
    main()
