"""LiteX SimSoC target for Setun (Task 0.3)."""

from __future__ import annotations

import argparse

from litex.build.generic_platform import Pins, Subsignal
from litex.build.sim import SimPlatform
from litex.build.sim.config import SimConfig
from litex.soc.integration.builder import Builder, builder_argdict, builder_args
from litex.soc.integration.soc_core import SoCCore, soc_core_argdict, soc_core_args
from migen.genlib.io import CRG

_io = [
    ("sys_clk", 0, Pins(1)),
    ("sys_rst", 0, Pins(1)),
    ("serial", 0, Subsignal("tx", Pins(1)), Subsignal("rx", Pins(1))),
]


class SetunSimPlatform(SimPlatform):
    def __init__(self):
        super().__init__("sim", _io)


class SetunSimSoC(SoCCore):
    def __init__(self, **kwargs):
        platform = SetunSimPlatform()
        kwargs.pop("uart_name", None)

        super().__init__(
            platform=platform,
            clk_freq=int(1e6),
            ident="Setun Simulated SoC",
            uart_name="serial",
            **kwargs,
        )
        self.crg = CRG(platform.request("sys_clk"), platform.request("sys_rst"))


def main():
    parser = argparse.ArgumentParser(description="Setun LiteX SimSoC Target")
    builder_args(parser)
    soc_core_args(parser)
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

    soc = SetunSimSoC(**soc_kwargs)

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
