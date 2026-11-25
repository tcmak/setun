"""LiteX SimSoC target for Setun (Tasks 0.3, 0.4, 3.1a, 3.1b, 3.2, & 3.3a)."""

from __future__ import annotations

import argparse
from pathlib import Path

from litex.build.generic_platform import Pins, Subsignal
from litex.build.sim import SimPlatform
from litex.build.sim.config import SimConfig
from litex.soc.cores import uart
from litex.soc.integration.builder import Builder, builder_argdict, builder_args
from litex.soc.integration.soc_core import (
    SoCCore,
    get_mem_data,
    soc_core_argdict,
    soc_core_args,
)
from litex.soc.interconnect import wishbone
from migen.genlib.io import CRG

from setun.accelerators.stub import AccelStub
from setun.soc.ternary_accel import TernaryAccel, export_soc_csr_artifacts

_io_rs232 = [
    ("sys_clk", 0, Pins(1)),
    ("sys_rst", 0, Pins(1)),
    ("serial", 0, Subsignal("tx", Pins(1)), Subsignal("rx", Pins(1))),
]

_io_sim_stream = [
    ("sys_clk", 0, Pins(1)),
    ("sys_rst", 0, Pins(1)),
    (
        "serial",
        0,
        Subsignal("source_valid", Pins(1)),
        Subsignal("source_ready", Pins(1)),
        Subsignal("source_data", Pins(8)),
        Subsignal("sink_valid", Pins(1)),
        Subsignal("sink_ready", Pins(1)),
        Subsignal("sink_data", Pins(8)),
    ),
]


class SetunSimPlatform(SimPlatform):
    def __init__(self, sim_serial: bool = False) -> None:
        super().__init__("sim", _io_sim_stream if sim_serial else _io_rs232)


class SetunSimSoC(SoCCore):
    def __init__(
        self,
        with_accel_stub: bool = False,
        with_ternary_accel: bool = True,
        accel_rows: int = 4,
        accel_lanes: int = 64,
        accel_group_size: int = 128,
        accel_beat_bits: int = 256,
        accel_max_k: int = 2048,
        accel_max_rows: int = 2048,
        ram_init: list[int] | Path | str | None = None,
        expose_host_wb: bool = False,
        with_uartbone: bool = False,
        use_external_crg: bool = True,
        **kwargs,
    ) -> None:
        platform = SetunSimPlatform(sim_serial=with_uartbone)
        kwargs.pop("uart_name", None)
        kwargs.setdefault("csr_data_width", 32)
        kwargs.setdefault("bus_bursting", True)
        kwargs.setdefault("ident_version", False)

        if with_uartbone:
            kwargs["with_uart"] = False

        if ram_init is not None:
            if isinstance(ram_init, (str, Path)):
                kwargs["integrated_main_ram_init"] = get_mem_data(
                    str(ram_init), data_width=32, endianness="little"
                )
            else:
                kwargs["integrated_main_ram_init"] = list(ram_init)

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

        if with_ternary_accel:
            self.ternary_accel = TernaryAccel(
                rows=accel_rows,
                lanes=accel_lanes,
                group_size=accel_group_size,
                beat_bits=accel_beat_bits,
                max_k=accel_max_k,
                max_rows=accel_max_rows,
            )
            self.bus.add_master(name="ternary_accel_dma", master=self.ternary_accel.dma_wb)

        if with_uartbone:
            self.uartbone_phy = uart.RS232PHYModel(platform.request("serial"))
            self.uartbone = uart.UARTBone(
                phy=self.uartbone_phy,
                clk_freq=self.clk_freq,
                cd="sys",
                address_width=self.bus.address_width,
            )
            self.bus.add_master(name="uartbone", master=self.uartbone.wishbone)

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
        "--no-ternary-accel",
        action="store_true",
        help="Disable TernaryAccel row pipeline peripheral",
    )
    parser.add_argument(
        "--serial2tcp-port",
        type=int,
        default=0,
        help="TCP port for serial2tcp module when --with-uartbone is active",
    )
    parser.add_argument(
        "--ram-init",
        type=str,
        default=None,
        help="Preload little-endian binary image into main_ram at 0x40000000 (Task 3.2)",
    )
    parser.add_argument(
        "--export-csr-reports",
        action="store_true",
        help="Deterministically export reports/csr.json and reports/csr.h and exit",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Launch interactive Verilator binary after building",
    )

    args = parser.parse_args()

    if args.export_csr_reports:
        export_soc_csr_artifacts()
        print("[sim-csr] Exported deterministic reports/csr.json and reports/csr.h")
        return

    soc_kwargs = soc_core_argdict(args)
    builder_kwargs = builder_argdict(args)

    soc_kwargs["cpu_type"] = None
    soc_kwargs["integrated_rom_size"] = 0
    soc_kwargs["integrated_main_ram_size"] = 0x400000
    builder_kwargs["compile_software"] = False

    if not builder_kwargs.get("csr_json"):
        out_dir = Path(builder_kwargs.get("output_dir") or "build/sim")
        builder_kwargs["csr_json"] = str(out_dir / "csr.json")

    with_uartbone = bool(soc_kwargs.pop("with_uartbone", getattr(args, "with_uartbone", False)))
    soc = SetunSimSoC(
        with_accel_stub=args.with_accel_stub,
        with_ternary_accel=not args.no_ternary_accel,
        with_uartbone=with_uartbone,
        ram_init=args.ram_init,
        **soc_kwargs,
    )

    sim_config = SimConfig()
    sim_config.add_clocker("sys_clk", freq_hz=soc.clk_freq)
    if args.with_uartbone and args.serial2tcp_port > 0:
        sim_config.add_module("serial2tcp", "serial", args={"port": str(args.serial2tcp_port)})
    else:
        sim_config.add_module("serial2console", "serial")

    builder = Builder(soc, **builder_kwargs)
    builder.build(sim_config=sim_config, run=args.run)
    print(
        "[sim-smoke] Verilator SoC gateware built successfully in",
        builder.output_dir,
    )


if __name__ == "__main__":
    main()
