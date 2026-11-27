"""Terasic DE10-Nano (Intel Cyclone V SE 5CSEBA6U23I7) LiteX target (Tasks 6.2 & 6.3)."""

from __future__ import annotations

from litex.build.altera import AlteraPlatform
from litex.build.generic_platform import IOStandard, Pins, Subsignal
from litex.soc.cores.clock import CycloneVPLL
from litex.soc.integration.soc import SoCRegion
from litex.soc.integration.soc_core import SoCCore
from litex.soc.interconnect import wishbone
from migen import ClockDomain, ClockSignal, If, Instance, Module, Signal
from migen.genlib.resetsync import AsyncResetSynchronizer

from setun.soc.ternary_accel import TernaryAccel

_DE10NANO_IO = [
    ("clk50", 0, Pins("V11"), IOStandard("3.3-V LVTTL")),
    ("rst_n", 0, Pins("AH17"), IOStandard("3.3-V LVTTL")),
    (
        "serial",
        0,
        Subsignal("tx", Pins("AG13")),
        Subsignal("rx", Pins("AF13")),
        IOStandard("3.3-V LVTTL"),
    ),
    ("user_led", 0, Pins("W15"), IOStandard("3.3-V LVTTL")),
]


class DE10NanoPlatform(AlteraPlatform):
    default_clk_name = "clk50"
    default_clk_period = 1e9 / 50e6
    create_rbf = False

    def __init__(self) -> None:
        super().__init__("5CSEBA6U23I7", _DE10NANO_IO, toolchain="quartus")


class _DE10NanoCRG(Module):
    def __init__(self, platform: DE10NanoPlatform, sys_clk_freq: int) -> None:
        self.clock_domains.cd_sys = ClockDomain()
        clk50 = platform.request("clk50")
        rst_n = platform.request("rst_n")

        self.submodules.pll = pll = CycloneVPLL(speedgrade="-I7")
        self.comb += pll.reset.eq(~rst_n)
        pll.register_clkin(clk50, 50e6)
        pll.create_clkout(self.cd_sys, sys_clk_freq)
        self.specials += AsyncResetSynchronizer(self.cd_sys, ~rst_n | ~pll.locked)
        platform.add_period_constraint(clk50, 1e9 / 50e6)
        platform.add_period_constraint(self.cd_sys.clk, 1e9 / sys_clk_freq)


class _CycloneVHPSDDRBridge(Module):
    """32-bit Wishbone slave bridging `0x40000000` (1 GiB) to Cyclone V `f2sdram` hard macro."""

    def __init__(self) -> None:
        self.bus = wb = wishbone.Interface(data_width=32, address_width=30, addressing="word")
        av_waitreq = Signal()
        av_rdata = Signal(32)
        av_rvalid = Signal()
        av_read = Signal()
        av_write = Signal()
        rd_pending = Signal()

        req = Signal()
        self.comb += [
            req.eq(wb.cyc & wb.stb & ~wb.ack),
            av_write.eq(req & wb.we & ~rd_pending),
            av_read.eq(req & ~wb.we & ~rd_pending),
        ]
        self.sync += [
            wb.ack.eq(0),
            If(av_write & ~av_waitreq, wb.ack.eq(1)),
            If(av_read & ~av_waitreq, rd_pending.eq(1)),
            If(rd_pending & av_rvalid, wb.dat_r.eq(av_rdata), wb.ack.eq(1), rd_pending.eq(0)),
        ]
        self.specials += Instance(
            "cyclonev_hps_interface_fpga2sdram",
            i_clk=ClockSignal("sys"),
            i_address=wb.adr,
            i_byteenable=wb.sel,
            i_read=av_read,
            i_write=av_write,
            i_writedata=wb.dat_w,
            o_readdata=av_rdata,
            o_readdatavalid=av_rvalid,
            o_waitrequest=av_waitreq,
        )


class SetunDE10NanoSoC(SoCCore):
    def __init__(
        self,
        sys_clk_freq: int = int(100e6),
        accel_rows: int = 4,
        accel_lanes: int = 64,
        accel_group_size: int = 128,
        accel_beat_bits: int = 256,
        accel_max_k: int = 2048,
        accel_max_rows: int = 2048,
        uartbone_baudrate: int = 1_000_000,
    ) -> None:
        platform = DE10NanoPlatform()
        super().__init__(
            platform=platform,
            clk_freq=sys_clk_freq,
            cpu_type=None,
            integrated_rom_size=0,
            integrated_sram_size=0,
            integrated_main_ram_size=0,
            with_uart=False,
            csr_data_width=32,
            bus_bursting=True,
            ident="Setun DE10-Nano SoC",
            ident_version=False,
        )
        self.submodules.crg = _DE10NanoCRG(platform, sys_clk_freq)
        self.submodules.hps_ddr = _CycloneVHPSDDRBridge()
        self.bus.add_slave(
            "main_ram",
            self.hps_ddr.bus,
            SoCRegion(origin=0x40000000, size=0x40000000, mode="rwx"),
        )
        self.add_uartbone(baudrate=uartbone_baudrate)
        self.ternary_accel = TernaryAccel(
            rows=accel_rows,
            lanes=accel_lanes,
            group_size=accel_group_size,
            beat_bits=accel_beat_bits,
            max_k=accel_max_k,
            max_rows=accel_max_rows,
        )
        self.bus.add_master(name="ternary_accel_dma", master=self.ternary_accel.dma_wb)
