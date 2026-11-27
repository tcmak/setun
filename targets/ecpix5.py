"""LambdaConcept ECPIX-5 (LFE5UM5G-85F) LiteX FPGA target (Tasks 6.2 & 6.3)."""

from __future__ import annotations

from litedram.modules import MT41K256M16
from litedram.phy.ecp5ddrphy import ECP5DDRPHY
from liteeth.phy.ecp5rgmii import LiteEthPHYRGMII
from litex.build.generic_platform import IOStandard, Misc, Pins, Subsignal
from litex.build.lattice import LatticeECP5Platform
from litex.gen.genlib.misc import WaitTimer
from litex.soc.cores.clock import ECP5PLL
from litex.soc.integration.soc_core import SoCCore
from migen import ClockDomain, Instance, Module, Signal
from migen.genlib.resetsync import AsyncResetSynchronizer

from setun.soc.ternary_accel import TernaryAccel

_ECPIX5_IO = [
    ("clk100", 0, Pins("K23"), IOStandard("LVCMOS33")),
    ("rst_n", 0, Pins("N5"), IOStandard("LVCMOS33")),
    (
        "serial",
        0,
        Subsignal("rx", Pins("R26"), IOStandard("LVCMOS33")),
        Subsignal("tx", Pins("R24"), IOStandard("LVCMOS33")),
    ),
    (
        "ddram",
        0,
        Subsignal("a", Pins("T5 M3 L3 V6 K2 W6 K3 L1 H2 L2 N1 J1 M1 K1"), IOStandard("SSTL15_I")),
        Subsignal("ba", Pins("U6 N3 N4"), IOStandard("SSTL15_I")),
        Subsignal("ras_n", Pins("T3"), IOStandard("SSTL15_I")),
        Subsignal("cas_n", Pins("P2"), IOStandard("SSTL15_I")),
        Subsignal("we_n", Pins("R3"), IOStandard("SSTL15_I")),
        Subsignal("dm", Pins("U4 U1"), IOStandard("SSTL15_I")),
        Subsignal(
            "dq",
            Pins("T4 W4 R4 W5 R6 P6 P5 P4 R1 W3 T2 V3 U3 W1 T1 W2"),
            IOStandard("SSTL15_I"),
            Misc("TERMINATION=75"),
        ),
        Subsignal(
            "dqs_p",
            Pins("V4 V1"),
            IOStandard("SSTL15D_I"),
            Misc("TERMINATION=OFF"),
            Misc("DIFFRESISTOR=100"),
        ),
        Subsignal("clk_p", Pins("H3"), IOStandard("SSTL15D_I")),
        Subsignal("cke", Pins("P3"), IOStandard("SSTL15_I")),
        Subsignal("odt", Pins("P1"), IOStandard("SSTL15_I")),
        Subsignal("reset_n", Pins("L4"), IOStandard("SSTL15_I")),
        Misc("SLEWRATE=FAST"),
    ),
    (
        "eth_clocks",
        0,
        Subsignal("tx", Pins("A12")),
        Subsignal("rx", Pins("E11")),
        IOStandard("LVCMOS33"),
    ),
    (
        "eth",
        0,
        Subsignal("rst_n", Pins("C13")),
        Subsignal("mdio", Pins("A13")),
        Subsignal("mdc", Pins("C11")),
        Subsignal("rx_ctl", Pins("B12")),
        Subsignal("rx_data", Pins("A11 B13 B11 D11")),
        Subsignal("tx_ctl", Pins("C12")),
        Subsignal("tx_data", Pins("D12 D13 E12 E13")),
        IOStandard("LVCMOS33"),
    ),
]


class ECPIX5Platform(LatticeECP5Platform):
    default_clk_name = "clk100"
    default_clk_period = 1e9 / 100e6

    def __init__(self, toolchain: str = "trellis") -> None:
        super().__init__("LFE5UM5G-85F-8BG554C", _ECPIX5_IO, toolchain=toolchain)


class _ECPIX5CRG(Module):
    def __init__(self, platform: ECPIX5Platform, sys_clk_freq: int) -> None:
        self.clock_domains.cd_init = ClockDomain()
        self.clock_domains.cd_por = ClockDomain(reset_less=True)
        self.clock_domains.cd_sys = ClockDomain()
        self.clock_domains.cd_sys2x = ClockDomain()
        self.clock_domains.cd_sys2x_i = ClockDomain()
        self.stop = Signal()
        self.reset = Signal()

        clk100 = platform.request("clk100")
        rst_n = platform.request("rst_n")

        por_done = Signal()
        por_timer = WaitTimer(int(100e6 * 10e-6))
        self.submodules += por_timer
        self.comb += [
            self.cd_por.clk.eq(clk100),
            por_timer.wait.eq(rst_n),
            por_done.eq(por_timer.done),
        ]

        self.submodules.pll = pll = ECP5PLL()
        self.comb += pll.reset.eq(~por_done)
        pll.register_clkin(clk100, 100e6)
        pll.create_clkout(self.cd_sys2x_i, 2 * sys_clk_freq, with_reset=False)
        pll.create_clkout(self.cd_init, 25e6, with_reset=False)
        self.specials += [
            Instance(
                "ECLKSYNCB",
                i_ECLKI=self.cd_sys2x_i.clk,
                i_STOP=self.stop,
                o_ECLKO=self.cd_sys2x.clk,
            ),
            Instance(
                "CLKDIVF",
                p_DIV="2.0",
                i_ALIGNWD=0,
                i_CLKI=self.cd_sys2x.clk,
                i_RST=self.reset,
                o_CDIVX=self.cd_sys.clk,
            ),
            AsyncResetSynchronizer(self.cd_init, ~por_done | ~pll.locked),
            AsyncResetSynchronizer(self.cd_sys, ~por_done | ~pll.locked | self.reset),
        ]
        platform.add_period_constraint(clk100, 1e9 / 100e6)
        platform.add_period_constraint(self.cd_sys.clk, 1e9 / sys_clk_freq)
        platform.add_period_constraint(self.cd_sys2x.clk, 1e9 / (2 * sys_clk_freq))


class SetunECPIX5SoC(SoCCore):
    def __init__(
        self,
        sys_clk_freq: int = int(100e6),
        accel_rows: int = 4,
        accel_lanes: int = 64,
        accel_group_size: int = 128,
        accel_beat_bits: int = 256,
        accel_max_k: int = 2048,
        accel_max_rows: int = 2048,
        with_dram: bool = True,
        with_etherbone: bool = False,
        etherbone_ip: str = "192.168.1.50",
        uartbone_baudrate: int = 1_000_000,
    ) -> None:
        platform = ECPIX5Platform(toolchain="trellis")
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
            ident="Setun ECPIX-5 SoC",
            ident_version=False,
        )
        self.submodules.crg = _ECPIX5CRG(platform, sys_clk_freq)

        if with_dram:
            self.ddrphy = ECP5DDRPHY(platform.request("ddram"), sys_clk_freq=sys_clk_freq)
            self.comb += [
                self.crg.stop.eq(self.ddrphy.init.stop),
                self.crg.reset.eq(self.ddrphy.init.reset),
            ]
            self.add_sdram(
                "sdram",
                phy=self.ddrphy,
                module=MT41K256M16(sys_clk_freq, "1:2"),
                l2_cache_size=0,
            )

        self.add_uartbone(baudrate=uartbone_baudrate)
        if with_etherbone:
            self.ethphy = LiteEthPHYRGMII(
                clock_pads=platform.request("eth_clocks"),
                pads=platform.request("eth"),
            )
            self.add_etherbone(phy=self.ethphy, ip_address=etherbone_ip)

        self.ternary_accel = TernaryAccel(
            rows=accel_rows,
            lanes=accel_lanes,
            group_size=accel_group_size,
            beat_bits=accel_beat_bits,
            max_k=accel_max_k,
            max_rows=accel_max_rows,
        )
        self.bus.add_master(name="ternary_accel_dma", master=self.ternary_accel.dma_wb)
