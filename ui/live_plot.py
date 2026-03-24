# ui/live_plot.py
# Rich terminal dashboard — edge-friendly, no matplotlib needed

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from intelligence.anomaly_engine import AnomalyResult
from intelligence.spike_tracker import SpikeTracker

console = Console()


# ── sparkline ─────────────────────────────────────────────────────────
_BLOCKS = " ▁▂▃▄▅▆▇█"

def _spark(values: list[float], width: int = 60) -> str:
    if not values:
        return " " * width
    mn, mx = min(values), max(values)
    span = mx - mn if mx != mn else 1.0
    out = ""
    for v in values[-width:]:
        out += _BLOCKS[int((v - mn) / span * (len(_BLOCKS) - 1))]
    return out.ljust(width)


def _bar(val: float, lo: float, hi: float, w: int = 18) -> str:
    pct = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    filled = int(pct * w)
    return "█" * filled + "░" * (w - filled)


# ── layout builder ────────────────────────────────────────────────────
def _build(result: AnomalyResult, tracker: SpikeTracker, patterns: int) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header",  size=3),
        Layout(name="body",    size=11),
        Layout(name="spark",   size=5),
        Layout(name="footer",  size=3),
    )
    layout["body"].split_row(Layout(name="sensors"), Layout(name="status"))

    # ── header ────────────────────────────────────────────────────────
    phase = "WARMUP" if "WARMUP" in result.reason else "DETECTING"
    phase_style = "bold yellow" if phase == "WARMUP" else "bold green"
    h = Text(justify="center")
    h.append("⚡ QDRANT EDGE ANOMALY DETECTION ", style="bold cyan")
    h.append(f"│ {phase} ", style=phase_style)
    h.append(f"│ step {result.step}  │ patterns: {patterns}", style="dim")
    layout["header"].update(Panel(h, box=box.MINIMAL))

    # ── sensors ───────────────────────────────────────────────────────
    st = Table(box=box.SIMPLE, show_header=True, header_style="bold blue")
    st.add_column("Sensor",  width=13)
    st.add_column("Value",   width=9)
    st.add_column("Level",   width=20)
    st.add_column("Unit",    width=5)

    tc = "red"    if result.temperature > 65 or result.temperature < 25 else "green"
    vc = "red"    if result.vibration   > 4.0 else ("yellow" if result.vibration > 2.0 else "green")
    pc = "red"    if result.pressure    < 0.95 or result.pressure > 1.4 else "green"

    st.add_row("Temperature", f"[{tc}]{result.temperature:6.1f}[/]",
               f"[{tc}]{_bar(result.temperature, 20, 90)}[/]", "°C")
    st.add_row("Vibration",   f"[{vc}]{result.vibration:6.3f}[/]",
               f"[{vc}]{_bar(result.vibration,   0, 10)}[/]",  "g")
    st.add_row("Pressure",    f"[{pc}]{result.pressure:6.4f}[/]",
               f"[{pc}]{_bar(result.pressure,   0.8, 1.6)}[/]", "bar")

    layout["sensors"].update(Panel(st, title="[bold]Live Sensors[/bold]", box=box.ROUNDED))

    # ── status ────────────────────────────────────────────────────────
    if result.is_anomaly:
        status_txt = Text("🚨  ANOMALY DETECTED", style="bold red blink")
    elif "WARMUP" in result.reason:
        status_txt = Text("⏳  WARMUP", style="bold yellow")
    else:
        status_txt = Text("✅  NORMAL", style="bold green")

    dt = Table(box=box.SIMPLE, show_header=False)
    dt.add_column("k", width=18)
    dt.add_column("v")

    conf_bar = "█" * int(result.confidence * 10) + "░" * (10 - int(result.confidence * 10))
    spike_icon = "⚡ YES" if result.spike else "—"
    gt_icon    = "✓ Anomaly" if result.ground_truth else "✗ Normal"

    dt.add_row("Status:",       status_txt)
    dt.add_row("Anomaly Score:",f"[bold]{result.anomaly_score:.4f}[/bold]")
    dt.add_row("Similarity:",   f"[cyan]{result.similarity:.4f}[/cyan]")
    dt.add_row("Confidence:",   f"[magenta]{conf_bar}[/magenta] {result.confidence:.2f}")
    dt.add_row("Spike:",        f"[yellow]{spike_icon}[/yellow]")
    dt.add_row("Ground Truth:", f"[dim]{gt_icon}[/dim]")
    dt.add_row("Reason:",       f"[dim]{result.reason[:38]}[/dim]")

    layout["status"].update(Panel(dt, title="[bold]Detection[/bold]", box=box.ROUNDED))

    # ── sparkline ─────────────────────────────────────────────────────
    scores = tracker.scores
    color  = "red" if result.is_anomaly else "cyan"
    sp = Text()
    sp.append("Anomaly Score  ", style="dim")
    sp.append(_spark(scores, 65), style=color)
    sp.append(f"  [{tracker.total_anomalies} flagged]",
              style="bold red" if tracker.total_anomalies else "dim")
    layout["spark"].update(Panel(sp, title="[bold]Score Timeline ▁▄█[/bold]", box=box.ROUNDED))

    # ── footer ────────────────────────────────────────────────────────
    f = Text(justify="center")
    f.append(
        f"Anomalies: {tracker.total_anomalies}  │  "
        f"TP: {tracker.true_positives}  │  "
        f"FP: {tracker.false_positives}  │  "
        f"Precision: {tracker.precision:.0%}  │  "
        f"Ctrl+C to stop",
        style="dim"
    )
    layout["footer"].update(Panel(f, box=box.MINIMAL))

    return layout


# ── public API ────────────────────────────────────────────────────────
class Dashboard:
    def __init__(self, tracker: SpikeTracker):
        self._tracker = tracker
        self._live: Live | None = None

    def __enter__(self):
        self._live = Live(console=console, refresh_per_second=12, screen=True)
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        if self._live:
            self._live.__exit__(*args)

    def update(self, result: AnomalyResult, patterns: int):
        if self._live:
            self._live.update(_build(result, self._tracker, patterns))
