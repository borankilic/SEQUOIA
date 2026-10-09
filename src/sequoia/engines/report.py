import math

from sequoia.engines.classical import SimulationResult


def _mm(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value * 1e3:.4g}"


def format_report(result: SimulationResult) -> str:
    lines = []
    header = (
        f"{'link':<10}{'kind':<11}{'from -> to':<28}{'P [mW]':>10}{'w_x [mm]':>10}"
        f"{'w_y [mm]':>10}{'R_x [m]':>10}{'DOP':>7}"
    )
    lines += ["Links", header, "-" * len(header)]
    for state in result.links.values():
        route = f"{state.src} -> {state.dst}"
        light = state.end
        if light is None:
            lines.append(f"{state.id:<10}{state.kind:<11}{route:<28}{'(no light)':>10}")
            continue
        m = light.mode
        radius = m.radius_x
        lines.append(
            f"{state.id:<10}{state.kind:<11}{route:<28}{light.power * 1e3:>10.4g}"
            f"{_mm(m.w_x):>10}{_mm(m.w_y):>10}"
            f"{('inf' if math.isinf(radius) else f'{radius:.4g}'):>10}{light.dop:>7.3f}"
        )
    fibers = [s for s in result.links.values() if "coupling_efficiency" in s.info]
    if fibers:
        lines += ["", "Fiber links"]
        for s in fibers:
            lines.append(
                f"  {s.id}: coupling efficiency {100 * s.info['coupling_efficiency']:.2f}%, "
                f"transmission {100 * s.info['transmission']:.2f}%"
            )
    if result.readouts:
        lines += ["", "Readouts"]
        for inst_id, values in result.readouts.items():
            parts = ", ".join(f"{k} = {v:.6g}" for k, v in values.items())
            lines.append(f"  {inst_id}: {parts}")
    if result.probes:
        lines += ["", "Probes"]
        for probe_id, values in result.probes.items():
            scalars = {k: v for k, v in values.items() if not isinstance(v, list)}
            parts = ", ".join(f"{k} = {v:.6g}" for k, v in scalars.items())
            lines.append(f"  {probe_id}: {parts}")
    if result.issues:
        lines += ["", "Diagnostics"]
        for issue in result.issues:
            lines.append(f"  [{issue.severity}] {issue.code}: {issue.subject}: {issue.message}")
    return "\n".join(lines)
