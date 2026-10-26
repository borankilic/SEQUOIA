import networkx as nx

from sequoia.components.base import PortDirection, PortKind, PortSpec
from sequoia.errors import Issue, SequoiaError, SetupError
from sequoia.library.store import Library
from sequoia.measurement import check_measurement
from sequoia.setup.model import Setup
from sequoia.setup.resolve import instantiate

FIBER_TYPE = "OpticalFiber"

_PORT_KIND_FOR_LINK = {
    "free_space": PortKind.FREE_SPACE,
    "fiber": PortKind.FIBER,
    "cable": PortKind.ELECTRICAL,
}


class SetupValidationError(SetupError):
    def __init__(self, issues: list[Issue]) -> None:
        self.issues = tuple(issues)
        lines = "\n".join(f"  - {i}" for i in issues if i.severity == "error")
        super().__init__(f"invalid setup:\n{lines}")


def _unique(items, kind: str, issues: list[Issue]) -> dict:
    seen = {}
    for item in items:
        if item.id in seen:
            issues.append(
                Issue(
                    "error",
                    "duplicate_id",
                    f"{kind} '{item.id}'",
                    f"{kind} id is used more than once",
                )
            )
        else:
            seen[item.id] = item
    return seen


def check_setup(setup: Setup, library: Library) -> list[Issue]:
    issues: list[Issue] = []

    def error(code: str, subject: str, message: str) -> None:
        issues.append(Issue("error", code, subject, message))

    instances = _unique(setup.instances, "instance", issues)
    links = _unique(setup.links, "link", issues)
    probes = _unique(setup.probes, "probe", issues)

    ports_of: dict[str, dict[str, PortSpec]] = {}
    for inst in instances.values():
        try:
            component = instantiate(library, inst)
        except SequoiaError as exc:
            error("invalid_instance", f"instance '{inst.id}'", str(exc))
            continue
        if component.category == "probes":
            error(
                "wrong_category",
                f"instance '{inst.id}'",
                f"'{inst.part}' is a probe; probes attach to links and go under 'probes'",
            )
            continue
        ports_of[inst.id] = {p.name: p for p in component.ports()}
        try:
            component.tagger_model()
        except SequoiaError as exc:
            error("invalid_settings", f"instance '{inst.id}'", str(exc))

    port_use: dict[str, str] = {}
    graph = nx.DiGraph()
    graph.add_nodes_from(instances)
    for link in links.values():
        subject = f"link '{link.id}'"
        wanted = _PORT_KIND_FOR_LINK[link.kind]
        ends = ((link.src, PortDirection.OUT, "from"), (link.dst, PortDirection.IN, "to"))
        for endpoint, direction, role in ends:
            inst_id, port_name = endpoint.split(".")
            if inst_id not in instances:
                error("bad_endpoint", subject, f"'{endpoint}': there is no instance '{inst_id}'")
                continue
            if inst_id not in ports_of:
                continue
            port = ports_of[inst_id].get(port_name)
            if port is None:
                names = ", ".join(ports_of[inst_id]) or "none"
                error(
                    "bad_endpoint",
                    subject,
                    f"'{endpoint}': instance '{inst_id}' has no port '{port_name}' "
                    f"(ports: {names})",
                )
                continue
            if port.kind != wanted:
                hint = ""
                if {port.kind, wanted} == {PortKind.FREE_SPACE, PortKind.FIBER}:
                    hint = " (a beam needs a fiber coupler to enter or leave a fiber)"
                error(
                    "kind_mismatch",
                    subject,
                    f"{link.kind} link cannot use {port.kind} port '{endpoint}'{hint}",
                )
            if port.direction != direction:
                error(
                    "direction",
                    subject,
                    f"'{endpoint}' is an {port.direction} port but the link goes {role} it; "
                    "links run from an output port to an input port",
                )
            if endpoint in port_use:
                error(
                    "port_reused",
                    subject,
                    f"port '{endpoint}' is already connected by link '{port_use[endpoint]}'",
                )
            else:
                port_use[endpoint] = link.id
        src_inst, dst_inst = link.src.split(".")[0], link.dst.split(".")[0]
        if src_inst in instances and dst_inst in instances:
            graph.add_edge(src_inst, dst_inst)

        if link.kind == "fiber":
            try:
                fiber_part = library.load_part(link.fiber)
            except SequoiaError as exc:
                error("bad_fiber", subject, str(exc))
            else:
                if fiber_part.type != FIBER_TYPE:
                    error(
                        "bad_fiber",
                        subject,
                        f"'{link.fiber}' is a {fiber_part.type}, not an {FIBER_TYPE}",
                    )

    for probe in probes.values():
        subject = f"probe '{probe.id}'"
        if probe.link not in links:
            error("bad_probe", subject, f"there is no link '{probe.link}'")
        try:
            component = instantiate(library, probe)
        except SequoiaError as exc:
            error("invalid_instance", subject, str(exc))
            continue
        if component.category != "probes":
            error("bad_probe", subject, f"'{probe.part}' is not a probe part")

    for inst_id, ports in ports_of.items():
        for port in ports.values():
            if f"{inst_id}.{port.name}" in port_use:
                continue
            subject = f"port '{inst_id}.{port.name}'"
            if port.required:
                error(
                    "missing_required",
                    subject,
                    f"required {port.kind} {port.direction} port is not connected",
                )
            elif port.direction == PortDirection.OUT and port.kind != PortKind.ELECTRICAL:
                issues.append(
                    Issue(
                        "warning",
                        "unconnected_output",
                        subject,
                        "output port is not connected; light leaving it is lost",
                    )
                )

    try:
        cycle = nx.find_cycle(graph)
    except nx.NetworkXNoCycle:
        pass
    else:
        path = " -> ".join([cycle[0][0]] + [edge[1] for edge in cycle])
        error(
            "cycle", "setup", f"links form a cycle ({path}); only feed-forward setups are supported"
        )
    issues += check_measurement(setup)
    return issues


def validate_setup(setup: Setup, library: Library) -> list[Issue]:
    issues = check_setup(setup, library)
    if any(i.severity == "error" for i in issues):
        raise SetupValidationError(issues)
    return issues
