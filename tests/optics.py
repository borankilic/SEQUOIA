import sequoia.components.builtin  # noqa: F401
from sequoia.components.base import get_component
from sequoia.engines.classical import simulate
from sequoia.library.store import Library, Part
from sequoia.setup.io import setup_from_dict

LASER = {
    "wavelength": "780 nm",
    "power": "1 mW",
    "waist_radius": "1 mm",
    "polarization_extinction_ratio": 1e9,
}


def put(library: Library, ref: str, type_name: str, **params) -> str:
    cls = get_component(type_name)
    library.save_part(Part(ref, type_name, cls.Params.model_validate(params)))
    return ref


def inst(inst_id, part, settings=None, **extra):
    return {"id": inst_id, "part": part, "settings": settings or {}, **extra}


def fs(link_id, src, dst, length="100 mm"):
    return {"id": link_id, "kind": "free_space", "from": src, "to": dst, "length": length}


def fiber(link_id, src, dst, part, length="1 m", **extra):
    return {
        "id": link_id,
        "kind": "fiber",
        "from": src,
        "to": dst,
        "fiber": part,
        "length": length,
        **extra,
    }


def probe(probe_id, link, part="probes/power"):
    return {"id": probe_id, "part": part, "link": link}


def run(library, instances, links, probes=()):
    setup = setup_from_dict(
        {"instances": list(instances), "links": list(links), "probes": list(probes)}
    )
    return simulate(setup, library)


def basic_library(library: Library) -> Library:
    put(library, "laser", "Laser", **LASER)
    put(library, "probes/power", "PowerMeter")
    put(library, "probes/pol", "Polarimeter")
    put(library, "probes/beam", "BeamProfiler")
    put(library, "mirror", "Mirror")
    put(library, "bs", "BeamSplitter")
    put(library, "pd", "Photodiode", responsivity="0.5 A/W", dark_current="2 nA")
    put(library, "polarizer", "Polarizer", extinction_ratio=1e9)
    put(library, "hwp", "WavePlate", retardance=0.5, design_wavelength="780 nm")
    put(library, "qwp", "WavePlate", retardance=0.25, design_wavelength="780 nm")
    put(library, "pbs", "PBS")
    put(library, "iris", "Iris")
    return library


def power(result, probe_id="p"):
    return result.probes[probe_id]["power_W"]
