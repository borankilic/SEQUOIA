import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from sequoia.fock.experiment import save_experiment
from sequoia.setup.io import save_setup


def _plain(value):
    from sequoia.runs import json_safe

    return json_safe(value)


def run_record(res) -> dict:
    """Machine-readable record of a Photon counts run: what was run and what came out.
    Non-finite numbers are written as null."""
    se = res.setup_experiment
    cnt = res.counts
    ids = res.detector_ids

    def label(state):
        return ",".join(f"{d}={n}" for d, n in zip(ids, state, strict=True))

    from sequoia.runs import new_run_id, setup_yaml, used_parts

    text = setup_yaml(se.setup)
    try:
        fingerprint = se.library.fingerprint(used_parts(se.setup))
    except Exception:
        fingerprint = None
    record = {
        "format": "sequoia-photon-counts-run",
        "version": 1,
        "run_id": new_run_id("fock"),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "setup_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "library_fingerprint": fingerprint,
        "status": res.status,
        "engine": {"name": "fock", "backend": res.result.backend, **res.result.versions},
        "setup": se.setup.name,
        "detectors": ids,
        "detector_models": [s.model.to_dict() for s in se.compiled.detectors],
        "dead_time_s": se.dead_times,
        "measurement": {
            "window_s": se.window_s,
            "windows_per_second": cnt.windows_per_second,
            "duration_s": cnt.duration_s,
            "n_windows": cnt.n_windows,
            "acquisition": cnt.acquisition,
            "rates": "potential clicks before dead time (dead time not modelled)",
        },
        "inputs": [d.__dict__ for d in se.inputs],
        "limits": {"max_photons": se.experiment.limits.max_photons},
        "normalization": {
            "denominator": "probability per detection window",
            "resolved_probability": res.mass,
            "omitted_input_weight": res.omitted_weight,
        },
        "probabilities": {label(s): p for s, p in res.detected.items()},
        "expected_counts": {label(s): v for s, v in (cnt.total or {}).items()},
        "sigma_counts": {label(s): v for s, v in (cnt.sigma or {}).items()},
        "dead_time_load": res.dead_time_load,
        "notes": list(se.notes),
        "messages": list(res.messages),
    }
    if res.samples is not None:
        record["samples"] = {
            "shots": res.samples.shots,
            "seed": res.samples.seed,
            "counts": {label(s): c for s, c in res.samples.counts.items()},
            "unresolved_shots": res.samples.unresolved,
        }
    return _plain(record)


def write_run_bundle(res, directory: str | Path) -> Path:
    """A directory another researcher can rerun from: the setup as it was run, the library parts
    it uses, the photon-number experiment, the machine-readable record and a summary."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    se = res.setup_experiment
    save_setup(se.setup, out / "setup.yaml")
    refs = {i.part for i in se.setup.instances}
    refs |= {getattr(link, "fiber", None) for link in se.setup.links} - {None}
    refs |= {p.part for p in se.setup.probes}
    for ref in sorted(refs):
        source = se.library.part_dir(ref)
        if source.is_dir():
            shutil.copytree(source, out / "library" / ref, dirs_exist_ok=True)
    save_experiment(se.experiment, out / "experiment.yaml")
    record = run_record(res)
    (out / "run.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        f"Photon counts run of '{se.setup.name}'",
        f"window {se.window_s:.6g} s, {res.counts.windows_per_second:.6g} windows/s, "
        f"duration {res.counts.duration_s:g} s; detectors {', '.join(res.detector_ids)}",
        f"resolved probability {res.mass:.12g}, omitted input weight {res.omitted_weight:.3g}",
        f"status: {res.status}",
        "rates are potential clicks before dead time; sigma is binomial over fixed windows",
        "rerun: sequoia GUI > open setup.yaml with library/ as the library, Photon counts > Run",
        "",
    ]
    lines += [f"- {m}" for m in res.messages]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
