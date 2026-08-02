"""Generate the bundled montage CSVs from the source templates in ``_templates/``.

This is a maintenance script, not imported at runtime. Run it to (re)produce the
montage CSVs in this directory::

    python -m spherical_eeg_sim.montages._generate

All positions are placed on the model scalp sphere (radius ``R_TEMPLATE``) by
radial projection of the vetted, BSD-licensed reference templates (see
``_templates/NOTICE`` for provenance and attribution):

  * standard 10-20 / 10-10 / 10-05 family  <- standard_1005.elc
        Fiducial-anchored projection; the frame is aligned to the head model
        (z -> Cz vertex, y -> Fpz anterior, x -> right). Exact left-right
        symmetry is enforced. Deprecated aliases (T3/T4/T5/T6, A1/A2) are
        dropped. Positions follow Oostenveld & Praamstra (2001).

  * EGI / Philips Geodesic Sensor Net       <- GSN-HydroCel-*.sfp  (``_spherical``)
        Template axes (z=vertex, y=anterior) are trusted; anchored to Cz when
        the template includes it.

  * BioSemi nets                            <- biosemi*.txt        (``_spherical``)
        EEGLAB polar coordinates (Site, Theta, Phi in degrees) are converted
        with unit = [sin(t)cos(p), sin(t)sin(p), cos(t)], a convention verified
        to 2.5 deg against the shared 10-10 subset. Only the true anatomical
        fiducials (nasion and the preauricular points) are excluded; the inion
        (Iz) is retained as a recording electrode where the layout includes it,
        so the channel counts match the manufacturer/MNE designations (e.g. the
        64-channel cap keeps all 64 recording electrodes). 10-10-named layouts
        are symmetrised; geodesic (A-H) layouts keep their native arrangement.

Vendor-net files carry the ``_spherical`` suffix to make explicit that they are
spherical approximations of the manufacturers' layouts, not the original 3-D
geometry.
"""
from __future__ import annotations
import re
import csv
from pathlib import Path
import numpy as np

R_TEMPLATE = 0.092  # radius (m) at which the shipped CSVs are stored
_HERE = Path(__file__).resolve().parent
_TEMPLATES = _HERE / '_templates'

FIDUCIALS = {'LPA', 'RPA', 'Nz', 'Iz', 'FidNz', 'FidT9', 'FidT10'}
DEPRECATED = {'T3', 'T4', 'T5', 'T6', 'A1', 'A2'}

# BioSemi caps list the inion (Iz) as a genuine recording electrode -- as does
# MNE-Python's ``biosemi*`` montage set -- whereas in the standard 10-05
# template the inion is used only as an anatomical anchor. Parse BioSemi layouts
# against the true fiducials only (nasion and the preauricular points), keeping
# the inion, so the shipped channel counts match the manufacturer/MNE designations.
BIOSEMI_FIDUCIALS = {'Nz', 'LPA', 'RPA', 'FidNz', 'FidT9', 'FidT10'}

# Channel selections for the standard-family subsets (positions come from the
# 10-05 projection; these lists fix which electrodes each montage contains).
_SEL_1020_19 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
                'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
_SEL_1020_20 = ['Fz', 'Cz', 'Pz', 'Oz', 'Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8',
                'T7', 'C3', 'C4', 'T8', 'P7', 'P3', 'P4', 'P8', 'O1', 'O2']
_SEL_1010_32 = ['AFz', 'Fz', 'Cz', 'Pz', 'POz', 'Oz', 'Fp1', 'Fp2', 'F7', 'F3',
                'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8',
                'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'P4',
                'P8', 'O1', 'O2']
_SEL_1010_61 = ['Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz',
                'Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1',
                'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4',
                'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'T8',
                'TP7', 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6', 'TP8', 'P7',
                'P5', 'P3', 'P1', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'PO4',
                'PO8', 'O1', 'O2']
_SEL_1010_69 = ['Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz',
                'Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1',
                'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FC2',
                'FC4', 'FC6', 'FT8', 'FT10', 'T9', 'T7', 'C5', 'C3', 'C1', 'C2',
                'C4', 'C6', 'T8', 'T10', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CP2',
                'CP4', 'CP6', 'TP8', 'TP10', 'P9', 'P7', 'P5', 'P3', 'P1', 'P2',
                'P4', 'P6', 'P8', 'P10', 'PO7', 'PO3', 'PO4', 'PO8', 'O1', 'O2']


# ------------------------------------------------------------------ parsers
def _parse_elc(path):
    L = path.read_text().splitlines()
    pi = next(i for i, l in enumerate(L) if l.strip().lower().startswith('positions'))
    li = next(i for i, l in enumerate(L) if l.strip().lower().startswith('labels'))
    pos = [[float(x) for x in l.split()[-3:]] for l in L[pi + 1:li] if len(l.split()) >= 3]
    labs = [w for l in L[li + 1:] for w in l.split()]
    return {n: np.array(v) for n, v in zip(labs, np.array(pos, float))}


def _parse_sfp(path):
    d = {}
    for l in path.read_text().splitlines():
        p = l.split()
        if len(p) >= 4:
            try:
                d[p[0]] = np.array([float(p[1]), float(p[2]), float(p[3])])
            except ValueError:
                pass
    return d


def _parse_biosemi(path):
    elec = {}
    for i, l in enumerate(path.read_text().splitlines()):
        p = l.split()
        if i == 0 or len(p) < 3:
            continue
        try:
            t, ph = float(p[1]), float(p[2])
        except ValueError:
            continue
        if p[0] not in BIOSEMI_FIDUCIALS:
            elec[p[0]] = (t, ph)
    return elec


# ------------------------------------------------------------------ geometry
def _fit_sphere(Q):
    A = np.c_[2 * Q, np.ones(len(Q))]
    sol, *_ = np.linalg.lstsq(A, (Q ** 2).sum(1), rcond=None)
    return sol[:3]


def _mirror_name(n):
    core, suf = (n[:-1], n[-1]) if n and n[-1] == 'h' else (n, '')
    m = re.match(r'^(.*?)(\d+)$', core)
    if not m:
        return None
    stem, num = m.group(1), int(m.group(2))
    return f'{stem}{num + 1}{suf}' if num % 2 == 1 else None


def _symmetrise(proj):
    for n in list(proj):
        if n.endswith('z') or abs(proj[n][0]) < 1e-9:
            v = np.array([0.0, proj[n][1], proj[n][2]])
            proj[n] = R_TEMPLATE * v / np.linalg.norm(v)
        part = _mirror_name(n)
        if part in proj:
            a, b = proj[n], proj[part]
            mx = 0.5 * (abs(a[0]) + abs(b[0])); my = 0.5 * (a[1] + b[1]); mz = 0.5 * (a[2] + b[2])
            for tgt, sgn in ((n, -1), (part, +1)):
                v = np.array([sgn * mx, my, mz]); proj[tgt] = R_TEMPLATE * v / np.linalg.norm(v)
    return proj


def _project_cartesian(P, up_ref, ant_ref):
    rec = {n: v for n, v in P.items() if n not in FIDUCIALS and n not in DEPRECATED}
    c = _fit_sphere(np.array(list(rec.values())))
    zh = (P[up_ref] - c) if up_ref in P else np.array([0.0, 0.0, 1.0])
    zh = zh / np.linalg.norm(zh)
    ya = (P[ant_ref] - c); yh = ya - (ya @ zh) * zh; yh /= np.linalg.norm(yh)
    xh = np.cross(yh, zh); xh /= np.linalg.norm(xh)
    out = {}
    for n, v in rec.items():
        u = (v - c); u /= np.linalg.norm(u)
        out[n] = R_TEMPLATE * np.array([u @ xh, u @ yh, u @ zh])
    return out


def _biosemi_units(elec):
    out = {}
    for n, (t, p) in elec.items():
        t, p = np.deg2rad(t), np.deg2rad(p)
        out[n] = R_TEMPLATE * np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    return out


# ------------------------------------------------------------------ output
def _row(name, xyz):
    xyz = np.asarray(xyz, dtype=np.float64).copy()
    r = float(np.linalg.norm(xyz))
    # Sphere fits can leave stack-dependent 1e-18 residuals on axes that are
    # mathematically zero. Canonicalize them before both Cartesian formatting
    # and azimuth calculation; at an exact pole the conventional azimuth is 0.
    xyz[np.abs(xyz) < 1e-12 * r] = 0.0
    x, y, z = xyz
    theta = float(np.degrees(np.arccos(np.clip(z / r, -1, 1))))
    phi = 0.0 if x == 0.0 and y == 0.0 else float(
        np.degrees(np.arctan2(y, x)) % 360.0
    )
    return [name, f'{x:.9g}', f'{y:.9g}', f'{z:.9g}', f'{r:.9g}', f'{theta:.9g}', f'{phi:.9g}']


def _write_validate(names, proj, label, symmetric):
    XYZ = np.array([proj[n] for n in names])
    D = np.linalg.norm(XYZ[:, None, :] - XYZ[None, :, :], axis=-1); np.fill_diagonal(D, np.inf)
    ms = D.min() * 1000
    osd = np.max(np.abs(np.linalg.norm(XYZ, axis=1) - R_TEMPLATE)) * 1000
    sym = 0.0
    if symmetric:
        r = [max(abs(proj[n][0] + proj[_mirror_name(n)][0]),
                 abs(proj[n][1] - proj[_mirror_name(n)][1]),
                 abs(proj[n][2] - proj[_mirror_name(n)][2])) * 1000
             for n in names if _mirror_name(n) in names]
        sym = max(r) if r else 0.0
    ok = ms > 1.0 and osd < 1e-6 and (sym < 1e-6 or not symmetric)
    with (_HERE / f'{label}.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['Name', 'x', 'y', 'z', 'R', 'theta', 'phi'])
        for n in names:
            w.writerow(_row(n, proj[n]))
    print(f'  [{"OK" if ok else "CHECK"}] {label:26s} n={len(names):3d}  '
          f'min-space={ms:4.1f}mm  on-sphere={osd:.0e}m  sym={sym:.0e}m')
    return ok


def main():
    ok = True

    std = _symmetrise(_project_cartesian(_parse_elc(_TEMPLATES / 'standard_1005.elc'), 'Cz', 'Fpz'))
    allnames = sorted(std)
    print(f'standard 10-05 template: {len(allnames)} recording electrodes\n\nStandard family:')
    std_targets = {
        f'standard_1005_{len(allnames)}': allnames,
        'standard_1020_19': _SEL_1020_19, 'standard_1020_20': _SEL_1020_20,
        'standard_1010_32': _SEL_1010_32, 'standard_1010_61': _SEL_1010_61,
        'standard_1010_69': _SEL_1010_69,
    }
    for label, names in std_targets.items():
        miss = [n for n in names if n not in std]
        if miss:
            print(f'  [MISS] {label}: {miss}'); ok = False; continue
        ok &= _write_validate(names, std, label, symmetric=True)

    print('\nEGI / Philips HydroCel geodesic nets:')
    for f in sorted(_TEMPLATES.glob('GSN-HydroCel-*.sfp')):
        cnt = f.stem.split('-')[-1].replace('_1.0', '')
        proj = _project_cartesian(_parse_sfp(f), 'Cz', 'FidNz')
        ok &= _write_validate(list(proj), proj, f'gsn_hydrocel_{cnt}_spherical', symmetric=False)

    print('\nBioSemi nets:')
    for f in sorted(_TEMPLATES.glob('biosemi*.txt')):
        match = re.search(r'biosemi(\d+)', f.name)
        if match is None:
            raise ValueError(f"unexpected BioSemi template name: {f.name}")
        cnt = match.group(1)
        proj = _biosemi_units(_parse_biosemi(f))
        named_1010 = all(not re.match(r'^[A-H]\d+$', n) for n in proj)
        if named_1010:
            proj = _symmetrise(proj)
        ok &= _write_validate(list(proj), proj, f'biosemi_{cnt}_spherical', symmetric=named_1010)

    print('\n' + ('ALL MONTAGES VALID' if ok else 'SOME CHECKS FAILED'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
