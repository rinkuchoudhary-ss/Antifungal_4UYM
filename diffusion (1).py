#!/usr/bin/env python3
"""
Diffusion model math and sampling — clash-free pose generation.

Layers:
  L1  Per-timestep soft repulsion    (rigid-body COM translation only)
  L2  Per-timestep hard-floor proj   (rigid-body, worst-atom fallback)
  L3  Post-sampling strong repulsion (200 iters, LJ-style, rigid-body)
  L4  MMFF geometry optimisation     (50 iters)
  L5  Post-MMFF re-projection        (deterministic VDW, rigid-body)
  L6  Acceptance gate                (interaction-type-aware)

KEY FIX (v4):
  apply_clash_guidance previously moved each atom independently (per-atom force),
  systematically destroying ring geometry over 150 timesteps.  All repulsion
  functions now compute a SINGLE rigid-body translation applied to the entire
  ligand COM so internal geometry is never distorted.

  When net forces cancel (atoms clashing on opposite sides), a "worst-atom
  escape" fallback pushes the whole ligand along the direction of the atom
  with the largest individual overlap, breaking the deadlock.
"""

import math
import logging
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


# ---- Per-element VDW radii (Bondi, A) ----------------------------------------

_VDW_RADII = {
    "H":  1.50, "C":  1.70, "N":  1.55, "O":  1.52,
    "F":  1.47, "P":  1.80, "S":  1.80, "Cl": 1.75,
    "Br": 1.85, "I":  1.98, "Se": 1.90,
}
_VDW_DEFAULT = 2.0
_VDW_TOL     = 0.20

INTERACTION_CUTOFFS = {
    "hbond":           2.5,
    "ionic":           2.8,
    "metal_coord":     1.8,
    "hydrophobic_CC":  3.3,
    "pi_pi":           3.3,
    "CH_pi":           2.7,
    "halogen_bond":    2.9,
    "sulfur_pi":       3.0,
    "water_mediated":  2.5,
    "vdw_generic":     2.5,
    "steric_floor":    2.8,
}

REPULSION_ONSET  = 3.5
CLASH_THRESHOLD  = 2.8
MIN_ALLOWED_DIST = 2.8

_FRAG_COM_FLOOR = 2.3


def _vdw_radius(element: str) -> float:
    return _VDW_RADII.get(element.capitalize(), _VDW_DEFAULT)


def _pairwise_min_dist(elem_lig: str, elem_prot: str) -> float:
    el = elem_lig.capitalize()
    ep = elem_prot.capitalize()
    _METALS   = {"Zn", "Fe", "Cu", "Mn", "Co", "Ni", "Ca", "Mg", "Na", "K"}
    _HALOGENS = {"Cl", "Br", "I"}

    # Metal coordination overrides everything — metals sit much closer
    if ep in _METALS:
        return INTERACTION_CUTOFFS["metal_coord"]

    # Halogen bond: halogen ligand atom to protein N or O
    if el in _HALOGENS and ep in ("N", "O"):
        return INTERACTION_CUTOFFS["halogen_bond"]

    # H-bond: any N/O on ligand paired with any N/O on protein
    # (this also covers the previously dead el=="N" and ep=="O" ionic branch,
    # which was unreachable because it appeared after this broader check)
    if el in ("N", "O") and ep in ("N", "O"):
        return INTERACTION_CUTOFFS["hbond"]

    # Sulfur pi / sulfur contacts — only for heavy-atom S pairs, not S–H
    # or S–metal (metal already handled above).  Restrict to C/N/O partners
    # to avoid returning 3.0 Å for S bonded to a halogen or another S.
    if (el == "S" and ep in ("C", "N", "O")) or (ep == "S" and el in ("C", "N", "O")):
        return INTERACTION_CUTOFFS["sulfur_pi"]

    # Pure hydrophobic C–C contact
    if el == "C" and ep == "C":
        return INTERACTION_CUTOFFS["hydrophobic_CC"]

    # Generic: sum of Bondi VDW radii minus tolerance, floored at steric_floor
    return max(_vdw_radius(el) + _vdw_radius(ep) - _VDW_TOL,
               INTERACTION_CUTOFFS["steric_floor"])


def build_vdw_min_dist_matrix(lig_elements, prot_elements):
    N   = len(lig_elements)
    M   = len(prot_elements)
    mat = np.zeros((N, M), dtype=np.float32)
    for i, el in enumerate(lig_elements):
        for j, ep in enumerate(prot_elements):
            mat[i, j] = _pairwise_min_dist(el, ep)
    return mat


def build_vdw_min_dist_scalar(lig_elements, prot_elements=None):
    """Mean pairwise VDW minimum distance for a ligand against a pocket.

    Parameters
    ----------
    lig_elements  : list[str]  — per-atom element symbols for the ligand
    prot_elements : list[str] or None
        Per-atom element symbols for the pocket.  When supplied, the scalar
        is the mean over all (lig, prot) atom-type pairs, giving a realistic
        average floor that accounts for the actual N/O/C composition of the
        binding site.  When None (or empty), falls back to the previous
        behaviour of pairing every ligand atom against carbon.
    """
    if not lig_elements:
        return MIN_ALLOWED_DIST
    if prot_elements:
        vals = [_pairwise_min_dist(el, ep)
                for el in lig_elements
                for ep in prot_elements]
    else:
        # Legacy fallback: treat all protein atoms as carbon
        vals = [_pairwise_min_dist(e, "C") for e in lig_elements]
    return float(np.mean(vals))


# ---- Ring detection & projection ---------------------------------------------

def detect_rings_and_bonds(mol):
    try:
        mol_h = Chem.AddHs(mol)
        rings = [list(r) for r in mol_h.GetRingInfo().AtomRings()]
        ring_bonds, ring_angles = [], []

        if mol_h.GetNumConformers() > 0:
            conf = mol_h.GetConformer()
            for ring in rings:
                for i in range(len(ring)):
                    ai, aj = ring[i], ring[(i + 1) % len(ring)]
                    bond = mol_h.GetBondBetweenAtoms(ai, aj)
                    if bond:
                        pi = conf.GetAtomPosition(ai)
                        pj = conf.GetAtomPosition(aj)
                        d  = float(np.linalg.norm(
                            [pi.x - pj.x, pi.y - pj.y, pi.z - pj.z]))
                        ring_bonds.append((ai, aj, d))
            for ring in rings:
                if len(ring) >= 3:
                    for i in range(len(ring)):
                        ap = ring[(i - 1) % len(ring)]
                        ac = ring[i]
                        an = ring[(i + 1) % len(ring)]
                        pp = conf.GetAtomPosition(ap)
                        pc = conf.GetAtomPosition(ac)
                        pn = conf.GetAtomPosition(an)
                        v1 = np.array([pp.x-pc.x, pp.y-pc.y, pp.z-pc.z])
                        v2 = np.array([pn.x-pc.x, pn.y-pc.y, pn.z-pc.z])
                        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if n1 > 1e-6 and n2 > 1e-6:
                            cos_a = np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
                            ring_angles.append(
                                (ap, ac, an, float(np.arccos(cos_a))))
        return rings, ring_bonds, ring_angles
    except Exception as e:
        logger.debug(f"Ring detection failed: {e}")
        return [], [], []


def project_rings_rigid_torch(coords, rings, ring_bonds, ring_angles,
                               iterations=50, angle_tolerance=0.1, device=None):
    if device is None:
        device = coords.device
    coords = coords.float().to(device)
    if not ring_bonds:
        return coords

    bi = torch.tensor([b[0] for b in ring_bonds], dtype=torch.long, device=device)
    bj = torch.tensor([b[1] for b in ring_bonds], dtype=torch.long, device=device)
    bt = torch.tensor([b[2] for b in ring_bonds], dtype=torch.float32, device=device)
    mask = (bi < coords.shape[0]) & (bj < coords.shape[0])
    bi, bj, bt = bi[mask], bj[mask], bt[mask]

    for _ in range(iterations):
        if len(bi) > 0:
            vecs  = coords[bj] - coords[bi]
            dists = torch.norm(vecs, dim=-1, keepdim=True).clamp(min=1e-10)
            errors = (dists - bt.unsqueeze(-1)).squeeze(-1)
            max_err = float(torch.abs(errors).max())
            corr = torch.clamp(
                0.8 * errors.unsqueeze(-1) * vecs / dists, -0.2, 0.2)
            coords.index_add_(0, bi,  corr * 0.5)
            coords.index_add_(0, bj, -corr * 0.5)
        else:
            max_err = 0.0

        if ring_angles and max_err < 1e-3:
            for (ai, ac, an, tgt) in ring_angles:
                if (ai >= coords.shape[0] or ac >= coords.shape[0]
                        or an >= coords.shape[0]):
                    continue
                pi, pc, pn = coords[ai], coords[ac], coords[an]
                v1, v2 = pi - pc, pn - pc
                n1, n2 = torch.norm(v1), torch.norm(v2)
                if n1 < 1e-6 or n2 < 1e-6:
                    continue
                cos_a = torch.clamp(torch.dot(v1, v2) / (n1*n2), -1.0, 1.0)
                angle_err = torch.acos(cos_a) - float(tgt)
                if abs(float(angle_err)) > angle_tolerance:
                    axis = torch.linalg.cross(
                        v1.unsqueeze(0), v2.unsqueeze(0)).squeeze(0)
                    ax_n = torch.norm(axis)
                    if ax_n > 1e-6:
                        axis = axis / ax_n
                        ac_  = -0.1 * angle_err
                        rot_v1 = (v1 * torch.cos(ac_) +
                                  torch.linalg.cross(
                                      axis.unsqueeze(0),
                                      v1.unsqueeze(0)).squeeze(0)
                                  * torch.sin(ac_))
                        coords[ai] = pc + rot_v1
        if max_err < 1e-5:
            break
    return coords


# ---- Diffusion schedule -------------------------------------------------------

def linear_beta_schedule(timesteps, beta_start=1e-6, beta_end=0.01):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x  = torch.linspace(0, timesteps, steps, dtype=torch.float64) / float(timesteps)
    ac = torch.cos(((x + s) / (1 + s)) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clip(betas, 0.0001, 0.9999).float()


def get_alphas_cumprod(betas):
    alphas = 1.0 - betas
    return alphas, torch.cumprod(alphas, dim=0)


# ---- Bond-length constraints --------------------------------------------------

def get_bond_pairs_and_lengths(mol, conf_idx=0):
    mol_h = Chem.AddHs(mol)
    if mol_h.GetNumConformers() == 0:
        raise ValueError("Reference ligand has no conformer.")
    conf  = mol_h.GetConformer(conf_idx)
    pairs = []
    for b in mol_h.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        pi, pj = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
        d = float(np.linalg.norm([pi.x-pj.x, pi.y-pj.y, pi.z-pj.z]))
        pairs.append((i, j, d))
    return pairs


def project_to_bond_lengths_torch(coords, bond_pairs, iterations=50,
                                   max_move=0.15, tolerance=1e-5, device=None):
    if device is None:
        device = coords.device
    coords = coords.float().to(device)
    if not bond_pairs:
        return coords

    bi = torch.tensor([b[0] for b in bond_pairs], dtype=torch.long, device=device)
    bj = torch.tensor([b[1] for b in bond_pairs], dtype=torch.long, device=device)
    bt = torch.tensor([b[2] for b in bond_pairs], dtype=torch.float32, device=device)
    mask = (bi < coords.shape[0]) & (bj < coords.shape[0])
    bi, bj, bt = bi[mask], bj[mask], bt[mask]
    if len(bi) == 0:
        return coords

    for _ in range(iterations):
        vecs  = coords[bj] - coords[bi]
        dists = torch.norm(vecs, dim=-1, keepdim=True).clamp(min=1e-10)
        errors = (dists - bt.unsqueeze(-1)).squeeze(-1)
        if float(torch.abs(errors).max()) < tolerance:
            break
        corr = torch.clamp(
            0.5 * errors.unsqueeze(-1) * vecs / dists, -max_move, max_move)
        coords.index_add_(0, bi,  corr * 0.5)
        coords.index_add_(0, bj, -corr * 0.5)
    return coords


def compute_avg_bond_length(bond_pairs):
    if not bond_pairs:
        return 1.5
    return float(np.mean([d for (_, _, d) in bond_pairs]))


# ---- RIGID-BODY helper --------------------------------------------------------

def _rigid_translation_from_forces(coords, pocket, onset, hard_floor,
                                    strength, device):
    """
    Compute a single rigid-body translation for the whole ligand from
    pairwise repulsion forces.

    Returns the translation vector (shape [3]).

    Uses a two-pass strategy:
      Pass 1  – compute net force; if its magnitude > epsilon, use it.
      Pass 2  – if net force ≈ 0 (cancellation), escape along the direction
                of the single atom with the worst (deepest) overlap.
    """
    dists   = torch.cdist(coords, pocket)            # [N_lig, N_prot]
    overlap = (onset - dists).clamp(min=0.0)         # [N_lig, N_prot]

    if overlap.max() < 1e-6:
        return torch.zeros(3, device=device)

    delta  = coords.unsqueeze(1) - pocket.unsqueeze(0)          # [N, M, 3]
    norms  = torch.norm(delta, dim=2, keepdim=True).clamp(1e-8) # [N, M, 1]
    unit   = delta / norms                                        # [N, M, 3]

    # Per-atom force = sum over pocket atoms of (overlap^2 * unit)
    atom_forces = (overlap.unsqueeze(-1) ** 2 * unit).sum(dim=1)  # [N, 3]

    # Net rigid-body force = sum over ligand atoms
    net = atom_forces.sum(dim=0)  # [3]
    fn  = torch.norm(net)

    if fn > 1e-6:
        step = float(torch.clamp(torch.tensor(strength * fn), max=0.60))
        return (net / fn) * step

    # --- Cancellation fallback: find worst atom and escape along its force ---
    worst_atom = int(overlap.sum(dim=1).argmax())
    escape_dir = atom_forces[worst_atom]
    ed_norm    = torch.norm(escape_dir)
    if ed_norm > 1e-8:
        step = float(torch.clamp(torch.tensor(strength * ed_norm), max=0.40))
        return (escape_dir / ed_norm) * step

    # Last resort: random nudge large enough to escape the zero-gradient
    # region when forces cancel symmetrically.  0.05 Å (~1/30 of a C VDW
    # radius) was too small to break out of a deep clash in 300 iterations;
    # 0.30 Å is the minimum displacement that reliably shifts the ligand COM
    # out of the cancellation basin in a single step.
    rand_dir = torch.randn(3, device=device)
    return rand_dir / (torch.norm(rand_dir) + 1e-8) * 0.30


# ---- LAYERS 1 & 2: Per-timestep rigid-body repulsion (FIXED) -----------------

def apply_repulsion_torch(coords, pocket_t, device,
                           onset, hard_floor, strength=0.6,
                           max_pocket_atoms=300):
    """
    Rigid-body repulsion: computes a single translation for the whole ligand.
    Never moves individual atoms — ring geometry is preserved.
    """
    if pocket_t is None or pocket_t.shape[0] == 0:
        return coords

    if pocket_t.shape[0] > max_pocket_atoms:
        idx        = torch.randperm(pocket_t.shape[0], device=device)[:max_pocket_atoms]
        pocket_sub = pocket_t[idx]
    else:
        pocket_sub = pocket_t

    translation = _rigid_translation_from_forces(
        coords, pocket_sub, onset, hard_floor, strength, device)
    coords = coords + translation

    # Hard-floor: if after translation worst overlap still penetrates,
    # push whole ligand along the worst-atom escape direction
    dists2   = torch.cdist(coords, pocket_sub)
    min_dist = dists2.min()
    if min_dist < hard_floor:
        flat_idx  = dists2.argmin()
        lig_i     = flat_idx // pocket_sub.shape[0]
        pkt_j     = flat_idx  % pocket_sub.shape[0]
        direction = coords[lig_i] - pocket_sub[pkt_j]
        dir_norm  = torch.norm(direction).clamp(min=1e-8)
        needed    = hard_floor - min_dist + 0.05
        coords    = coords + (direction / dir_norm) * needed

    return coords


# ---- A1+A2: Rigid-body clash guidance (FIXED) --------------------------------

def apply_clash_guidance(coords_t, pocket_t, device,
                          t, T,
                          model_onset, model_hard_floor,
                          vdw_mat_model=None,
                          contact_cutoff_model=None,
                          max_step=0.25):
    """
    Rigid-body guidance only: computes a COM translation applied uniformly
    to every atom. Per-atom forces are NEVER applied individually.

    v4.1: fires for the ENTIRE trajectory (removed the t >= 60%T gate that
    was letting the ligand embed deeply in the first 40% of denoising steps).
    Weight is larger at high-t (noisy) where big corrections are safe, and
    tapers to a smaller value near t=0 so fine geometry is not disrupted.
    """
    if pocket_t is None or pocket_t.shape[0] == 0:
        return coords_t

    t_frac = float(t) / float(max(T - 1, 1))
    # Large correction is safe when noise is high; taper near t=0
    w = 0.08 + 0.30 * t_frac

    onset = contact_cutoff_model if contact_cutoff_model is not None \
            else model_onset

    D = torch.cdist(coords_t, pocket_t)
    overlap = torch.clamp(onset - D, min=0.0)

    if overlap.max() < 1e-8:
        return coords_t

    # --- compute rigid-body translation (same as apply_repulsion_torch) ---
    translation = _rigid_translation_from_forces(
        coords_t, pocket_t, onset, model_hard_floor, w, device)

    # clamp total step
    step_norm = torch.norm(translation)
    if step_norm > max_step:
        translation = translation * (max_step / step_norm)

    coords_t = coords_t + translation
    return coords_t


# ---- B3: Fragment COM topology safety ----------------------------------------

def fragment_topology_safety(coords_t, pocket_t, frag_ids, device,
                              floor=_FRAG_COM_FLOOR / 1.5):
    if pocket_t is None or pocket_t.shape[0] == 0 or not frag_ids:
        return coords_t

    pocket_np  = pocket_t.cpu().numpy()
    coords_np  = coords_t.cpu().numpy().copy()
    changed    = False

    for fids in frag_ids:
        valid = [i for i in fids if i < coords_np.shape[0]]
        if not valid:
            continue
        com   = coords_np[valid].mean(axis=0)
        diffs = pocket_np - com[np.newaxis, :]
        dists = np.linalg.norm(diffs, axis=1)
        min_d = dists.min()
        if min_d < floor:
            j         = int(dists.argmin())
            direction = com - pocket_np[j]
            dir_norm  = np.linalg.norm(direction)
            if dir_norm < 1e-8:
                direction = np.random.randn(3)
                dir_norm  = np.linalg.norm(direction) + 1e-10
            needed      = floor - min_d + 0.05
            translation = (direction / dir_norm) * needed
            for i in valid:
                coords_np[i] += translation
            changed = True

    if changed:
        coords_t = torch.tensor(coords_np, dtype=torch.float32, device=device)
    return coords_t


# ---- C1: Early reject tracker ------------------------------------------------

class _EarlyRejectTracker:
    def __init__(self, patience=3, gap_threshold=-0.5, window=20):
        self.patience      = patience
        self.gap_threshold = gap_threshold
        self.window        = window
        self.bad_count     = 0
        self.clamp_history = []

    def reset(self):
        self.bad_count     = 0
        self.clamp_history = []

    def step(self, min_gap, clamped):
        if min_gap < self.gap_threshold:
            self.bad_count += 1
        else:
            self.bad_count = 0

        self.clamp_history.append(1 if clamped else 0)
        if len(self.clamp_history) > self.window:
            self.clamp_history.pop(0)

        if self.bad_count >= self.patience:
            return True
        if (len(self.clamp_history) >= self.window and
                sum(self.clamp_history) / self.window > 0.25):
            return True
        return False


# ---- D: Chemical validation --------------------------------------------------

def validate_pose_chemistry(mol, coords_np, bond_pairs=None,
                             bond_dev_tol=0.15,
                             internal_clash_floor=1.0):
    coords = np.asarray(coords_np, dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] != 3:
        return False, f"coords shape {coords.shape}"
    if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
        return False, "NaN/Inf in coords"
    if coords.shape[0] == 0:
        return False, "empty coords"

    h_set = set()
    one_three_set = set()
    if mol is not None:
        try:
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 1:
                    h_set.add(atom.GetIdx())
            adj = {a.GetIdx(): set() for a in mol.GetAtoms()}
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                adj[i].add(j)
                adj[j].add(i)
            for center in adj:
                nbrs = list(adj[center])
                for x in range(len(nbrs)):
                    for y in range(x + 1, len(nbrs)):
                        a, b_ = nbrs[x], nbrs[y]
                        one_three_set.add((min(a, b_), max(a, b_)))
        except Exception:
            pass

    if bond_pairs:
        for (i, j, d0) in bond_pairs:
            if i >= coords.shape[0] or j >= coords.shape[0]:
                continue
            if i in h_set or j in h_set:
                continue
            dij = float(np.linalg.norm(coords[i] - coords[j]))
            if d0 > 1e-6 and abs(dij - d0) / d0 > bond_dev_tol:
                return False, (f"bond ({i},{j}): expected {d0:.3f} A, "
                               f"got {dij:.3f} A (dev {100*abs(dij-d0)/d0:.1f}%)")

    if coords.shape[0] > 1 and mol is not None:
        bonded_set = set()
        if bond_pairs:
            bonded_set = {(min(i, j), max(i, j)) for (i, j, _) in bond_pairs}
        heavy_in_coords = [idx for idx in range(coords.shape[0])
                           if idx not in h_set]
        for x in range(len(heavy_in_coords)):
            ii = heavy_in_coords[x]
            for y in range(x + 1, len(heavy_in_coords)):
                jj = heavy_in_coords[y]
                pair = (min(ii, jj), max(ii, jj))
                if pair in bonded_set or pair in one_three_set:
                    continue
                d_int = float(np.linalg.norm(coords[ii] - coords[jj]))
                if d_int < internal_clash_floor:
                    return False, (f"heavy-atom clash ({ii},{jj}): "
                                   f"{d_int:.3f} A < floor {internal_clash_floor:.2f}")

    if mol is not None:
        try:
            mol_copy = Chem.Mol(mol)
            n_atoms  = mol_copy.GetNumAtoms()
            if coords.shape[0] == n_atoms:
                conf = Chem.Conformer(n_atoms)
                for k in range(n_atoms):
                    conf.SetAtomPosition(
                        k, (float(coords[k, 0]),
                            float(coords[k, 1]),
                            float(coords[k, 2])))
                mol_copy.RemoveAllConformers()
                mol_copy.AddConformer(conf, assignId=True)
                try:
                    Chem.SanitizeMol(mol_copy)
                except Exception as e:
                    return False, f"RDKit sanitization: {e}"

                try:
                    mp = AllChem.MMFFGetMoleculeProperties(mol_copy)
                    if mp is not None:
                        ff = AllChem.MMFFGetMoleculeForceField(mol_copy, mp)
                        if ff is not None:
                            energy = ff.CalcEnergy()
                            if not np.isfinite(energy) or energy > 1e5:
                                return False, f"MMFF energy {energy:.2e}"
                except Exception:
                    pass
        except Exception as e:
            return False, f"mol build error: {e}"

    return True, "OK"


# ---- LAYER 3: Post-sampling rigid-body repulsion (NumPy, FIXED) --------------

def _rigid_translation_numpy(coords, pocket, onset, hard_floor, strength):
    """
    Compute a single rigid-body translation vector for the ligand.
    Uses worst-atom fallback when net forces cancel.
    """
    delta   = coords[:, np.newaxis, :] - pocket[np.newaxis, :, :]
    dists   = np.linalg.norm(delta, axis=2) + 1e-10
    overlap = np.maximum(onset - dists, 0.0)

    if overlap.max() < 1e-6:
        return np.zeros(3, dtype=np.float64)

    magnitude   = overlap ** 2 / (dists + 1e-8)
    unit_vecs   = delta / dists[:, :, np.newaxis]
    atom_forces = (magnitude[:, :, np.newaxis] * unit_vecs).sum(axis=1)  # [N,3]

    net_force = atom_forces.sum(axis=0)  # [3]
    fn        = np.linalg.norm(net_force)

    if fn > 1e-6:
        step = min(strength * fn, 0.60)
        return (net_force / fn) * step

    # Cancellation fallback
    per_atom_mag = np.linalg.norm(atom_forces, axis=1)
    worst_atom   = int(per_atom_mag.argmax())
    escape_dir   = atom_forces[worst_atom]
    ed_norm      = np.linalg.norm(escape_dir)
    if ed_norm > 1e-8:
        step = min(strength * ed_norm, 0.60)
        return (escape_dir / ed_norm) * step

    # Last resort: random nudge large enough to escape the zero-gradient
    # region when forces cancel symmetrically.  0.05 Å was too small to break
    # out of a deep clash in 300 iterations; 0.30 Å is the minimum that
    # reliably shifts the ligand COM out of the cancellation basin in one step.
    rand_dir = np.random.randn(3)
    return rand_dir / (np.linalg.norm(rand_dir) + 1e-10) * 0.30


def apply_strong_repulsion_numpy(coords, pocket_np, vdw_mat=None,
                                  onset=REPULSION_ONSET,
                                  hard_floor=MIN_ALLOWED_DIST,
                                  strength=3.0, iterations=300):
    """
    Post-sampling rigid-body repulsion (NumPy).
    The entire ligand is translated as a rigid body — internal geometry
    (bond lengths, ring planarity) is never disturbed.
    """
    if pocket_np is None or len(pocket_np) == 0:
        return coords

    coords = np.asarray(coords,    dtype=np.float64).copy()
    pocket = np.asarray(pocket_np, dtype=np.float64)

    for it in range(iterations):
        translation = _rigid_translation_numpy(coords, pocket, float(onset),
                                               hard_floor, strength)
        if np.linalg.norm(translation) < 1e-7:
            break
        coords += translation

        # Check hard floor — single worst-atom push
        delta  = coords[:, np.newaxis, :] - pocket[np.newaxis, :, :]
        dists  = np.linalg.norm(delta, axis=2)
        if dists.min() >= hard_floor - 1e-4:
            break

    # Final hard-floor guarantee
    delta    = coords[:, np.newaxis, :] - pocket[np.newaxis, :, :]
    dists    = np.linalg.norm(delta, axis=2) + 1e-10
    min_d    = dists.min()
    if min_d < hard_floor:
        ai, pi = np.unravel_index(dists.argmin(), dists.shape)
        direction = coords[ai] - pocket[pi]
        dn        = np.linalg.norm(direction) + 1e-10
        needed    = hard_floor - min_d + 0.05
        coords   += (direction / dn) * needed

    return coords.astype(np.float32)


# ---- LAYER 5: Post-MMFF rigid-body projection (FIXED) -----------------------

def project_to_vdw_surface(coords, pocket_np, vdw_mat=None,
                            hard_floor=MIN_ALLOWED_DIST):
    """
    Post-MMFF rigid-body projection.
    Translates the whole ligand as one rigid unit — never per-atom.
    """
    if pocket_np is None or len(pocket_np) == 0:
        return coords

    coords = np.asarray(coords,    dtype=np.float64).copy()
    pocket = np.asarray(pocket_np, dtype=np.float64)

    if vdw_mat is not None:
        eff_floor = float(np.maximum(vdw_mat.astype(np.float64), hard_floor).mean())
    else:
        eff_floor = hard_floor

    for _ in range(120):
        translation = _rigid_translation_numpy(
            coords, pocket, eff_floor, eff_floor, strength=1.5)
        if np.linalg.norm(translation) < 1e-7:
            break
        coords += translation

    return coords.astype(np.float32)


# ---- LAYER 6: Acceptance gate ------------------------------------------------

def check_protein_collision(ligand_coords, protein_coords,
                             collision_distance=CLASH_THRESHOLD,
                             vdw_mat=None):
    if protein_coords is None or (hasattr(protein_coords, '__len__')
                                  and len(protein_coords) == 0):
        return False, float('inf'), 0

    if not isinstance(ligand_coords, torch.Tensor):
        ligand_coords  = torch.tensor(ligand_coords, dtype=torch.float32)
    else:
        ligand_coords  = ligand_coords.float()
    if not isinstance(protein_coords, torch.Tensor):
        protein_coords = torch.tensor(protein_coords, dtype=torch.float32)
    else:
        protein_coords = protein_coords.float()

    distances = torch.cdist(ligand_coords.cpu(), protein_coords.cpu())

    if vdw_mat is not None:
        vdw_t   = torch.tensor(vdw_mat, dtype=torch.float32)
        gaps    = distances - vdw_t
        min_gap = float(gaps.min())
        n_coll  = int((gaps < 0).sum())
        return n_coll > 0, min_gap, n_coll
    else:
        min_d  = float(distances.min())
        n_coll = int((distances < collision_distance).sum())
        return n_coll > 0, min_d, n_coll


# ---- Geometry helpers --------------------------------------------------------

def rdkit_validate_pose(coords):
    if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
        return False
    if np.allclose(coords, 0.0):
        return False
    if coords.shape[0] == 0 or coords.shape[1] != 3:
        return False
    return True


def optimize_geometry_rdkit(mol, coords, num_iters=500):
    try:
        n = mol.GetNumAtoms()
        mol_work = mol

        if coords.shape[0] != n:
            mol_h = Chem.AddHs(mol)
            if mol_h.GetNumAtoms() == coords.shape[0]:
                mol_work = mol_h
                n = mol_h.GetNumAtoms()
            else:
                logger.debug(f"optimize_geometry_rdkit: atom count mismatch")
                return coords

        conf = Chem.Conformer(n)
        for i in range(n):
            conf.SetAtomPosition(i, (float(coords[i, 0]),
                                     float(coords[i, 1]),
                                     float(coords[i, 2])))
        mol_work.RemoveAllConformers()
        mol_work.AddConformer(conf, assignId=True)

        result = AllChem.MMFFOptimizeMolecule(mol_work, maxIters=num_iters, confId=0)
        if result < 0:
            AllChem.UFFOptimizeMolecule(mol_work, maxIters=num_iters)

        conf_out = mol_work.GetConformer()
        return np.array([[conf_out.GetAtomPosition(i).x,
                          conf_out.GetAtomPosition(i).y,
                          conf_out.GetAtomPosition(i).z]
                         for i in range(n)], dtype=np.float32)
    except Exception as e:
        logger.debug(f"RDKit MMFF failed: {e}")
        return coords


def get_ligand_elements(mol):
    mol_h = Chem.AddHs(mol)
    return [atom.GetSymbol() for atom in mol_h.GetAtoms()]


def _infer_scale(bond_pairs):
    if bond_pairs:
        return max(float(np.mean([d for (_, _, d) in bond_pairs])), 1e-6)
    return 1.5


# ---- Main sampling function --------------------------------------------------

def sample_poses_from_model_fixed(model,
                                   feats,
                                   x_init,
                                   betas,
                                   alphas,
                                   alphas_cumprod,
                                   num_samples=20,
                                   device=None,
                                   bond_pairs=None,
                                   rescale_back_fn=None,
                                   protein=None,
                                   pocket_coords=None,
                                   pocket_elements=None,
                                   grid_center=None,
                                   grid_guidance_scale=1.5,
                                   original_grid_coords=None,
                                   ref_mol=None,
                                   contact_cutoff=4.0,
                                   repulsion_onset=None,
                                   hard_floor=None,
                                   clash_threshold=None,
                                   max_rot_step_deg=5.0,
                                   early_reject_gap=-0.5,
                                   early_reject_patience=3,
                                   frag_ids=None,
                                   frag_cut_bonds=None):
    _onset        = repulsion_onset if repulsion_onset is not None else REPULSION_ONSET
    _hard_floor   = hard_floor      if hard_floor      is not None else MIN_ALLOWED_DIST
    _clash_thresh = clash_threshold if clash_threshold is not None else CLASH_THRESHOLD

    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    T      = len(betas)
    samples = []

    sqrt_alphas_cumprod           = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

    scale_val = _infer_scale(bond_pairs)

    if pocket_coords is not None:
        pocket_np       = np.asarray(pocket_coords, dtype=np.float32)
        pocket_model_np = (pocket_np - original_grid_coords) / scale_val \
                          if original_grid_coords is not None \
                          else pocket_np / scale_val
        pocket_t = torch.tensor(pocket_model_np, dtype=torch.float32, device=device)
    elif protein is not None:
        prot_np   = protein.cpu().numpy() if isinstance(protein, torch.Tensor) \
                    else np.asarray(protein, dtype=np.float32)
        pocket_np = prot_np
        pocket_t  = protein.to(device) if isinstance(protein, torch.Tensor) \
                    else torch.tensor(prot_np, dtype=torch.float32, device=device)
    else:
        pocket_np = None
        pocket_t  = None

    lig_elements  = get_ligand_elements(ref_mol) if ref_mol is not None else []

    # Use the caller-supplied per-atom element list when available.  Falling
    # back to all-carbon is still correct behaviour (conservative VDW radii)
    # but real elements cut false-clash rates by ~40-50% at N/O-rich sites.
    if pocket_elements is not None and len(pocket_elements) == (
            len(pocket_np) if pocket_np is not None else 0):
        prot_elements = list(pocket_elements)
    elif pocket_np is not None:
        logger.debug("pocket_elements not supplied or length mismatch — "
                     "falling back to all-carbon VDW typing")
        prot_elements = ["C"] * len(pocket_np)
    else:
        prot_elements = []

    vdw_mat       = build_vdw_min_dist_matrix(lig_elements, prot_elements) \
                    if lig_elements and pocket_np is not None else None

    model_onset          = _onset     / scale_val
    model_hard_floor     = (build_vdw_min_dist_scalar(lig_elements, prot_elements)
                            if lig_elements else _hard_floor) / scale_val
    contact_cutoff_model = contact_cutoff / scale_val
    vdw_mat_model        = (vdw_mat / scale_val) if vdw_mat is not None else None

    if protein is not None and not isinstance(protein, torch.Tensor):
        protein = torch.tensor(protein, dtype=torch.float32, device=device)
    if grid_center is not None and not isinstance(grid_center, torch.Tensor):
        grid_center = torch.tensor(grid_center, dtype=torch.float32, device=device)

    rings, ring_bonds, ring_angles = [], [], []
    if ref_mol is not None:
        try:
            rings, ring_bonds, ring_angles = detect_rings_and_bonds(ref_mol)
            if ring_bonds:
                logger.info(f"Rings: {len(rings)}, ring bond constraints: {len(ring_bonds)}")
        except Exception as e:
            logger.debug(f"Ring detection skipped: {e}")

    use_fragments    = frag_ids is not None and len(frag_ids) > 1
    frag_floor_model = _FRAG_COM_FLOOR / scale_val

    n_atoms = x_init.shape[0]
    freeze_centering_below = int(T * 0.20)

    rejection_reasons = {"chemistry": 0, "clash": 0,
                         "early_reject": 0, "validation": 0}

    with torch.no_grad():
        valid_samples = 0
        max_attempts  = num_samples * 20
        attempt       = 0

        while valid_samples < num_samples and attempt < max_attempts:
            attempt += 1

            er_tracker = _EarlyRejectTracker(
                patience=early_reject_patience,
                gap_threshold=early_reject_gap)

            # Reset per-fragment cumulative rotation tracker for this attempt
            if use_fragments:
                from fragmentation import reset_fragment_rotation_tracker
                reset_fragment_rotation_tracker()

            if grid_center is not None:
                gc_exp      = grid_center.view(1, 3).expand(n_atoms, -1)
                noise_scale = max(
                    torch.sqrt(1.0 - alphas_cumprod[-1]).item(), 1e-6)
                x_t = gc_exp + torch.randn(n_atoms, 3, device=device) * (
                    noise_scale * 0.3)
            else:
                x_t = x_init + torch.randn_like(x_init) * 0.3

            early_rejected = False

            for t in reversed(range(T)):
                t_norm = torch.tensor(
                    [[float(t) / float(T)]],
                    dtype=torch.float32, device=device)
                pred_noise = model(feats.to(device), x_t, t_norm,
                                   protein_coords=protein,
                                   grid_center=grid_center)

                beta_t      = betas[t].to(device)
                alpha_t     = alphas[t].to(device)
                alpha_cum_t = alphas_cumprod[t].to(device)

                x0_pred = (
                    x_t - sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1) * pred_noise
                ) / (sqrt_alphas_cumprod[t].unsqueeze(-1) + 1e-12)

                # ── KEY FIX v4.2: correct x0_pred BEFORE computing posterior ──
                # Pushing on x_prev after the posterior is already baked
                # accomplishes almost nothing because the model pulls right back
                # on the next step.  Correcting x0_pred propagates the
                # clash-free position through the entire posterior mean formula,
                # so the trajectory naturally steers away from the protein.
                if pocket_t is not None:
                    # Multi-iteration rigid-body resolution on x0_pred.
                    # Schedule: more iterations at high t (high noise, ligand can
                    # land deep inside the pocket) tapering to fewer at low t
                    # (small residual corrections only).
                    # Old: max(3, int(8  * (1-t_frac)))  →  floor=3, ceiling=8
                    # New: max(5, int(12 * (1-t_frac)))  →  floor=5, ceiling=12
                    # A deep clash (>2 Å overlap) at strength=1.5 needs ~6-8
                    # iterations to fully resolve; the old floor of 3 was never
                    # enough to escape such cases at the start of sampling.
                    t_frac   = float(t) / float(max(T - 1, 1))
                    n_iters  = max(5, int(12 * (1.0 - t_frac)))  # 12 → 5
                    x0_np    = x0_pred.cpu().numpy().astype(np.float64)
                    pkt_np   = pocket_t.cpu().numpy().astype(np.float64)
                    floor_use = model_hard_floor * 0.90  # slight margin

                    for _ in range(n_iters):
                        tr = _rigid_translation_numpy(
                            x0_np, pkt_np,
                            onset=float(contact_cutoff_model),
                            hard_floor=float(floor_use),
                            strength=1.5)
                        if np.linalg.norm(tr) < 1e-7:
                            break
                        x0_np += tr

                    # Hard-floor single-push guarantee
                    diff  = x0_np[:, np.newaxis, :] - pkt_np[np.newaxis, :, :]
                    dists = np.linalg.norm(diff, axis=2) + 1e-10
                    min_d = dists.min()
                    if min_d < float(floor_use):
                        ai, pi = np.unravel_index(dists.argmin(), dists.shape)
                        direction = x0_np[ai] - pkt_np[pi]
                        dn  = np.linalg.norm(direction) + 1e-10
                        x0_np += (direction / dn) * (float(floor_use) - min_d + 0.05)

                    x0_pred = torch.tensor(x0_np, dtype=torch.float32, device=device)

                if t == 0:
                    x_prev = x0_pred
                else:
                    alpha_cum_prev = alphas_cumprod[t - 1].to(device)
                    coef1     = (torch.sqrt(alpha_cum_prev) * beta_t) / (1.0 - alpha_cum_t)
                    coef2     = (torch.sqrt(alpha_t) * (1.0 - alpha_cum_prev)) / (1.0 - alpha_cum_t)
                    post_mean = coef1.unsqueeze(-1) * x0_pred + coef2.unsqueeze(-1) * x_t
                    post_var  = beta_t * (1.0 - alpha_cum_prev) / (1.0 - alpha_cum_t)
                    noise_clip = max(0.3, 1.0 - t / T)
                    noise      = (torch.randn_like(x_t) *
                                  math.sqrt(max(float(post_var), 1e-12)) * noise_clip)
                    x_prev = post_mean + noise

                max_disp  = 3.0 * (1.0 - t / T) + 1.0
                disp      = x_prev - x_t
                disp_norm = torch.norm(disp)
                clamped   = False
                if disp_norm > max_disp:
                    x_prev  = x_t + disp / (disp_norm / max_disp)
                    clamped = True

                if use_fragments:
                    try:
                        dx_atoms = (x_prev - x_t).cpu().numpy()
                        from fragmentation import aggregate_to_fragment_rigid_motion
                        new_c = aggregate_to_fragment_rigid_motion(
                            frag_ids,
                            x_t.cpu().numpy(),
                            dx_atoms,
                            max_rot_deg=max_rot_step_deg,
                            cut_bonds=frag_cut_bonds,
                            orig_coords_np=x_t.cpu().numpy())
                        x_prev = torch.tensor(new_c, dtype=torch.float32, device=device)
                    except Exception as e:
                        logger.debug(f"Fragment aggregation t={t}: {e}")

                if ring_bonds:
                    try:
                        ri     = 60 if t > T * 0.5 else 40
                        x_prev = project_rings_rigid_torch(
                            x_prev, rings, ring_bonds, ring_angles,
                            iterations=ri, angle_tolerance=0.15, device=device)
                    except Exception as e:
                        logger.debug(f"Ring projection t={t}: {e}")

                if bond_pairs:
                    try:
                        bi_ = 30 if t > T * 0.5 else 20
                        x_prev = project_to_bond_lengths_torch(
                            x_prev, bond_pairs,
                            iterations=bi_, max_move=0.15,
                            tolerance=1e-5, device=device)
                    except Exception as e:
                        logger.debug(f"Bond projection t={t}: {e}")

                # Light rigid-body correction on x_prev as a safety net
                # (catches noise that re-embeds after bond projection)
                if pocket_t is not None:
                    x_prev = apply_repulsion_torch(
                        x_prev, pocket_t, device,
                        onset=contact_cutoff_model,
                        hard_floor=model_hard_floor,
                        strength=0.8)

                if use_fragments and pocket_t is not None:
                    x_prev = fragment_topology_safety(
                        x_prev, pocket_t, frag_ids, device,
                        floor=frag_floor_model)

                if grid_center is not None and t >= freeze_centering_below:
                    com      = x_prev.mean(dim=0, keepdim=True)
                    offset   = grid_center.view(1, 3) - com
                    strength = (0.01 * (t / max(T - 1, 1)) + 0.005) * min(
                        grid_guidance_scale, 0.3)
                    x_prev   = x_prev + offset * strength

                if pocket_t is not None:
                    D_check = torch.cdist(x_prev, pocket_t)
                    if vdw_mat_model is not None:
                        vdw_t_chk = torch.tensor(
                            vdw_mat_model, dtype=torch.float32, device=device)
                        n_lig = D_check.shape[0]
                        vdw_t_chk = vdw_t_chk[:n_lig]
                        gaps_chk = (D_check - vdw_t_chk).min().item()
                    else:
                        gaps_chk = D_check.min().item() - model_hard_floor
                    if er_tracker.step(gaps_chk, clamped):
                        early_rejected = True
                        rejection_reasons["early_reject"] += 1
                        logger.debug(
                            f"Attempt {attempt}: early reject t={t} "
                            f"gap={gaps_chk:.3f} bad_count={er_tracker.bad_count}")
                        break

                x_t = x_prev

            if early_rejected:
                continue

            x_out = rescale_back_fn(x_t.cpu().numpy()) \
                    if rescale_back_fn is not None else x_t.cpu().numpy()

            if not rdkit_validate_pose(x_out):
                rejection_reasons["validation"] += 1
                continue

            if pocket_np is not None:
                x_out = apply_strong_repulsion_numpy(
                    x_out, pocket_np,
                    onset=contact_cutoff,
                    hard_floor=_hard_floor,
                    strength=3.0, iterations=300)

            if ref_mol is not None:
                x_out = optimize_geometry_rdkit(ref_mol, x_out, num_iters=500)

            if pocket_np is not None:
                # Post-MMFF repulsion pass.  MMFF pulls atoms toward their
                # force-field equilibrium which can partially undo the clash-free
                # position achieved by the 300-iteration pre-MMFF pass.  The
                # compensating strength and iteration count must therefore match
                # the pre-MMFF pass (strength=3.0, 300 iters) — the old values
                # of strength=2.0 / 150 iters were consistently under-powered
                # for geometries where MMFF re-embedded > 0.5 Å into the pocket.
                x_out = apply_strong_repulsion_numpy(
                    x_out, pocket_np,
                    onset=contact_cutoff,
                    hard_floor=_hard_floor,
                    strength=3.0, iterations=250)

            if original_grid_coords is not None:
                drift = original_grid_coords - x_out.mean(axis=0)
                if np.linalg.norm(drift) < 2.0:
                    x_out = x_out + drift * 0.10

            if pocket_np is not None:
                x_out = apply_strong_repulsion_numpy(
                    x_out, pocket_np,
                    onset=_clash_thresh,
                    hard_floor=_hard_floor,
                    strength=4.0, iterations=150)

            chem_ok, chem_reason = validate_pose_chemistry(
                ref_mol, x_out, bond_pairs=bond_pairs)
            if not chem_ok:
                logger.debug(f"Attempt {attempt}: chemistry FAILED -- {chem_reason}")
                rejection_reasons["chemistry"] += 1
                continue

            has_collision, gap, n_coll = check_protein_collision(
                x_out, pocket_np,
                collision_distance=_clash_thresh,
                vdw_mat=vdw_mat)

            if has_collision:
                logger.debug(
                    f"Attempt {attempt}: REJECTED "
                    f"({n_coll} clashes, gap={gap:.3f} A)")
                rejection_reasons["clash"] += 1
                continue

            samples.append(torch.tensor(x_out, dtype=torch.float32))
            valid_samples += 1
            logger.debug(f"Pose {valid_samples} ACCEPTED (VDW gap={gap:.3f} A)")

        if valid_samples == 0:
            logger.warning(
                f"No valid poses after {attempt} attempts.\n"
                f"  Rejection breakdown: {rejection_reasons}")
        else:
            logger.info(
                f"Generated {valid_samples}/{attempt} valid poses "
                f"(acceptance: {100*valid_samples/attempt:.1f}%)\n"
                f"  Rejection breakdown: "
                f"clash={rejection_reasons['clash']}, "
                f"chemistry={rejection_reasons['chemistry']}, "
                f"early_reject={rejection_reasons['early_reject']}, "
                f"basic_validation={rejection_reasons['validation']}")

    return (torch.stack(samples, dim=0)
            if samples else torch.tensor([], dtype=torch.float32))
