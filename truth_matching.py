"""Shared generator matching for conversion and truth diagnostics."""
import awkward as ak
import numpy as np
import vector

vector.register_awkward()


def electron_pt_mismatch(status, pt, gen_pt, max_relative_pt):
    """Identify prompt/direct-tau electrons failing the strict response requirement."""
    if not np.isfinite(max_relative_pt) or max_relative_pt <= 0:
        raise ValueError("electron_response_max_relative_pt must be finite and positive")
    return ((status == 1) | (status == 2)) & np.logical_not(
        (gen_pt > 0) & (abs(pt - gen_pt) < max_relative_pt * gen_pt))


def match_electron_truth(electrons, gen_particles, max_dr, electron_response_max_relative_pt=None):
    """Label nearest electrons: 0 unmatched, 1 prompt, 2 tau, 3 other, 5 bad response."""
    return match_candidate_truth(electrons, gen_particles, max_dr,
                                electron_response_max_relative_pt=electron_response_max_relative_pt)


def match_candidate_truth(candidates, gen_particles, max_dr, pdg_id=11, max_relative_pt=None,
                          electron_response_max_relative_pt=None):
    """Match final-state particles: photons use 0/1/3; electrons also have tau category 2.

    max_relative_pt filters matching candidates; electron response labeling never rematches.
    """
    if electron_response_max_relative_pt is not None and (pdg_id != 11 or max_relative_pt is not None):
        raise ValueError("Electron response labeling requires electrons and distance-only matching")
    gen_ele = gen_particles[(abs(gen_particles.pdgId) == pdg_id) & (gen_particles.status == 1)]
    reco_momentum = ak.zip({
        "pt": candidates.pt, "eta": candidates.eta,
        "phi": candidates.phi, "mass": candidates.m,
    }, with_name="Momentum4D")
    gen_momentum = ak.zip({
        "pt": gen_ele.pt, "eta": gen_ele.eta,
        "phi": gen_ele.phi, "mass": gen_ele.mass,
    }, with_name="Momentum4D")
    pairs = ak.cartesian({"ele": reco_momentum, "gen": gen_momentum}, nested=True)
    dr = pairs.ele.deltaR(pairs.gen)
    if max_relative_pt is not None:
        compatible = (pairs.gen.pt > 0) & (abs(pairs.ele.pt - pairs.gen.pt) <= max_relative_pt * pairs.gen.pt)
        dr = ak.mask(dr, compatible)
    closest_idx = ak.argmin(dr, axis=2)
    closest_dr = ak.min(dr, axis=2)
    matched = ak.fill_none(closest_dr <= max_dr, False)
    flags = ak.fill_none(gen_ele[closest_idx].statusFlags, 0)

    # CMS GenStatusFlags: bit 0 isPrompt, bit 4 isDirectTauDecayProduct.
    prompt = (flags & (1 << 0)) != 0
    category = ak.where(prompt, 1, 3)
    if pdg_id == 11:
        from_tau = (flags & (1 << 4)) != 0
        category = ak.where(from_tau, 2, category)
    candidates["status"] = ak.values_astype(ak.where(matched, category, 0), np.int32)
    candidates["gen_idx"] = closest_idx
    candidates["gen_dR"] = ak.fill_none(closest_dr, np.nan)
    candidates["gen_pt"] = ak.fill_none(ak.mask(gen_ele[closest_idx].pt, matched), np.nan)
    candidates["gen_energy"] = ak.fill_none(ak.mask(gen_momentum[closest_idx].energy, matched), np.nan)
    if electron_response_max_relative_pt is not None:
        failed = electron_pt_mismatch(candidates.status, candidates.pt, candidates.gen_pt,
                                      electron_response_max_relative_pt)
        candidates["status"] = ak.values_astype(ak.where(failed, 5, candidates.status), np.int32)
    return candidates

