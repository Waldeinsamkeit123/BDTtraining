from __future__ import annotations

from dataclasses import dataclass

import awkward as ak
import numpy as np


PEPPER_TOPOLOGY_FEATURES = (
    "minDR_b",
    "DR_Tarb1",
    "DR_Tarb2",
    "minDR_TarClean",
    "minDEta_TarClean",
    "minDPhi_TarClean",
)
COMPUTED_RELATIVE_FEATURES = (
    "dPhi_lepton_MET",
    "dPhi_TargetFatJet_MET",
    "dR_TargetFatJet_lepton",
)
AK4_TAG_CATEGORIES = {
    "B4": 54,
    "B3": 53,
    "B2": 52,
    "B1": 51,
    "B0": 50,
    "C4": 44,
    "C3": 43,
    "C2": 42,
    "C1": 41,
    "C0": 40,
}
FORBIDDEN_AK8_TOKENS = (
    "mass",
    "msoftdrop",
    "regressed_mass",
    "higgsvsqcd",
    "bbvscc",
    "globalpart3",
    "tag",
    "bbtagged",
    "cctagged",
    "hftagged",
    "pass_masswindow",
    "random_mass",
)
DISTANCE_SENTINEL = 900.0


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str
    kind: str
    formula: str
    missing: str
    engineered: bool
    ak4_tag: bool = False


def required_branches(channel: str) -> set[str]:
    branches = {
        "run",
        "luminosityBlock",
        "event",
        "weight",
        "higgs_decay",
        "z_decay",
        "tt_hf_flavor",
        "tt_hf_count",
        "nCleanedJet",
        "CleanedJet_pt",
        "CleanedJet_eta",
        "CleanedJet_phi",
        "CleanedJet_mass",
        "CleanedJet_tag",
        "nTargetFatJet",
        "TargetFatJet_pt",
        "TargetFatJet_eta",
        "TargetFatJet_phi",
        "MET_pt",
        "MET_phi",
        "CleanedHT",
        *PEPPER_TOPOLOGY_FEATURES,
    }
    if channel == "1L":
        branches.update(
            {
                "Lepton_pt",
                "Lepton_eta",
                "Lepton_phi",
                "Lepton_mass",
                "Lepton_pdgId",
                "nLepton",
            }
        )
    else:
        branches.update({"HT", "n_gentop", "w_decay"})
    return branches


def wrapped_delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    return (phi1 - phi2 + np.pi) % (2.0 * np.pi) - np.pi


def _pad(values: ak.Array, length: int) -> np.ndarray:
    padded = ak.pad_none(values, length, axis=1, clip=True)
    return np.asarray(ak.to_numpy(ak.fill_none(padded, np.nan)), dtype=np.float32)


def _clean_distance(values: np.ndarray) -> np.ndarray:
    cleaned = np.asarray(values, dtype=np.float32).copy()
    cleaned[cleaned >= DISTANCE_SENTINEL] = np.nan
    return cleaned


def feature_specs(channel: str, include_absolute_phi: bool = True) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    kinematics = ("pt", "eta", "phi", "mass") if include_absolute_phi else ("pt", "eta", "mass")
    for index in range(1, 9):
        for field in kinematics:
            specs.append(
                FeatureSpec(
                    f"CleanedJet_{field}_{index}",
                    f"CleanedJet_{field}",
                    "direct ROOT",
                    f"ROOT collection element {index - 1}, original ordering",
                    "NaN when slot is absent",
                    False,
                )
            )
        for category, value in AK4_TAG_CATEGORIES.items():
            specs.append(
                FeatureSpec(
                    f"CleanedJet_tag_{category}_{index}",
                    "CleanedJet_tag",
                    "computed",
                    f"1[CleanedJet_tag[{index - 1}] == {value}]",
                    "NaN when slot is absent",
                    True,
                    True,
                )
            )
    specs.extend(
        [
            FeatureSpec("nCleanedJet", "nCleanedJet", "direct ROOT", "nCleanedJet", "none", False),
            FeatureSpec(
                "CleanedHT",
                "CleanedJet_pt",
                "computed",
                "sum(CleanedJet_pt) over the same collection",
                "none",
                True,
            ),
        ]
    )
    fat_fields = ("pt", "eta", "phi") if include_absolute_phi else ("pt", "eta")
    for index in range(1, 4):
        for field in fat_fields:
            specs.append(
                FeatureSpec(
                    f"TargetFatJet_{field}_{index}",
                    f"TargetFatJet_{field}",
                    "direct ROOT",
                    f"ROOT collection element {index - 1}, original ordering",
                    "NaN when slot is absent",
                    False,
                )
            )
    if channel == "1L":
        lepton_fields = ("pt", "eta", "phi", "mass") if include_absolute_phi else ("pt", "eta", "mass")
        for field in lepton_fields:
            specs.append(
                FeatureSpec(
                    f"Lepton_{field}",
                    f"Lepton_{field}",
                    "direct ROOT",
                    "first lepton (the 1L selection has exactly one)",
                    "NaN if absent",
                    False,
                )
            )
        specs.extend(
            [
                FeatureSpec("Lepton_isMuon", "Lepton_pdgId", "computed", "1[pdgId==13], exactly as the V4 YAML", "NaN if absent", True),
                FeatureSpec("Lepton_isElectron", "Lepton_pdgId", "computed", "1[pdgId==11], exactly as the V4 YAML", "NaN if absent", True),
                FeatureSpec("MET_pt", "MET_pt", "direct ROOT", "MET_pt", "none", False),
            ]
        )
        if include_absolute_phi:
            specs.append(FeatureSpec("MET_phi", "MET_phi", "direct ROOT", "MET_phi", "none", False))
    for name in PEPPER_TOPOLOGY_FEATURES:
        specs.append(
            FeatureSpec(
                name,
                name,
                "direct ROOT",
                name,
                f"values >= {DISTANCE_SENTINEL:g} converted to NaN",
                False,
            )
        )
    if channel == "1L":
        specs.extend(
            [
                FeatureSpec("dPhi_lepton_MET", "Lepton_phi,MET_phi", "computed", "abs(wrap(Lepton_phi[0]-MET_phi))", "NaN if lepton absent", True),
                FeatureSpec("dPhi_TargetFatJet_MET", "TargetFatJet_phi,MET_phi", "computed", "abs(wrap(TargetFatJet_phi[0]-MET_phi))", "NaN if AK8 absent", True),
                FeatureSpec("dR_TargetFatJet_lepton", "TargetFatJet_eta,TargetFatJet_phi,Lepton_eta,Lepton_phi", "computed", "sqrt(dEta^2+wrap(dPhi)^2)", "NaN if lepton or AK8 absent", True),
            ]
        )
    return specs


def build_feature_matrix(
    arrays: dict[str, ak.Array], channel: str, include_absolute_phi: bool = True
) -> tuple[np.ndarray, list[FeatureSpec]]:
    specs = feature_specs(channel, include_absolute_phi)
    columns: dict[str, np.ndarray] = {}
    jet_slots = {field: _pad(arrays[f"CleanedJet_{field}"], 8) for field in ("pt", "eta", "phi", "mass")}
    tag_slots = _pad(arrays["CleanedJet_tag"], 8)
    for index in range(8):
        for field in ("pt", "eta", "phi", "mass"):
            columns[f"CleanedJet_{field}_{index + 1}"] = jet_slots[field][:, index]
        for category, value in AK4_TAG_CATEGORIES.items():
            valid = np.isfinite(tag_slots[:, index])
            columns[f"CleanedJet_tag_{category}_{index + 1}"] = np.where(
                valid, tag_slots[:, index] == value, np.nan
            ).astype(np.float32)
    columns["nCleanedJet"] = np.asarray(arrays["nCleanedJet"], dtype=np.float32)
    columns["CleanedHT"] = np.asarray(ak.sum(arrays["CleanedJet_pt"], axis=1), dtype=np.float32)

    fat_slots = {field: _pad(arrays[f"TargetFatJet_{field}"], 3) for field in ("pt", "eta", "phi")}
    for index in range(3):
        for field in ("pt", "eta", "phi"):
            columns[f"TargetFatJet_{field}_{index + 1}"] = fat_slots[field][:, index]

    if channel == "1L":
        lep_slots = {field: _pad(arrays[f"Lepton_{field}"], 1)[:, 0] for field in ("pt", "eta", "phi", "mass")}
        pdgid = _pad(arrays["Lepton_pdgId"], 1)[:, 0]
        columns.update({f"Lepton_{field}": values for field, values in lep_slots.items()})
        columns["Lepton_isMuon"] = np.where(np.isfinite(pdgid), pdgid == 13, np.nan).astype(np.float32)
        columns["Lepton_isElectron"] = np.where(np.isfinite(pdgid), pdgid == 11, np.nan).astype(np.float32)
        lep_phi = lep_slots["phi"]
        lep_eta = lep_slots["eta"]
        met_phi = np.asarray(arrays["MET_phi"], dtype=np.float32)
        fat_phi = fat_slots["phi"][:, 0]
        fat_eta = fat_slots["eta"][:, 0]
        columns["dPhi_lepton_MET"] = np.abs(wrapped_delta_phi(lep_phi, met_phi)).astype(np.float32)
        columns["dPhi_TargetFatJet_MET"] = np.abs(wrapped_delta_phi(fat_phi, met_phi)).astype(np.float32)
        dphi = wrapped_delta_phi(fat_phi, lep_phi)
        columns["dR_TargetFatJet_lepton"] = np.sqrt((fat_eta - lep_eta) ** 2 + dphi**2).astype(np.float32)

    columns["MET_pt"] = np.asarray(arrays["MET_pt"], dtype=np.float32)
    columns["MET_phi"] = np.asarray(arrays["MET_phi"], dtype=np.float32)
    for name in PEPPER_TOPOLOGY_FEATURES:
        columns[name] = _clean_distance(np.asarray(arrays[name]))

    feature_names = [spec.name for spec in specs]
    matrix = np.column_stack([columns[name] for name in feature_names]).astype(np.float32)
    audit_forbidden_features(feature_names)
    return matrix, specs


def audit_forbidden_features(feature_names: list[str]) -> None:
    leaked = []
    for name in feature_names:
        lowered = name.lower()
        if "targetfatjet" in lowered and any(token in lowered for token in FORBIDDEN_AK8_TOKENS):
            leaked.append(name)
    if leaked:
        raise ValueError("Forbidden AK8 information entered the feature matrix: " + ", ".join(leaked))
