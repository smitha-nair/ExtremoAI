
from io import StringIO
from collections import Counter
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_PROTEIN_LENGTH = 30
AA20 = list("ACDEFGHIKLMNPQRSTVWY")

AA_GROUPS = {
    "acidic": set("DE"),
    "basic": set("KR"),
    "charged": set("DEKR"),
    "aromatic": set("FWY"),
    "polar": set("STNQ"),
    "hydrophobic": set("AVILMFWY"),
    "small": set("AGSTCVP")
}

def safe_stats(values, prefix):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan
        }
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_median": float(np.median(a)),
        f"{prefix}_std": float(np.std(a, ddof=0))
    }

def aliphatic_index_from_fraction(frac):
    return 100.0 * (
        frac.get("A", 0.0)
        + 2.9 * frac.get("V", 0.0)
        + 3.9 * (frac.get("I", 0.0) + frac.get("L", 0.0))
    )

def extract_proteome_features_from_text(fasta_text):
    total_records = 0
    valid_sequences = []
    excluded_short = 0
    excluded_nonstandard = 0

    for record in SeqIO.parse(StringIO(fasta_text), "fasta"):
        total_records += 1
        seq = str(record.seq).upper().replace("*", "")

        if len(seq) < MIN_PROTEIN_LENGTH:
            excluded_short += 1
            continue

        if not set(seq).issubset(STANDARD_AA):
            excluded_nonstandard += 1
            continue

        valid_sequences.append(seq)

    if total_records == 0:
        raise ValueError(
            "No FASTA records were detected. Please upload a protein FASTA file."
        )

    if len(valid_sequences) == 0:
        raise ValueError(
            "No valid proteins remained after applying the Phase-3 QC rules."
        )

    lengths = np.array([len(s) for s in valid_sequences], dtype=float)
    total_aa = int(lengths.sum())

    counts = Counter()
    for seq in valid_sequences:
        counts.update(seq)

    aa_frac = {aa: counts.get(aa, 0) / total_aa for aa in AA20}

    pI_values = []
    mw_values = []
    instability_values = []
    aromaticity_values = []
    gravy_values = []
    aliphatic_values = []

    for seq in valid_sequences:
        pa = ProteinAnalysis(seq)
        pI_values.append(pa.isoelectric_point())
        mw_values.append(pa.molecular_weight())
        instability_values.append(pa.instability_index())
        aromaticity_values.append(pa.aromaticity())
        gravy_values.append(pa.gravy())

        c = Counter(seq)
        n = len(seq)
        frac = {aa: c.get(aa, 0) / n for aa in AA20}
        aliphatic_values.append(aliphatic_index_from_fraction(frac))

    pI_arr = np.asarray(pI_values, dtype=float)

    out = {
        "protein_count": len(valid_sequences),
        "total_amino_acids": total_aa,
        "protein_length_mean": float(lengths.mean()),
        "protein_length_median": float(np.median(lengths)),
        "protein_length_std": float(lengths.std(ddof=0)),
        "protein_length_min": int(lengths.min()),
        "protein_length_max": int(lengths.max()),
        "excluded_short_proteins": excluded_short,
        "excluded_nonstandard_proteins": excluded_nonstandard,
        "valid_protein_fraction": (
            len(valid_sequences) / total_records if total_records else np.nan
        )
    }

    for aa in AA20:
        out[f"AAC_{aa}"] = aa_frac[aa]

    for group, residues in AA_GROUPS.items():
        out[f"fraction_{group}"] = sum(aa_frac[a] for a in residues)

    acidic = aa_frac["D"] + aa_frac["E"]
    basic = aa_frac["K"] + aa_frac["R"]

    out["ratio_acidic_basic"] = acidic / basic if basic > 0 else np.nan
    out["ratio_DE_KR"] = acidic / basic if basic > 0 else np.nan
    out["ratio_R_K"] = aa_frac["R"] / aa_frac["K"] if aa_frac["K"] > 0 else np.nan
    out["proteome_aliphatic_index"] = aliphatic_index_from_fraction(aa_frac)

    out.update(safe_stats(pI_values, "pI"))
    out.update(safe_stats(mw_values, "molecular_weight"))
    out.update(safe_stats(instability_values, "instability_index"))
    out.update(safe_stats(aromaticity_values, "aromaticity"))
    out.update(safe_stats(gravy_values, "gravy"))
    out.update(safe_stats(aliphatic_values, "aliphatic_index"))

    out["fraction_acidic_proteins_pI_lt_5_5"] = float(np.mean(pI_arr < 5.5))
    out["fraction_neutral_proteins_pI_5_5_to_8"] = float(
        np.mean((pI_arr >= 5.5) & (pI_arr <= 8.0))
    )
    out["fraction_basic_proteins_pI_gt_8"] = float(np.mean(pI_arr > 8.0))
    out["total_fasta_records"] = total_records

    return out
