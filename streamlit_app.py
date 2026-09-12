
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances

from feature_extraction import extract_proteome_features_from_text

APPDIR = Path(__file__).resolve().parent

SAMPLE_FASTA = """>example_protein_1 hypothetical protein
MKTIIALSYIFCLVFADYKDDDDKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
>example_protein_2 DNA-binding protein
MDEKRHILMFWYAGSTCVPNQDEKRHILMFWYAGSTCVPNQDEKRHILMFWY
>example_protein_3 metabolic enzyme
MAVTAPGKGILAALEAGADVVVVAGHSMGGKSTLLKQLAERAGADVVVVAAA
"""


st.set_page_config(
    page_title="ExtremoAI",
    page_icon="🧬",
    layout="wide"
)

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        corr = pd.DataFrame(X).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        self.drop_idx_ = [
            j for j in range(upper.shape[1])
            if any(upper.iloc[:, j] > self.threshold)
        ]
        self.keep_idx_ = [
            j for j in range(X.shape[1])
            if j not in self.drop_idx_
        ]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_idx_]

    def get_support(self):
        return np.asarray(self.keep_idx_, dtype=int)


@st.cache_resource
def load_model():
    return joblib.load(APPDIR / "extremoai_deploy_model.joblib")


@st.cache_data
def load_support_files():
    with open(APPDIR / "extremoai_abstention_calibration.json") as f:
        calibration = json.load(f)

    with open(APPDIR / "extremoai_deploy_model_metadata.json") as f:
        metadata = json.load(f)

    reference = pd.read_csv(
        APPDIR / "extremoai_deploy_reference_feature_profiles.csv"
    )

    ood = np.load(
        APPDIR / "extremoai_ood_reference.npz",
        allow_pickle=True
    )

    sample_metadata = pd.read_csv(
        APPDIR / "samples" / "refseq_demo_proteomes_metadata.csv"
    )

    return (
        calibration,
        metadata,
        reference,
        ood["transformed_training"],
        ood["labels"].astype(str),
        sample_metadata
    )


def transform_before_model(pipeline, Xframe):
    z = pipeline.named_steps["imputer"].transform(Xframe)
    z = pipeline.named_steps["corr"].transform(z)
    z = pipeline.named_steps["scaler"].transform(z)
    return np.asarray(z, dtype=float)


def evaluate_abstention(
    predicted_class,
    max_probability,
    transformed_sample,
    calibration,
    transformed_training,
    training_labels
):
    prob_thr = float(
        calibration["probability_thresholds"][predicted_class]
    )
    dist_thr = float(
        calibration["distance_thresholds"][predicted_class]
    )

    class_ref = transformed_training[
        training_labels == predicted_class
    ]

    nearest_distance = float(
        pairwise_distances(
            transformed_sample,
            class_ref,
            metric="euclidean"
        ).min()
    )

    low_support = max_probability < prob_thr
    out_of_distribution = nearest_distance > dist_thr

    return {
        "abstain": low_support or out_of_distribution,
        "low_support": low_support,
        "out_of_distribution": out_of_distribution,
        "probability_threshold": prob_thr,
        "distance_threshold": dist_thr,
        "nearest_distance": nearest_distance
    }


bundle = load_model()

(
    calibration,
    metadata,
    reference_profiles,
    transformed_training,
    training_labels,
    sample_metadata
) = load_support_files()

pipeline = bundle["pipeline"]
label_encoder = bundle["label_encoder"]
feature_cols = list(bundle["feature_cols"])
class_names = list(bundle["class_names"])

st.title("🧬 ExtremoAI")
st.subheader(
    "Explainable proteome-based prediction of microbial adaptation"
)

st.markdown(
    """
Upload a **whole microbial proteome FASTA** to obtain a research prediction
of **Halophile, Mesophile, or Thermophile**.

ExtremoAI analyses proteome-wide amino-acid and physicochemical signatures.
It does **not** classify an organism from a single protein sequence.
"""
)

with st.expander("Model scope and validation"):
    st.write(
        "Training cohort: 165 curated organisms "
        "(55 Halophile, 55 Mesophile, 55 Thermophile), 149 unique genera."
    )
    st.write(
        "Research performance was estimated in Phase 5B using "
        "genus-grouped nested cross-validation."
    )
    st.write(
        "Multiclass macro-F1: "
        f"{metadata.get('phase5b_reported_multiclass_macro_f1', 'NA')}"
    )
    st.write(
        "'Uncertain / Outside model scope' is an abstention outcome, "
        "not a fourth trained class."
    )

st.markdown("### Try a genuine RefSeq demonstration proteome")

st.write(
    "You can first test ExtremoAI using one of three **genuine complete "
    "NCBI RefSeq protein FASTA files** selected from the frozen ExtremoAI "
    "development cohort."
)

demo_cols = st.columns(3)

for i, cname in enumerate(["Halophile", "Mesophile", "Thermophile"]):
    r = sample_metadata[
        sample_metadata["extremophile_class"] == cname
    ].iloc[0]

    sample_path = APPDIR / "samples" / r["file_name"]

    with demo_cols[i]:
        st.markdown(f"#### {cname}")
        st.write(f"**{r['species_name']}**")
        st.caption(
            f"RefSeq assembly: {r['assembly_accession']} | "
            f"{int(r['protein_records']):,} protein records"
        )

        st.download_button(
            f"Download {cname} RefSeq proteome",
            data=sample_path.read_bytes(),
            file_name=r["file_name"],
            mime="text/plain",
            key=f"demo_{cname}"
        )

st.info(
    "These three genuine RefSeq files are **demonstration examples drawn "
    "from the model-development cohort**. They are not an independent "
    "external validation set."
)

st.markdown("### Upload your own proteome")

st.write(
    "Upload **one whole-organism proteome FASTA** containing many protein "
    "sequences. Each protein record should start with a `>` header line."
)

with st.expander("Show example FASTA format"):
    st.code(SAMPLE_FASTA, language="text")
    st.warning(
        "This short example is only to demonstrate FASTA formatting. "
        "It is **not** a complete proteome and should not be used to judge "
        "the biological accuracy of ExtremoAI."
    )

st.download_button(
    "Download example FASTA format",
    data=SAMPLE_FASTA,
    file_name="extremoai_example_format.faa",
    mime="text/plain"
)

uploaded = st.file_uploader(
    "Upload whole-proteome FASTA",
    type=["faa", "fasta", "fa", "txt"],
    help=(
        "Upload a protein FASTA containing the full annotated proteome "
        "for one microbial organism. A single protein sequence is outside "
        "the validated scope of ExtremoAI."
    )
)

if uploaded is None:
    st.info(
        "Upload a whole-organism protein FASTA file to begin. "
        "You can download the example above to see the expected format."
    )
    st.stop()

try:
    fasta_text = uploaded.getvalue().decode("utf-8", errors="ignore")
    features = extract_proteome_features_from_text(fasta_text)
except Exception as e:
    st.error(f"FASTA processing failed: {e}")
    st.stop()

st.markdown("### 1. Proteome quality check")
q1, q2, q3, q4 = st.columns(4)

q1.metric("FASTA records", int(features["total_fasta_records"]))
q2.metric("Valid proteins", int(features["protein_count"]))
q3.metric("Valid protein fraction", f"{features['valid_protein_fraction']:.1%}")
q4.metric("Total amino acids", f"{int(features['total_amino_acids']):,}")

if features["valid_protein_fraction"] < 0.90:
    st.warning(
        "Less than 90% of FASTA records passed the same QC rules "
        "used during model development."
    )

if features["protein_count"] < 100:
    st.warning(
        "This file contains relatively few valid proteins and may not "
        "represent a complete microbial proteome."
    )

missing_features = [f for f in feature_cols if f not in features]

if missing_features:
    st.error(
        "The FASTA could not generate all model features: "
        + ", ".join(missing_features)
    )
    st.stop()

Xnew = pd.DataFrame(
    [[features[f] for f in feature_cols]],
    columns=feature_cols
)

pred_encoded = pipeline.predict(Xnew)[0]
probabilities = pipeline.predict_proba(Xnew)[0]

predicted_class = label_encoder.inverse_transform(
    [pred_encoded]
)[0]

prob_map = {
    cname: float(p)
    for cname, p in zip(class_names, probabilities)
}

max_probability = max(prob_map.values())
Znew = transform_before_model(pipeline, Xnew)

abstention = evaluate_abstention(
    predicted_class,
    max_probability,
    Znew,
    calibration,
    transformed_training,
    training_labels
)

display_class = (
    "Uncertain / Outside model scope"
    if abstention["abstain"]
    else predicted_class
)

st.markdown("### 2. ExtremoAI prediction")

if abstention["abstain"]:
    st.warning("### Uncertain / Outside model scope")
    st.write(
        "The underlying 3-class model's closest prediction is "
        f"**{predicted_class}**, but ExtremoAI is abstaining from "
        "a definitive class assignment."
    )

    reasons = []
    if abstention["low_support"]:
        reasons.append(
            "model support is lower than expected from correct "
            "genus-grouped validation predictions"
        )
    if abstention["out_of_distribution"]:
        reasons.append(
            "the proteome is unusually distant from the "
            f"{predicted_class} training distribution"
        )

    if reasons:
        st.write("Reason: " + "; ".join(reasons) + ".")
else:
    st.success(f"### Predicted adaptation: {predicted_class}")

st.markdown("### 3. Model-estimated class probabilities")

prob_df = pd.DataFrame({
    "Class": class_names,
    "Probability": [prob_map[c] for c in class_names]
}).sort_values("Probability", ascending=False)

st.bar_chart(prob_df.set_index("Class"))

st.dataframe(
    prob_df.style.format({"Probability": "{:.3f}"}),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Probabilities are model-estimated values and are not equivalent "
    "to experimentally measured biological certainty."
)

st.markdown("### 4. Reliability / scope diagnostics")

d1, d2 = st.columns(2)

d1.metric(
    "Maximum model probability",
    f"{max_probability:.3f}"
)

d2.metric(
    "Nearest training distance",
    f"{abstention['nearest_distance']:.3f}"
)

diag = pd.DataFrame([
    {
        "Check": "Model support",
        "Observed": max_probability,
        "Reference threshold": abstention["probability_threshold"],
        "Status": (
            "Pass"
            if not abstention["low_support"]
            else "Outside expected range"
        )
    },
    {
        "Check": "Distribution distance",
        "Observed": abstention["nearest_distance"],
        "Reference threshold": abstention["distance_threshold"],
        "Status": (
            "Pass"
            if not abstention["out_of_distribution"]
            else "Outside expected range"
        )
    }
])

st.dataframe(diag, use_container_width=True, hide_index=True)

st.markdown("### 5. Proteome characteristics")

signature_features = [
    "ratio_acidic_basic",
    "pI_mean",
    "fraction_acidic",
    "fraction_basic",
    "fraction_hydrophobic",
    "proteome_aliphatic_index",
    "gravy_mean",
    "fraction_acidic_proteins_pI_lt_5_5",
    "fraction_basic_proteins_pI_gt_8"
]

pretty = {
    "ratio_acidic_basic": "Acidic/basic residue ratio",
    "pI_mean": "Mean protein pI",
    "fraction_acidic": "Acidic residue fraction",
    "fraction_basic": "Basic residue fraction",
    "fraction_hydrophobic": "Hydrophobic residue fraction",
    "proteome_aliphatic_index": "Proteome aliphatic index",
    "gravy_mean": "Mean GRAVY",
    "fraction_acidic_proteins_pI_lt_5_5":
        "Fraction acidic proteins (pI < 5.5)",
    "fraction_basic_proteins_pI_gt_8":
        "Fraction basic proteins (pI > 8)"
}

comparison_rows = []

for feat in signature_features:
    if feat not in features:
        continue

    row = {
        "Feature": pretty.get(feat, feat),
        "Uploaded proteome": features[feat]
    }

    for cname in class_names:
        ref_row = reference_profiles[
            reference_profiles["extremophile_class"] == cname
        ]
        col = f"{feat}__median"

        if len(ref_row) == 1 and col in ref_row.columns:
            row[f"{cname} median"] = float(
                ref_row.iloc[0][col]
            )

    comparison_rows.append(row)

comparison = pd.DataFrame(comparison_rows)

st.dataframe(
    comparison.style.format(precision=4),
    use_container_width=True,
    hide_index=True
)

result_record = {
    "uploaded_file": uploaded.name,
    "display_result": display_class,
    "underlying_3class_prediction": predicted_class,
    "abstained": bool(abstention["abstain"]),
    "max_model_probability": max_probability,
    "nearest_training_distance": abstention["nearest_distance"],
    **{f"prob_{k}": v for k, v in prob_map.items()},
    "valid_proteins": int(features["protein_count"]),
    "valid_protein_fraction": float(features["valid_protein_fraction"])
}

result_csv = pd.DataFrame([result_record]).to_csv(index=False)

st.download_button(
    "Download prediction summary",
    data=result_csv,
    file_name="extremoai_prediction.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption(
    "ExtremoAI is a research prototype for proteome-level computational "
    "prediction. It should not replace experimental phenotyping or curated "
    "ecological annotation."
)
