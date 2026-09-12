# ExtremoAI Web Application

## Purpose
ExtremoAI predicts microbial adaptation from a whole-proteome FASTA.

Underlying trained classes:
- Halophile
- Mesophile
- Thermophile

The application may additionally return:
- Uncertain / Outside model scope

This is an abstention outcome, not a fourth trained class.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Input
Upload a whole-organism protein FASTA file (`.faa`, `.fasta`, `.fa`).

A single protein sequence is outside the validated scope.

## Scientific basis
The deployment model is the Phase-6B multiclass RBF-SVC trained on the
165-organism curated ExtremoAI cohort after research performance was
estimated independently using genus-grouped nested cross-validation.

The web app reproduces the Phase-3 proteome feature definitions.

## Disclaimer
ExtremoAI is a research prototype. Predictions should not replace experimental
phenotyping or curated ecological annotation.


## Genuine RefSeq demonstration proteomes

The web application includes one genuine NCBI RefSeq complete-proteome
`protein.faa` file for each modeled class:

- Halophile
- Mesophile
- Thermophile

The organisms are selected automatically from the frozen 165-organism
ExtremoAI development cohort using their exact RefSeq `GCF_...` assembly
accessions and downloaded through NCBI Datasets.

These files are provided for **software demonstration only** and must not be
reported as independent external validation samples.
