# PathOmicDRP — Final Submission Bundle (v7)

Generated: 2026-04-12
Target journal: npj Digital Medicine
GitHub: https://github.com/Soonlab/BRCA_Drug_Pred (commit `d17153c`)

## Contents

```
FINAL/
├── manuscript/
│   └── PathOmicDRP_Full_Manuscript_v7.docx        ← submission-ready manuscript
├── cover/
│   └── Cover_Letter_npjDigMed.docx
├── figures/
│   ├── main/                                      ← Figures 1–8 (submission set)
│   ├── supplementary/                             ← Figures S1–S16
│   └── reinforce_v6_v7/                           ← Figures 5A (revised), 9, S17, S18
├── tables/
│   ├── Table4_drug_winner_table.csv
│   └── Table5_imputed_ic50_landscape.csv
├── results/
│   ├── reinforce/                                 ← JSONs + CSV used by v6/v7 text
│   ├── cv_core/                                   ← Phase 2 / Phase 3 CV results
│   ├── w1_clinical_validation.json
│   └── analysis4_ablation.json
└── scripts/                                       ← every script needed to regenerate v6/v7
```

## Claims ↔ evidence map (21 manuscript claims + 7 tables/figures legends)

| Claim | Where it lives | Source file |
|---|---|---|
| Abstract: METABRIC external validation, ρ=0.961 | Abstract (¶8) | `results/reinforce/metabric_validation.json` |
| Abstract: drug-class mechanism-coherent split | Abstract (¶8) | `results/reinforce/drug_heterogeneity.json` |
| Abstract: bootstrap 95% CI disclosure | Abstract (¶8) | `results/w1_clinical_validation.json` |
| Results: CV-avg modality importance (79.8% H) | Results (¶119) | `results/reinforce/cv_ablation.json` |
| Results: bootstrap CI per drug (Dox / Cyclo / Pacl / Doce) | Results (¶104) | `results/w1_clinical_validation.json` |
| Results: fair PCA-256 comparison | Results (¶126) | `results/reinforce/fair_embedding_and_bootstrap.json` |
| Results: drug-level heterogeneity subsection (Heading 3) | Results (¶127–130) | `results/reinforce/drug_heterogeneity.json` |
| Results: METABRIC biomarker concordance (ER/HER2) | Results (¶136) | `results/reinforce/metabric_validation.json` |
| Results: prognostic framing (Cox HR 1.08–1.12) | Results (¶137) | `results/reinforce/metabric_validation.json` |
| Discussion: "not a universal predictor" bridge | Discussion (¶155) | `results/reinforce/drug_heterogeneity.json` |
| Limitations: bootstrap CI small-n caveat | Limitations (¶165) | `results/w1_clinical_validation.json` |
| Limitations: 4-modal external cohort infeasibility | Limitations (¶166) | 13-cohort survey (see Discussion in this README) |
| Future: METABRIC transcriptomic-only note | Future directions (¶168) | `results/reinforce/metabric_validation.json` |
| Table 3 (revised): CV-averaged modality importance | Tables | `results/reinforce/cv_ablation.json` |
| Table 4: per-drug clinical AUC winners | Tables | `tables/Table4_drug_winner_table.csv` |
| Table 5: per-drug imputed-IC₅₀ histology contribution by MOA | Tables | `tables/Table5_imputed_ic50_landscape.csv` |
| Fig 5A (revised): CV-averaged ablation with CIs | Figures | `figures/reinforce_v6_v7/Fig5A_cv_ablation.pdf` |
| Fig 9: Drug-level heterogeneity map | Figures | `figures/reinforce_v6_v7/Fig9_drug_heterogeneity.pdf` |
| Fig S17: METABRIC external validation | Figures | `figures/reinforce_v6_v7/FigS17_metabric_validation.pdf` |
| Fig S18: fair-embedding AUC comparison | Figures | `figures/reinforce_v6_v7/FigS18_fair_embedding.pdf` |

## Core defensive points vs. reviewers

1. **External validation present** → METABRIC n=1,980, biomarker p≈10⁻³⁰, drug-drug ρ=0.961.
2. **89.5% single-fold claim removed** → CV-averaged 79.8% [CI 0.20, 0.28] with variance-stabilization framing.
3. **Bootstrap CI on all clinical AUCs** → Dox [0.81, 1.00], Cyclo [0.55, 1.00] strong; Pacl/Doce flagged as hypothesis-generating.
4. **Fair dimensionality baseline** → PathOmicDRP 256d vs PCA-256 4-modal (0.594 vs 0.527 mean), with drug-level split reported honestly.
5. **4-modal external cohort unavailability** → systematic 13-cohort survey: METABRIC, CPTAC-BRCA, ICGC (UK/EU/KR), GENIE, GENIE-BPC, I-SPY 1/2, NCI-MATCH, ORIEN/Avatar, TCIA, CBCGA, AURORA, MBCproject, SCAN-B, BRCA1/2 consortia. TCGA-BRCA uniquely provides {mutation + mRNA + RPPA + WSI + drug-response}.
6. **Drug-class scoping** → DNA-damaging agents favor PathOmicDRP fusion; taxanes favor PCA-256; scope explicitly delimited to cross-modal multi-factorial response drugs.

## How to regenerate v7 from scratch

```bash
# 1. Data setup (TCGA: see scripts 01-14 in main repo; METABRIC: via cBioPortal LFS)
#    Produces 07_integrated/*.csv and 08_metabric/*.txt

# 2. CV-averaged modality ablation (~45 min, 5-fold × 5 drop conditions)
python scripts/reinforce_cv_ablation.py
# → results/reinforce/cv_ablation.json + fold{1..5}_model.pt

# 3. METABRIC transfer + biomarker/survival (~1 min after data download)
python scripts/reinforce_metabric.py
# → results/reinforce/metabric_validation.json + metabric_predicted_IC50.csv

# 4. Fair PCA-256 embedding + bootstrap CIs (~2 min)
python scripts/reinforce_fair_embedding.py
# → results/reinforce/fair_embedding_and_bootstrap.json

# 5. Drug-level heterogeneity integration (<5 s)
python scripts/reinforce_drug_heterogeneity.py
# → results/reinforce/drug_heterogeneity.json + drug_winner_table.csv

# 6. All reinforcement figures
python scripts/reinforce_figures.py  
# → Fig5A + FigS17 + FigS18

# 7. Manuscript v6 → v7 (chained)
python scripts/generate_manuscript_v6.py
python scripts/generate_manuscript_v7.py
python scripts/patch_v7_discussion.py
python scripts/patch_v7_tables_figures.py
# → manuscript/PathOmicDRP_Full_Manuscript_v7.docx
```

## Environment
- Python 3.10 (conda env `brca_drugres`)
- PyTorch 2.11 + CUDA on RTX 5090
- python-docx, lifelines, scikit-learn, pandas, matplotlib, scipy, statsmodels
- git-lfs 3.7.1 for METABRIC download

## Outstanding questions for next iteration (not blockers)
1. Supplementary note reconciling **direct IC50 output vs 256-d shared embedding** AUC differences on Docetaxel/Paclitaxel (same model, two representations, opposite directions — framed as expected in v7 Discussion but worth an explicit methodological note).
2. Prospective 4-modal validation once an RPPA + WSI cohort with treatment response becomes publicly available (most likely I-SPY 2 if they release extended RPPA).
