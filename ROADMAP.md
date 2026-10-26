# Research Roadmap: The Probabilistic Digital Twin for ICU Vital-Sign Forecasting

> From class MVP → publishable research paper (Nov 2026 → Jul 2027)
> Team: Nandini, Anjaneya, Faham · Volunteer RAs under a professor · UB MS (ES-AI), graduated Jun 2026

---

## Context

In the Fall semester this team built a class project ("Idea 2" in `Ideas_ IPR project.pdf`):
a **conditional diffusion model that generates many plausible future vital-sign trajectories
for ICU patients**, so clinicians can reason about *risk distributions* instead of a single
point forecast. The goal now is to extend that MVP into a real research paper while the three
members serve as volunteer research assistants on post-completion OPT (Nandini's EAD ends
**Aug 8, 2027**; the other two are awaiting EAD approval).

This document answers four things: (1) a project blurb, (2) concrete ways to expand the work
into a paper, (3) the original paper(s) the idea came from, and (4) an 8-month work division
that defensibly supports ~20 hrs/week each for OPT.

### What actually exists today (gap analysis — read this first)

Reproduced from the three notebooks in the repo:

- **Data:** PhysioNet 2019 **Sepsis Challenge** dataset (`training_setA`, `.psv` files) — *not*
  MIMIC-III as the proposal PDF assumed. 5 vitals: HR, MAP, O2Sat, SBP, Resp. Hourly sampling,
  48-hour windows, stride 12, `MAX_FILES=3000` subset.
- **Diffusion model** (`IPR_Final_Project.ipynb`): a *custom* 1D-conv residual-block denoiser
  (`MedicalDiffusion`), DDPM with 100 steps, linear β schedule, ε-prediction, imputation-style
  conditioning (mask the last `PRED_LEN=3` hours, re-inject known past each reverse step).
  Weighted sampler oversamples sepsis windows. Inference draws 50 trajectories → fan chart.
- **Baseline** (`LSTM_Baseline.ipynb`): bidirectional LSTM encoder + Bahdanau attention,
  seq2seq, predicts 6 hours. Has per-horizon/per-vital MAE/MSE eval.
- **EDA** (`visualizations.ipynb`): sepsis vs non-sepsis distribution analysis (hist/violin/box/KDE).

**What is missing — and this is exactly the publishable work:**

| Gap | Why it matters for a paper |
|---|---|
| **No probabilistic metrics** (CRPS, calibration, coverage, sharpness) | Fan charts are not evidence. The PDF even promised CRPS; it was never implemented. |
| **No clinical risk / alarm module** | The entire *point* of the project ("12/50 futures show MAP collapse → alarm") is not built. This is the novelty. |
| **Patient-level leakage** | `random_split` splits *windows*, so the same patient appears in train and test. Reviewers reject this on sight. |
| **Crude preprocessing** | Fixed-divisor normalization + `fillna(0)`; informative missingness ignored (a known ICU-ML issue). |
| **Weak baselines** | Only LSTM. No CSDI, TimeGrad, deep ensembles, quantile regression, NEWS/MEWS. |
| **Single cohort** | Only PhysioNet 2019 setA. No external validation. |
| **No statistical rigor** | No seeds/variance, no confidence intervals, no ablations. |

The architecture does **not** need to be novel. The contribution is the **clinical-decision
framing + rigorous probabilistic evaluation + cross-cohort validation**. That is both
achievable for a 3-person volunteer team and genuinely publishable.

---

## 1. Project blurb

**One-liner:**
> A conditional diffusion model forecasts a *distribution* of future ICU vital-sign
> trajectories and converts it into calibrated, early deterioration-risk warnings —
> replacing a single averaged prediction with an actionable risk envelope.

**Three-sentence version (for the professor / abstract):**
> Deterministic forecasters (LSTM/GRU/Transformer) collapse volatile patient futures into a
> clinically meaningless average. We train a conditional diffusion model to generate many
> plausible next-hours trajectories for ICU vitals, then score deterioration risk from the
> *tails* of that distribution (e.g., the probability that mean arterial pressure collapses
> below 60 mmHg). We show this distribution-aware risk score is better-calibrated and gives
> earlier warning than point forecasts or direct classifiers, validated across two ICU cohorts.

---

## 2. How to expand it (ideas, ranked)

Grouped as **must-have** (needed for any credible paper) and **novelty/stretch** (the "hook").
The recommended paper thesis is at the end.

### Must-have — turns the MVP into a paper

**A. Rigorous probabilistic evaluation harness.**
Implement CRPS / CRPS-sum, energy score, quantile loss, prediction-interval coverage
probability (PICP) and reliability diagrams, sharpness/interval width, broken down per-horizon
and per-vital. This is what converts "a spread of red lines" into measurable claims.

**B. The clinical deterioration / early-warning module (the heart of the paper).**
For each patient window, generate N futures and compute distributional risk features:
P(MAP < 60), P(SpO2 < 90), P(HR > 130), tail quantiles, time-to-threshold. Aggregate into a
deterioration score and evaluate it as a *classifier* against labels (SepsisLabel, and/or a
composite deterioration outcome) using **AUROC / AUPRC, early-warning lead time, and
decision-curve / net-benefit analysis**. The core research question: *does the distributional
tail-risk score detect deterioration earlier and more reliably than (i) a deterministic point
forecast + same thresholds, (ii) a direct discriminative classifier, and (iii) NEWS/MEWS?*

**C. Leak-free, honest data pipeline.**
Patient-level train/val/test splits (no window leakage); z-score normalization from *train*
statistics only; explicit cohort definition and exclusion criteria; missingness masks as model
input (diffusion/CSDI handles this naturally); report dataset table + flow diagram.

**D. Strong baselines.**
Deterministic: LSTM (have it), GRU, Transformer, seasonal/naive. Probabilistic: **CSDI**
(reference impl), **TimeGrad**, deep ensembles, MC-dropout, quantile regression (MQ-RNN),
Gaussian process. Reviewers demand several of these.

**E. External validation (needed for conference/journal grade).**
Validate on a second cohort. Two options: (i) **PhysioNet 2019 setB** — a *different* hospital
system from setA, so you already have a built-in external-ish test set at zero cost; (ii)
**MIMIC-IV or eICU** (credentialed) for a stronger cross-database claim. Recommend doing both:
setB immediately, MIMIC-IV as the headline external cohort.

### Novelty / stretch — pick ONE as the paper's hook

- **Calibration contribution:** show diffusion gives better-calibrated risk than deterministic
  + MC-dropout + deep ensembles. (Low-risk, high-value — recommended default hook.)
- **Fast sampling for real-time use:** DDIM / consistency-model sampling so 50 trajectories run
  in near-real-time — clinically meaningful and a clean systems contribution.
- **Latent / AE-diffusion** (as the proposal suggested): compress windows with a 1D autoencoder,
  diffuse in latent space for speed/robustness.
- **Guided sampling:** classifier-guided reverse diffusion toward "deterioration" to explicitly
  stress-test worst-case tails.
- **Multi-horizon uncertainty growth:** extend 3h → 6/12h and characterize how/where the risk
  envelope widens.

### Recommended paper thesis (de-risked, achievable)
> *Distribution-aware forecasting via conditional diffusion yields better-calibrated and earlier
> clinical-deterioration warnings than point forecasts and direct classifiers on ICU vitals,
> validated across two cohorts.*
Contributions = (1) distributional risk-scoring framework, (2) calibration + early-warning
evaluation, (3) cross-cohort validation. Architecture novelty optional.

---

## 3. The original paper(s)

The proposal PDF names **CSDI** and **TimeGrad** as the basis. The MVP architecture (treat the
future as missing data, condition on the observed past, impute by reverse diffusion) is directly
the **CSDI** recipe in simplified form. Anchor the related-work section on these:

- **CSDI (the primary source)** — Tashiro, Song, Song, Ermon, *"CSDI: Conditional Score-based
  Diffusion Models for Probabilistic Time Series Imputation,"* NeurIPS 2021.
  [arXiv:2107.03502](https://arxiv.org/abs/2107.03502).
  Code: `github.com/ermongroup/CSDI`. This is the paper the project is built on.
- **TimeGrad** — Rasul et al., *"Autoregressive Denoising Diffusion Models for Multivariate
  Probabilistic Time Series Forecasting,"* ICML 2021. (Named in the PDF.)
- **TDSTF — closest prior work on this exact problem; benchmark against it and cite as proof the
  direction is publishable** — *"A Transformer-based Diffusion Probabilistic Model for Heart Rate
  and Blood Pressure Forecasting in ICU,"* Computer Methods and Programs in Biomedicine, 2024.
  [arXiv:2301.06625](https://arxiv.org/abs/2301.06625). Diffusion + transformer, ICU vitals,
  MIMIC-III. **Differentiate from it** via the clinical risk/deterioration framing, calibration
  focus, and sepsis cohort — not just forecasting accuracy.
- **SSSD** — Alcaraz & Strodthoff, *"Diffusion-based Time Series Imputation and Forecasting with
  Structured State Space Models,"* TMLR 2023. Diffusion on physiological signals; good backbone idea.

Early reading task (Person C): read CSDI + TDSTF in full in Month 1; skim TimeGrad + SSSD.

---

## 4. Work division for OPT (3 people, ~20 hrs/week)

### 4a. How the 20 hr/week OPT rule works (advice — then confirm with your DSO)

**This is orientation, not legal/immigration advice; your UB DSO / International Student Services
is the only authority.** General framework for **post-completion OPT**:

- Employment must be **≥ 20 hrs/week** and **directly related to your degree** to count as valid
  OPT employment and to avoid accruing **unemployment days** (post-completion OPT allows a max of
  **90 days** of unemployment; going over risks your status).
- **Unpaid / volunteer work can qualify** for standard post-completion OPT *if* it is ≥ 20
  hrs/week, degree-related, and does not violate labor law — **but you must document it** and
  **report it in the SEVP Portal** (employer name = the professor/lab/university, start date,
  hours, and a description tying it to your MS in ES-AI).
- **Important caveat on the STEM OPT extension:** the 24-month STEM extension generally does
  **not** allow unpaid/volunteer work — it requires **paid** employment with an **E-Verify**
  employer and a formal **Form I-983** training plan. So this volunteer research is fine for your
  *current* post-completion OPT, but do **not** assume it bridges into a STEM extension. Ask your
  DSO explicitly if a STEM extension is in your future plans.

**Recommended safe logging structure (all three):**
- A simple shared **weekly log** (date, hours, task category, deliverable) per person.
- Target categories that each plausibly fill ~20 hrs: coding/experiments, running/monitoring
  training, reading papers, weekly professor meeting + notes, writing (paper/figures), data work.
- A short **signed statement from the supervising professor** confirming role, start date, and
  ~20 hrs/week — keep it on file for any RFE/audit.
- Each person adds the role to the **SEVP Portal** on their own start date (Nandini now; the
  other two the day their EAD is valid).
- **Staggered EADs:** Nandini can begin and log hours immediately. Anjaneya and Faham should
  **not** log OPT hours until their EAD start date is valid — but they *can* do unpaid
  preparation (reading, environment setup) that simply isn't counted yet.

### 4b. Three durable workstreams (each sustains ~20 hrs/week)

Built so Nandini (hard Aug 8 deadline, starts first) owns the **critical-path foundation** that
everything else depends on, so she is never blocked waiting on the others, and so B and C have a
ready pipeline to plug into when their EADs arrive.

**Person A — Data, cohorts & evaluation infrastructure → Nandini (starts Month 1, solo).**
- Rebuild the data pipeline: patient-level splits, train-stat normalization, missingness masks,
  cohort/exclusion definition, PhysioNet setB as external set.
- Lead **MIMIC-IV / eICU credentialing** (CITI training) and ETL once the team decides (see TODO).
- Build the **evaluation harness** (CRPS, calibration, coverage, AUROC/AUPRC, early-warning
  lead-time, baseline runner) — the shared backbone every experiment uses.

**Person B — Models & diffusion research → Anjaneya *or* Faham.**
- Upgrade the denoiser toward CSDI-style 2D (time × variable) attention / transformer / SSM.
- Implement and port baselines (CSDI reference, TimeGrad, deep ensembles, quantile regression).
- Training infrastructure, seeds/variance, ablations, fast-sampling (DDIM) experiments.

**Person C — Clinical risk module, experiments & writing → the other of Anjaneya/Faham.**
- Build the deterioration risk-scoring framework and clinical labels/lead-time analysis.
- Own results, figures, literature review, and lead the paper + poster.

Roles are durable (good for OPT: a stable, describable position), but the team pair-programs and
reviews across boundaries. Rotate the weekly professor meeting as the "status owner."

### 4c. 8-month timeline (Nov 2026 → Jul 2027)

| Phase | Months | Focus | Milestone |
|---|---|---|---|
| **0. Setup** | Nov | Reproduce MVP, repo hygiene, lit review, finalize dataset decision. Nandini solo if others not yet EAD-valid. | Clean repo + reading done |
| **1. Foundation** | Nov–Dec | Leak-free pipeline + eval harness + honest baselines (add CSDI/TimeGrad/deterministic to existing LSTM). | Reproducible benchmark table |
| **2. Core system** | Jan–Feb | Model upgrades + risk-scoring module v1; first end-to-end results on setA. | Working risk score + metrics |
| **3. Experiments** | Mar–Apr | External validation (setB + MIMIC-IV), ablations, calibration study. | **Workshop paper submission (de-risk win)** |
| **4. Full paper** | May–Jun | Complete results, write conference/journal paper, internal review w/ professor. | Submission-ready draft |
| **5. Submit** | Jul | Polish, submit to conference/journal, poster. Wrap before Aug 8. | Paper submitted |

---

## Open decisions / TODO (team to resolve)

- [ ] **Dataset decision** (the team is discussing — *recommendation: open sepsis set now +
  start MIMIC-IV credentialing in parallel for external validation*). Blocks Phase 3 scope.
- [ ] Confirm the **target venue** shortlist (workshop milestone + conference/journal target).
  Candidates: ML4H, CHIL, NeurIPS/ICLR TS workshops; journals CMPB, JBHI.
- [ ] Each member confirm **OPT logging** with the UB DSO and set up the SEVP Portal entry /
  professor letter / weekly log.
- [ ] Assign **Person B vs Person C** between Anjaneya and Faham.
- [ ] Confirm **compute** (Colab Pro+ vs. a UB lab GPU from the professor).

## Verification / "definition of done" per phase

- **Phase 1:** a check that no patient ID appears in two splits; eval harness reproduces LSTM MAE
  from the old notebook and emits CRPS/coverage for the diffusion model.
- **Phase 2:** risk score produces an AUROC/AUPRC vs SepsisLabel with a confidence interval over
  ≥3 seeds; fan-chart + reliability diagram regenerate from a single script.
- **Phase 3:** results table holds across setA → setB (and MIMIC-IV) with variance bars; ablation
  shows which components matter.
- **Phase 4/5:** full paper compiles with all tables/figures from scripts in the repo; professor sign-off.

## Engineering notes (for implementation)

- Migrate the three notebooks into a small package (`src/data`, `src/models`, `src/eval`,
  `src/risk`) with notebooks as thin drivers — needed for reproducibility and for reviewers.
- Add seeds, a config system (the `CONFIG` dicts are a good start), and a results/experiment logger.
- The leakage fix in `train_lstm()` / `train_model()` (`random_split` on windows) is the first
  code change and affects every number produced so far.
