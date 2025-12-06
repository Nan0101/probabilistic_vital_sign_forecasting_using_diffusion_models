# 🔬 How Diffusion-Based Anomaly Detection Works

## The Core Idea

The diffusion model learns what "normal" vital signs look like from thousands of patients. When we show it a patient's history and ask it to predict the next few hours, it predicts what it **expects** based on normal patterns.

**If the actual values are very different from predictions → that's an anomaly!**

---

## What is a Z-Score?

The **Z-score** measures how many standard deviations the actual value is from the predicted value:

$$Z = \frac{|actual - predicted|}{uncertainty}$$

- **actual** = The real vital sign value we observed
- **predicted** = What the model expected (mean of multiple samples)
- **uncertainty** = The model's confidence (standard deviation of samples)

---

## Z-Score Interpretation

| Z-Score | Meaning          | How Common     |
| ------- | ---------------- | -------------- |
| 0 - 1   | Normal           | 68% of data    |
| 1 - 2   | Slightly unusual | 27% of data    |
| 2 - 3   | Very unusual     | 4.5% of data   |
| > 3     | Extremely rare   | < 0.3% of data |

**We use Z > 1.5 as our threshold** - anything above this is flagged as anomalous.

---

## Example: Normal vs Crash

### Normal case:

- History: HR = [75, 78, 74, 76, 77...]
- Model predicts: ~76 bpm ± 5 bpm
- Actual: 78 bpm
- Z = |78-76|/5 = **0.4** ✅ Normal

### Crash case:

- History shows sudden drop to 35 bpm
- Model still predicts: ~76 bpm ± 5 bpm (expects normal)
- Actual: 35 bpm
- Z = |35-76|/5 = **8.2** 🚨 ANOMALY!

---

## Why This Works

1. The model learned "normal" patterns from 3,000 patients
2. It has rarely/never seen crashes during training
3. When a crash happens, the model still predicts normal values
4. The huge gap between prediction and reality creates a high Z-score
5. We detect the anomaly **before** the crash fully develops (early warning)

---

## Results

In our evaluation, **90% of crashes had early warning detection** with 1-10 hours of lead time!
