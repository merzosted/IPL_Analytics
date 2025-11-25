# IPL Auction Analytics: The Ghost in the Machine

## Project Overview

This project quantifies "Killer Instinct" in cricket bowlers using Bayesian inference, proving that mental strength can be measured with data.

**Challenge:** Choose between two death-over specialists with identical base prices  
**Method:** Bayesian Logistic Regression to measure pressure response  
**Result:** Strong evidence for "Killer Instinct" in Bowler B  

---

## Quick Start

### Visualizations

- **executive_summary_plot.png** - Key findings (2 panels)
- **killer_instinct_analysis.png** - Comprehensive analysis (6 panels)

### Code Files

- **analysis.py** - Complete analysis pipeline (can be run standalone)
- **create_visualizations.py** - Visualization generation

### Data Files

- **IPL_Bowler_Detailed_Data.xls** - Input dataset (4,800 balls)
- **bayesian_trace.nc** - MCMC posterior samples (for reproducibility)

---

## How to Use

### Option 1: View the Jupyter Notebook (Recommended)

```bash
jupyter notebook IPL_Auction_Analytics.ipynb
```

This notebook contains:
- Complete analysis workflow
- Inline visualizations
- Detailed explanations
- All results and interpretations

### Option 2: Run the Analysis Script

```bash
python analysis.py
```

This will:
- Load and explore the data
- Engineer the "pressure" feature
- Build and sample the Bayesian model
- Print all results to console
- Save the trace to `bayesian_trace.nc`

### Option 3: Generate Visualizations

```bash
python create_visualizations.py
```

This creates:
- `killer_instinct_analysis.png` - 6-panel comprehensive analysis
- `executive_summary_plot.png` - 2-panel executive summary

---

## Key Findings

### The Killer Instinct Effect

**Bowler A (The Machine):**
- Wicket probability DECREASES by 2.0 pp after bowling a dot ball
- Shows signs of mental fragility under pressure

**Bowler B (The Gambler):**
- Wicket probability INCREASES by 30.8 pp after bowling a dot ball
- Demonstrates genuine "Killer Instinct"

### Statistical Evidence

- **Killer Instinct Coefficient:** 2.68 (log-odds scale)
- **94% Credible Interval:** [2.00, 3.34] (excludes zero)
- **Posterior Probability > 0:** 100%
- **Model Convergence:** Perfect (R-hat = 1.0)

### Recommendation

**🎯 BUY BOWLER B**

The data provides conclusive evidence that Bowler B has a genuine mental edge in pressure situations.

---

## Technical Details

### Requirements

```bash
pip install pandas numpy matplotlib seaborn pymc arviz openpyxl xlrd
```

### Data Structure

- **4,800 balls** from T20 leagues (2022-2023)
- **2,400 balls** in death overs (16-20)
- **Equal representation:** 2,400 balls per bowler
- **Balanced phases:** 2,400 powerplay, 2,400 death

### Model Specification

```
logit(P(Wicket)) = β₀ + β₁(Pressure) + β₂(Bowler_B) + β₃(Pressure × Bowler_B) 
                   + β₄(Pitch_Batting) + β₅(Pitch_Bowling) + β₆(Batter_Avg)
```

Where:
- **β₃** = The "Killer Instinct" coefficient (our key parameter)
- **Pressure** = 1 if previous ball was a dot ball (0 runs)
- **Bowler_B** = 1 for Bowler B, 0 for Bowler A

### Priors

- Intercept: N(-2.5, 1) [base wicket rate ~7%]
- All coefficients: N(0, 1) [weakly informative]
- Killer Instinct: N(0, 1.5) [slightly more variance]

### Sampling

- **Method:** NUTS (No-U-Turn Sampler)
- **Chains:** 2
- **Draws per chain:** 2,000
- **Tune:** 1,000
- **Total posterior samples:** 4,000
- **Random seed:** 42 (for reproducibility)

---

## File Descriptions

### Analysis Files

| File | Description |
|------|-------------|
| `IPL_Auction_Analytics.ipynb` | Main Jupyter notebook |
| `analysis.py` | Standalone analysis script |
| `create_visualizations.py` | Visualization generation |

### Data Files

| File | Description | Format |
|------|-------------|--------|
| `IPL_Bowler_Detailed_Data.xls` | Input dataset | Excel |
| `bayesian_trace.nc` | MCMC posterior samples | NetCDF |

### Visualizations

| File | Description | Panels |
|------|-------------|--------|
| `executive_summary_plot.png` | Key findings | 2 |
| `killer_instinct_analysis.png` | Comprehensive analysis | 6 |

---

## Methodology Highlights

### Feature Engineering: The "Mental Proxy"

**Key Insight:** "Pressure" = Dot Ball (0 runs) in Death Overs

**Critical Implementation:**
- Dot ball on last ball of over does NOT apply pressure to first ball of next over
- Pressure only applies within same over OR if same bowler continues
- This avoids the sequential logic error that LLMs often make

### Bayesian Approach

**Why Bayesian?**
1. Quantifies uncertainty (credible intervals)
2. Provides probability statements about parameters
3. Naturally handles small sample sizes
4. Allows incorporation of domain knowledge (priors)

**Model Diagnostics:**
- ✅ R-hat = 1.0 (perfect convergence)
- ✅ ESS > 2,000 (sufficient effective sample size)
- ✅ No divergences (sampling was successful)
- ✅ Trace plots show good mixing

---

## Results Summary

### Descriptive Statistics (Death Overs)

| Metric | Bowler A | Bowler B |
|--------|----------|----------|
| Wicket rate (after pressure) | 2.42% | 35.61% |
| Wicket rate (no pressure) | 4.71% | 6.01% |
| **Pressure effect** | **-2.29 pp** | **+29.61 pp** |

### Model-Adjusted Probabilities

| Situation | Bowler A | Bowler B | Difference |
|-----------|----------|----------|------------|
| No Pressure | 4.99% | 6.58% | +1.59 pp |
| After Pressure | 2.97% | 37.36% | +34.39 pp |
| **Pressure Effect** | **-2.02 pp** | **+30.77 pp** | **+32.79 pp** |

### Bayesian Inference

| Parameter | Mean | 94% HDI | Interpretation |
|-----------|------|---------|----------------|
| Intercept | -2.95 | [-3.29, -2.60] | Baseline log-odds |
| Pressure (β₁) | -0.54 | [-1.12, 0.08] | Bowler A pressure effect |
| Bowler B (β₂) | 0.29 | [-0.08, 0.70] | Bowler B baseline |
| **Killer Instinct (β₃)** | **2.68** | **[2.00, 3.34]** | **THE KEY FINDING** |
| Pitch Batting | -0.36 | [-0.69, -0.04] | Batting pitch effect |
| Pitch Bowling | 0.08 | [-0.24, 0.41] | Bowling pitch effect |
| Batter Avg | -0.06 | [-0.20, 0.09] | Batsman quality |

---

## Interpretation

### What is "Killer Instinct"?

The Killer Instinct coefficient (β₃ = 2.68) represents the ADDITIONAL effect of pressure for Bowler B compared to Bowler A.

**In plain English:**
- When Bowler A bowls a dot ball, his wicket probability DECREASES (mental fragility)
- When Bowler B bowls a dot ball, his wicket probability INCREASES dramatically (mental toughness)
- The difference is 32.79 percentage points - a massive effect

### Why This Matters

In T20 cricket, death overs (16-20) are the most crucial. A bowler who can:
1. Apply pressure (bowl dot balls)
2. Capitalize on that pressure (take wickets on the next ball)

...is worth their weight in gold.

**Bowler B does both.**

---

## Addressing the Coach's Skepticism

### "You can't measure mental strength in a spreadsheet"

**We just did.** The Bayesian model quantifies:

1. **Resilience:** Bowler B doesn't crumble under pressure
2. **Aggression:** Bowler B becomes MORE dangerous after a dot ball
3. **Reading the Batsman:** Bowler B capitalizes on batsman anxiety

### "But the data is noisy"

**That's why we used Bayesian inference.** The model:
- Controls for pitch conditions
- Controls for batsman quality
- Accounts for baseline bowler differences
- Quantifies uncertainty with credible intervals

**The effect is REAL, not noise.**

---

## The DCA Way: AI-Assisted Analysis

This analysis was conducted using:
- ✅ Python + PyMC for Bayesian inference
- ✅ Careful feature engineering to avoid sequential logic errors
- ✅ Domain knowledge to set appropriate priors
- ✅ Rigorous validation of assumptions

**Note:** LLMs (ChatGPT, Claude, Gemini) were used for boilerplate code, but all logic, priors, and interpretations were human-validated.

---

## Reproducibility

### To Reproduce This Analysis

1. Clone this repository
2. Install requirements: `pip install pandas numpy matplotlib seaborn pymc arviz openpyxl xlrd`
3. Run: `python analysis.py`
4. Or open: `jupyter notebook IPL_Auction_Analytics.ipynb`

### Random Seed

All analyses use `random_seed=42` for reproducibility.

### Trace File

The MCMC posterior samples are saved in `bayesian_trace.nc` and can be loaded with:

```python
import arviz as az
trace = az.from_netcdf('bayesian_trace.nc')
```

---

**"Mental strength CAN be measured. We just proved it."**

🎯 **RECOMMENDATION: BUY BOWLER B**
