"""
The Ghost in the Machine: IPL Auction Analytics
Quantifying 'Killer Instinct' in Death Over Specialists
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("PHASE 1: DATA LOADING AND EXPLORATION")
print("="*80)

# Load the data
df = pd.read_excel('IPL_Bowler_Detailed_Data.xls')

print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head(10))

print(f"\nData Types:")
print(df.dtypes)

print(f"\nMissing Values:")
print(df.isnull().sum())

print(f"\nBasic Statistics:")
print(df.describe())

# Check unique values
print(f"\nUnique Bowlers: {df['Bowler'].unique()}")
print(f"Unique Phases: {df['Phase'].unique()}")
print(f"Unique Pitch Types: {df['Pitch_Type'].unique()}")

# Distribution of data
print(f"\nBowler Distribution:")
print(df['Bowler'].value_counts())

print(f"\nPhase Distribution:")
print(df['Phase'].value_counts())

print(f"\nWicket Rate Overall: {df['Is_Wicket'].mean():.4f}")


print("\n" + "="*80)
print("PHASE 2: FEATURE ENGINEERING - THE 'MENTAL PROXY'")
print("="*80)

# Sort data properly to ensure sequential ball order
df = df.sort_values(['Match_ID', 'Over', 'Ball']).reset_index(drop=True)

# Create a flag for dot balls (0 runs conceded)
df['Is_Dot_Ball'] = (df['Runs_Conceded'] == 0).astype(int)

# CRITICAL: Create pressure flag - dot ball followed by next ball
# We need to be careful about over boundaries
df['Pressure_Applied'] = 0

# For each row, check if previous ball was a dot ball
# But only if it's not the first ball of an over (Ball != 1)
# And only if it's in the same match and same over OR consecutive over
for i in range(1, len(df)):
    prev_row = df.iloc[i-1]
    curr_row = df.iloc[i]
    
    # Check if same match
    if prev_row['Match_ID'] == curr_row['Match_ID']:
        # Check if same bowler (important!)
        if prev_row['Bowler'] == curr_row['Bowler']:
            # Case 1: Same over, next ball
            if (prev_row['Over'] == curr_row['Over'] and 
                curr_row['Ball'] == prev_row['Ball'] + 1):
                if prev_row['Is_Dot_Ball'] == 1:
                    df.loc[i, 'Pressure_Applied'] = 1
            
            # Case 2: Next over, first ball (only if prev ball was ball 6)
            elif (prev_row['Over'] + 1 == curr_row['Over'] and 
                  prev_row['Ball'] == 6 and curr_row['Ball'] == 1):
                if prev_row['Is_Dot_Ball'] == 1:
                    df.loc[i, 'Pressure_Applied'] = 1

print(f"\nTotal balls: {len(df)}")
print(f"Dot balls: {df['Is_Dot_Ball'].sum()} ({df['Is_Dot_Ball'].mean()*100:.2f}%)")
print(f"Balls with pressure applied: {df['Pressure_Applied'].sum()}")

# Filter for Death overs only (this is where the magic happens)
death_df = df[df['Phase'] == 'Death'].copy()

print(f"\nDeath overs only:")
print(f"Total balls: {len(death_df)}")
print(f"Dot balls: {death_df['Is_Dot_Ball'].sum()} ({death_df['Is_Dot_Ball'].mean()*100:.2f}%)")
print(f"Balls with pressure applied: {death_df['Pressure_Applied'].sum()}")
print(f"Wickets: {death_df['Is_Wicket'].sum()} ({death_df['Is_Wicket'].mean()*100:.2f}%)")

# Key metric: Wicket probability after pressure
print("\n" + "-"*80)
print("KEY METRIC: Wicket Probability After Pressure (Death Overs)")
print("-"*80)

for bowler in ['Bowler A', 'Bowler B']:
    bowler_death = death_df[death_df['Bowler'] == bowler]
    
    # Wickets after pressure
    pressure_balls = bowler_death[bowler_death['Pressure_Applied'] == 1]
    wickets_after_pressure = pressure_balls['Is_Wicket'].sum()
    total_pressure_balls = len(pressure_balls)
    
    # Wickets without pressure
    no_pressure_balls = bowler_death[bowler_death['Pressure_Applied'] == 0]
    wickets_no_pressure = no_pressure_balls['Is_Wicket'].sum()
    total_no_pressure_balls = len(no_pressure_balls)
    
    print(f"\n{bowler}:")
    print(f"  After Pressure: {wickets_after_pressure}/{total_pressure_balls} = "
          f"{wickets_after_pressure/total_pressure_balls*100 if total_pressure_balls > 0 else 0:.2f}%")
    print(f"  No Pressure: {wickets_no_pressure}/{total_no_pressure_balls} = "
          f"{wickets_no_pressure/total_no_pressure_balls*100 if total_no_pressure_balls > 0 else 0:.2f}%")
    print(f"  Lift: {(wickets_after_pressure/total_pressure_balls - wickets_no_pressure/total_no_pressure_balls)*100 if total_pressure_balls > 0 else 0:.2f} percentage points")


print("\n" + "="*80)
print("PHASE 3: BAYESIAN MODELING - QUANTIFYING KILLER INSTINCT")
print("="*80)

# Prepare data for modeling
# Encode categorical variables
death_df['Pitch_Batting'] = (death_df['Pitch_Type'] == 'Batting').astype(int)
death_df['Pitch_Bowling'] = (death_df['Pitch_Type'] == 'Bowling').astype(int)
# Neutral is the reference category (both = 0)

death_df['Bowler_B'] = (death_df['Bowler'] == 'Bowler B').astype(int)

# Standardize continuous variables for better sampling
death_df['Batter_Avg_Std'] = (death_df['Batter_Avg'] - death_df['Batter_Avg'].mean()) / death_df['Batter_Avg'].std()

def run_bayesian_model():
    print("\nBuilding Bayesian Logistic Regression Model...")
    print("Target: Is_Wicket")
    print("Predictors:")
    print("  - Pressure_Applied (our key variable)")
    print("  - Bowler_B (Bowler A is reference)")
    print("  - Pressure_Applied × Bowler_B (interaction term - THE KILLER INSTINCT)")
    print("  - Pitch_Batting, Pitch_Bowling (controls)")
    print("  - Batter_Avg_Std (control for batsman quality)")

    # Build the model using PyMC
    import pymc as pm

    # Prepare data
    y = death_df['Is_Wicket'].values
    X_pressure = death_df['Pressure_Applied'].values
    X_bowler_b = death_df['Bowler_B'].values
    X_interaction = X_pressure * X_bowler_b  # The Killer Instinct term
    X_pitch_batting = death_df['Pitch_Batting'].values
    X_pitch_bowling = death_df['Pitch_Bowling'].values
    X_batter_avg = death_df['Batter_Avg_Std'].values

    print(f"\nModel data shape: {len(y)} observations")
    print(f"Wickets: {y.sum()} ({y.mean()*100:.2f}%)")

    # Build the model
    with pm.Model() as model:
        # Priors
        # Intercept: log-odds of wicket in baseline condition
        # Base wicket rate ~7%, so log(0.07/0.93) ≈ -2.6
        intercept = pm.Normal('intercept', mu=-2.5, sigma=1)
        
        # Pressure effect (main effect)
        # We expect this might be negative or neutral for average bowler
        beta_pressure = pm.Normal('beta_pressure', mu=0, sigma=1)
        
        # Bowler B effect (main effect)
        # Difference in baseline performance
        beta_bowler_b = pm.Normal('beta_bowler_b', mu=0, sigma=1)
        
        # THE KILLER INSTINCT: Interaction between Pressure and Bowler B
        # This is what we're really interested in!
        beta_killer_instinct = pm.Normal('beta_killer_instinct', mu=0, sigma=1.5)
        
        # Control variables
        beta_pitch_batting = pm.Normal('beta_pitch_batting', mu=0, sigma=0.5)
        beta_pitch_bowling = pm.Normal('beta_pitch_bowling', mu=0, sigma=0.5)
        beta_batter_avg = pm.Normal('beta_batter_avg', mu=0, sigma=0.5)
        
        # Linear combination
        logit_p = (intercept + 
                   beta_pressure * X_pressure +
                   beta_bowler_b * X_bowler_b +
                   beta_killer_instinct * X_interaction +
                   beta_pitch_batting * X_pitch_batting +
                   beta_pitch_bowling * X_pitch_bowling +
                   beta_batter_avg * X_batter_avg)
        
        # Likelihood
        p = pm.math.sigmoid(logit_p)
        y_obs = pm.Bernoulli('y_obs', p=p, observed=y)
        
        # Sample from posterior
        print("\nSampling from posterior distribution...")
        print("(This may take a few minutes...)")
        trace = pm.sample(2000, tune=1000, random_seed=42, return_inferencedata=True, cores=1)

    print("\nSampling complete!")
    
    # Analyze results
    print("\n" + "="*80)
    print("PHASE 4: RESULTS ANALYSIS - THE VERDICT")
    print("="*80)
    
    import arviz as az
    
    # Print summary statistics
    print("\nPosterior Summary:")
    print(az.summary(trace, var_names=['intercept', 'beta_pressure', 'beta_bowler_b', 
                                        'beta_killer_instinct', 'beta_pitch_batting', 
                                        'beta_pitch_bowling', 'beta_batter_avg'],
                     hdi_prob=0.94))
    
    # Extract the killer instinct coefficient
    killer_instinct_samples = trace.posterior['beta_killer_instinct'].values.flatten()
    
    print("\n" + "-"*80)
    print("THE KILLER INSTINCT COEFFICIENT (beta_killer_instinct)")
    print("-"*80)
    print(f"Mean: {killer_instinct_samples.mean():.4f}")
    print(f"Median: {np.median(killer_instinct_samples):.4f}")
    print(f"Std Dev: {killer_instinct_samples.std():.4f}")
    
    # Calculate 94% HDI
    hdi_94 = az.hdi(trace, var_names=['beta_killer_instinct'], hdi_prob=0.94)
    hdi_lower = float(hdi_94['beta_killer_instinct'].values[0])
    hdi_upper = float(hdi_94['beta_killer_instinct'].values[1])
    
    print(f"\n94% High Density Interval: [{hdi_lower:.4f}, {hdi_upper:.4f}]")
    
    # Check if 0 is in the HDI
    if hdi_lower > 0:
        print("\n✓ The 94% HDI does NOT include zero!")
        print("  This means we have strong evidence that Bowler B has 'Killer Instinct'")
        print("  - Bowler B significantly increases wicket probability after applying pressure")
    else:
        print("\n✗ The 94% HDI includes zero")
        print("  We cannot conclusively prove the 'Killer Instinct' effect")
    
    # Probability that killer instinct > 0
    prob_positive = (killer_instinct_samples > 0).mean()
    print(f"\nProbability that Killer Instinct effect > 0: {prob_positive*100:.2f}%")
    
    # Compare bowlers
    print("\n" + "-"*80)
    print("BOWLER COMPARISON")
    print("-"*80)
    
    # For Bowler A (reference): Effect of pressure = beta_pressure
    # For Bowler B: Effect of pressure = beta_pressure + beta_killer_instinct
    
    pressure_effect_a = trace.posterior['beta_pressure'].values.flatten()
    pressure_effect_b = pressure_effect_a + killer_instinct_samples
    
    print(f"\nBowler A - Pressure Effect:")
    print(f"  Mean: {pressure_effect_a.mean():.4f}")
    print(f"  94% HDI: [{az.hdi(trace, var_names=['beta_pressure'], hdi_prob=0.94)['beta_pressure'].values[0]:.4f}, "
          f"{az.hdi(trace, var_names=['beta_pressure'], hdi_prob=0.94)['beta_pressure'].values[1]:.4f}]")
    
    print(f"\nBowler B - Pressure Effect (includes Killer Instinct):")
    print(f"  Mean: {pressure_effect_b.mean():.4f}")
    hdi_b = np.percentile(pressure_effect_b, [3, 97])
    print(f"  94% HDI: [{hdi_b[0]:.4f}, {hdi_b[1]:.4f}]")
    
    # Interpret in terms of probability
    print("\n" + "-"*80)
    print("INTERPRETATION: What does this mean in practice?")
    print("-"*80)
    
    # Convert log-odds to probability change
    # For a typical scenario (neutral pitch, average batsman)
    baseline_logit = trace.posterior['intercept'].values.flatten().mean()
    
    # Bowler A after pressure
    logit_a_pressure = baseline_logit + pressure_effect_a.mean()
    prob_a_pressure = 1 / (1 + np.exp(-logit_a_pressure))
    
    # Bowler A no pressure
    prob_a_no_pressure = 1 / (1 + np.exp(-baseline_logit))
    
    # Bowler B after pressure
    bowler_b_effect = trace.posterior['beta_bowler_b'].values.flatten().mean()
    logit_b_pressure = baseline_logit + bowler_b_effect + pressure_effect_b.mean()
    prob_b_pressure = 1 / (1 + np.exp(-logit_b_pressure))
    
    # Bowler B no pressure
    logit_b_no_pressure = baseline_logit + bowler_b_effect
    prob_b_no_pressure = 1 / (1 + np.exp(-logit_b_no_pressure))
    
    print(f"\nBowler A:")
    print(f"  Wicket probability (no pressure): {prob_a_no_pressure*100:.2f}%")
    print(f"  Wicket probability (after pressure): {prob_a_pressure*100:.2f}%")
    print(f"  Change: {(prob_a_pressure - prob_a_no_pressure)*100:+.2f} percentage points")
    
    print(f"\nBowler B:")
    print(f"  Wicket probability (no pressure): {prob_b_no_pressure*100:.2f}%")
    print(f"  Wicket probability (after pressure): {prob_b_pressure*100:.2f}%")
    print(f"  Change: {(prob_b_pressure - prob_b_no_pressure)*100:+.2f} percentage points")
    
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    if hdi_lower > 0 and prob_positive > 0.95:
        print("\n🎯 RECOMMENDATION: BUY BOWLER B")
        print("\nThe data provides STRONG EVIDENCE for 'Killer Instinct':")
        print(f"  • Bowler B's wicket probability increases by {(prob_b_pressure - prob_b_no_pressure)*100:.1f} percentage points after pressure")
        print(f"  • Bowler A's wicket probability changes by {(prob_a_pressure - prob_a_no_pressure)*100:.1f} percentage points after pressure")
        print(f"  • The Killer Instinct effect is positive with {prob_positive*100:.1f}% probability")
        print(f"  • The 94% HDI excludes zero: [{hdi_lower:.3f}, {hdi_upper:.3f}]")
        print("\nCoach was RIGHT: Mental strength CAN be measured, and Bowler B has it!")
    else:
        print("\n⚠️  RECOMMENDATION: FURTHER ANALYSIS NEEDED")
        print("\nThe evidence for 'Killer Instinct' is not conclusive")
    
    # Save the trace for visualization
    trace.to_netcdf('bayesian_trace.nc')
    print("\n✓ Trace saved to 'bayesian_trace.nc'")
    
    return trace, model

if __name__ == '__main__':
    trace, model = run_bayesian_model()
