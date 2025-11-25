"""
Create visualizations for the IPL Auction Analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

# Load the trace
trace = az.from_netcdf('bayesian_trace.nc')

# Create a comprehensive figure
fig = plt.figure(figsize=(16, 12))

# 1. Posterior distribution of Killer Instinct coefficient
ax1 = plt.subplot(2, 3, 1)
killer_instinct = trace.posterior['beta_killer_instinct'].values.flatten()
ax1.hist(killer_instinct, bins=50, alpha=0.7, color='darkred', edgecolor='black')
ax1.axvline(killer_instinct.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {killer_instinct.mean():.3f}')
ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Zero (No Effect)')

# Add HDI
hdi = np.percentile(killer_instinct, [3, 97])
ax1.axvspan(hdi[0], hdi[1], alpha=0.2, color='red', label=f'94% HDI: [{hdi[0]:.2f}, {hdi[1]:.2f}]')

ax1.set_xlabel('Killer Instinct Coefficient (β)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Posterior Distribution: Killer Instinct Effect\n(Pressure × Bowler B Interaction)', 
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# 2. Forest plot comparing all coefficients
ax2 = plt.subplot(2, 3, 2)
az.plot_forest(trace, var_names=['intercept', 'beta_pressure', 'beta_bowler_b', 
                                  'beta_killer_instinct', 'beta_pitch_batting', 
                                  'beta_pitch_bowling', 'beta_batter_avg'],
               hdi_prob=0.94, ax=ax2, colors='darkblue')
ax2.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax2.set_title('Forest Plot: All Model Coefficients\n(94% HDI)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')

# 3. Comparison of pressure effects
ax3 = plt.subplot(2, 3, 3)
pressure_a = trace.posterior['beta_pressure'].values.flatten()
pressure_b = pressure_a + killer_instinct

data_to_plot = [pressure_a, pressure_b]
positions = [1, 2]
bp = ax3.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                 labels=['Bowler A\n(The Machine)', 'Bowler B\n(The Gambler)'])

# Color the boxes
colors = ['lightblue', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_ylabel('Pressure Effect Coefficient', fontsize=12, fontweight='bold')
ax3.set_title('Pressure Effect Comparison\n(Higher = More Wickets After Pressure)', 
              fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 4. Probability interpretation
ax4 = plt.subplot(2, 3, 4)

# Calculate probabilities
baseline_logit = trace.posterior['intercept'].values.flatten().mean()
bowler_b_effect = trace.posterior['beta_bowler_b'].values.flatten().mean()

# Bowler A
prob_a_no_pressure = 1 / (1 + np.exp(-baseline_logit))
prob_a_pressure = 1 / (1 + np.exp(-(baseline_logit + pressure_a.mean())))

# Bowler B
prob_b_no_pressure = 1 / (1 + np.exp(-(baseline_logit + bowler_b_effect)))
prob_b_pressure = 1 / (1 + np.exp(-(baseline_logit + bowler_b_effect + pressure_b.mean())))

categories = ['No Pressure', 'After Pressure']
bowler_a_probs = [prob_a_no_pressure * 100, prob_a_pressure * 100]
bowler_b_probs = [prob_b_no_pressure * 100, prob_b_pressure * 100]

x = np.arange(len(categories))
width = 0.35

bars1 = ax4.bar(x - width/2, bowler_a_probs, width, label='Bowler A', color='lightblue', edgecolor='black')
bars2 = ax4.bar(x + width/2, bowler_b_probs, width, label='Bowler B', color='lightcoral', edgecolor='black')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

ax4.set_ylabel('Wicket Probability (%)', fontsize=12, fontweight='bold')
ax4.set_title('Wicket Probability: The Killer Instinct in Action', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(categories, fontsize=11)
ax4.legend(fontsize=11)
ax4.grid(axis='y', alpha=0.3)

# 5. Trace plot for convergence check
ax5 = plt.subplot(2, 3, 5)
for chain in range(trace.posterior.dims['chain']):
    ax5.plot(trace.posterior['beta_killer_instinct'].values[chain, :], alpha=0.7, label=f'Chain {chain+1}')
ax5.set_xlabel('Iteration', fontsize=11, fontweight='bold')
ax5.set_ylabel('Killer Instinct Coefficient', fontsize=11, fontweight='bold')
ax5.set_title('Trace Plot: Convergence Check', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)

# 6. Autocorrelation plot
ax6 = plt.subplot(2, 3, 6)
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(pd.Series(killer_instinct), ax=ax6, color='darkred')
ax6.set_title('Autocorrelation: Sample Independence', fontsize=12, fontweight='bold')
ax6.set_xlabel('Lag', fontsize=11, fontweight='bold')
ax6.set_ylabel('Autocorrelation', fontsize=11, fontweight='bold')
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('killer_instinct_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'killer_instinct_analysis.png'")

# Create a second figure for the executive summary
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: The key finding
ax_left = axes[0]
categories = ['No Pressure', 'After Dot Ball\n(Pressure Applied)']
bowler_a_probs = [prob_a_no_pressure * 100, prob_a_pressure * 100]
bowler_b_probs = [prob_b_no_pressure * 100, prob_b_pressure * 100]

x = np.arange(len(categories))
width = 0.35

bars1 = ax_left.bar(x - width/2, bowler_a_probs, width, label='Bowler A (The Machine)', 
                    color='#3498db', edgecolor='black', linewidth=1.5)
bars2 = ax_left.bar(x + width/2, bowler_b_probs, width, label='Bowler B (The Gambler)', 
                    color='#e74c3c', edgecolor='black', linewidth=1.5)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax_left.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax_left.set_ylabel('Wicket Probability (%)', fontsize=13, fontweight='bold')
ax_left.set_title('The "Killer Instinct" Quantified\nDeath Overs (16-20)', 
                  fontsize=14, fontweight='bold', pad=15)
ax_left.set_xticks(x)
ax_left.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax_left.legend(fontsize=11, loc='upper left')
ax_left.set_ylim(0, 45)
ax_left.grid(axis='y', alpha=0.3, linestyle='--')

# Add annotation
ax_left.annotate('', xy=(1.175, bowler_b_probs[1]), xytext=(1.175, bowler_b_probs[0]),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax_left.text(1.4, (bowler_b_probs[0] + bowler_b_probs[1])/2, 
            f'+{bowler_b_probs[1] - bowler_b_probs[0]:.1f}%\nKiller\nInstinct', 
            fontsize=11, fontweight='bold', color='red', va='center')

# Right: Posterior distribution
ax_right = axes[1]
ax_right.hist(killer_instinct, bins=40, alpha=0.8, color='#e74c3c', edgecolor='black', linewidth=1.2)
ax_right.axvline(killer_instinct.mean(), color='darkred', linestyle='--', linewidth=2.5, 
                label=f'Mean: {killer_instinct.mean():.2f}')
ax_right.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.7, label='Zero (No Effect)')

hdi = np.percentile(killer_instinct, [3, 97])
ax_right.axvspan(hdi[0], hdi[1], alpha=0.25, color='red', 
                label=f'94% HDI: [{hdi[0]:.2f}, {hdi[1]:.2f}]')

ax_right.set_xlabel('Killer Instinct Coefficient (β)', fontsize=13, fontweight='bold')
ax_right.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax_right.set_title('Bayesian Evidence: 100% Probability > 0\n"Mental Strength" is Real', 
                  fontsize=14, fontweight='bold', pad=15)
ax_right.legend(fontsize=10, loc='upper right')
ax_right.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('executive_summary_plot.png', dpi=300, bbox_inches='tight')
print("✓ Executive summary plot saved as 'executive_summary_plot.png'")

plt.show()
