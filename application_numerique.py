import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =====================================================================
# --- 1. CONFIGURATION GÉNÉRALE (Paramètres Physiques) ---
# =====================================================================
V_REAL = 2.78        # Vitesse réelle de référence (m/s)
DT = 0.02             # Pas de temps (s)
F = 0.035            # Focale de la caméra (m)
B = 0.20             # Baseline stéréoscopique (m)
PIXEL_SIZE = 5e-6    # Taille d'un pixel sur le capteur (m)
D1_GT = 10.0         # Distance initiale à la façade (m)

R_VALUES = [10, 50, 500, np.inf] 
R_LABELS = ["Virage Serré (R=10m)", "Virage Fluide (R=50m)", "Quasi-Ligne Droite (R=500m)", "Ligne Droite Pure (R \u2192 \u221E)"]

SCENARIOS = ['Erreurs_i', 'Erreurs_D', 'Combine']
STYLE = {
    'Erreurs_i': {'color': 'blue',   'marker': 'o', 'label': 'Erreurs sur $i_1, i_2$ uniquement'},
    'Erreurs_D': {'color': 'orange', 'marker': 's', 'label': 'Erreurs sur $D_1, D_2$ uniquement'},
    'Combine':   {'color': 'black',  'marker': 'X', 'label': 'Erreurs sur toutes les grandeurs'}
}

NOISE_LEVELS = np.linspace(0.0, 1.0, 10)
FRACTIONS_PIXEL = np.linspace(0.0, 0.99, 20)

# =====================================================================
# --- 2. FONCTIONS DE SIMULATION ---
# =====================================================================

def run_study_1_gaussian(sigma_px, fixed_R):
    n_trials = 2000
    err_v = {s: [] for s in SCENARIOS}
    err_R = {s: [] for s in SCENARIOS} 
    
    for _ in range(n_trials):
        i1_px_gt = np.random.uniform(-400, 400)
        i1_m_gt = i1_px_gt * PIXEL_SIZE
        x1_gt = -i1_m_gt * D1_GT / F
        
        if np.isinf(fixed_R):
            theta_gt, x2_gt, D2_gt = 0.0, x1_gt - V_REAL * DT, D1_GT
        else:
            theta_gt = V_REAL * DT / fixed_R
            x2_gt = x1_gt * np.cos(theta_gt) - (D1_GT + fixed_R) * np.sin(theta_gt)
            D2_gt = x1_gt * np.sin(theta_gt) + (D1_GT + fixed_R) * np.cos(theta_gt) - fixed_R
            
        i2_m_gt = -F * x2_gt / D2_gt

        noise_i1 = np.random.normal(0, sigma_px) * PIXEL_SIZE
        noise_i2 = np.random.normal(0, sigma_px) * PIXEL_SIZE
        noise_D1 = (D1_GT**2 / (F * B)) * (2 * np.random.normal(0, sigma_px) * PIXEL_SIZE)
        noise_D2 = (D2_gt**2 / (F * B)) * (2 * np.random.normal(0, sigma_px) * PIXEL_SIZE)

        for scenario in SCENARIOS:
            i1, i2, D1, D2 = i1_m_gt, i2_m_gt, D1_GT, D2_gt
            if scenario == 'Erreurs_i': i1 += noise_i1; i2 += noise_i2
            elif scenario == 'Erreurs_D': D1 += noise_D1; D2 += noise_D2
            elif scenario == 'Combine': i1 += noise_i1; i2 += noise_i2; D1 += noise_D1; D2 += noise_D2

            x1_est = -i1 * D1 / F
            x2_est = -i2 * D2 / F
            
            lateral_diff = (x1_est - x2_est) - (V_REAL * DT)
            
            if abs(lateral_diff) < 1e-7: 
                R_est = np.inf
            else:
                R_est = (D1 * V_REAL * DT) / lateral_diff
                
            v_est = np.sqrt((x1_est - x2_est)**2 + (D1 - D2)**2) / DT
            
            err_v[scenario].append(v_est - V_REAL)
            
            if np.isinf(fixed_R):
                err_R[scenario].append(1.0 / R_est if not np.isinf(R_est) else 0.0)
            else:
                err_R[scenario].append(R_est - fixed_R)
            
    stats_v = {s: (np.median(err_v[s]), np.percentile(err_v[s], 25), np.percentile(err_v[s], 75)) for s in SCENARIOS}
    stats_R = {s: (np.median(err_R[s]), np.percentile(err_R[s], 25), np.percentile(err_R[s], 75)) for s in SCENARIOS}
    return stats_v, stats_R

def run_study_2_quantization(fraction_px, fixed_R):
    n_trials = 2000
    err_v = {s: [] for s in SCENARIOS}
    err_R = {s: [] for s in SCENARIOS}
    
    for _ in range(n_trials):
        base_pixel = np.random.randint(-400, 400)
        i1_px_gt = base_pixel + fraction_px 
        i1_m_gt = i1_px_gt * PIXEL_SIZE
        x1_gt = -i1_m_gt * D1_GT / F
        
        if np.isinf(fixed_R):
            theta_gt, x2_gt, D2_gt = 0.0, x1_gt - V_REAL * DT, D1_GT
        else:
            theta_gt = V_REAL * DT / fixed_R
            x2_gt = x1_gt * np.cos(theta_gt) - (D1_GT + fixed_R) * np.sin(theta_gt)
            D2_gt = x1_gt * np.sin(theta_gt) + (D1_GT + fixed_R) * np.cos(theta_gt) - fixed_R
            
        i2_m_gt = -F * x2_gt / D2_gt
        i2_px_gt = i2_m_gt / PIXEL_SIZE

        di1, di2 = np.abs(i1_px_gt) - np.floor(np.abs(i1_px_gt)), np.abs(i2_px_gt) - np.floor(np.abs(i2_px_gt))
        sign1, sign2 = np.sign(i1_px_gt) if i1_px_gt != 0 else 1, np.sign(i2_px_gt) if i2_px_gt != 0 else 1
        err_i1, err_i2 = sign1 * di1 * PIXEL_SIZE, sign2 * di2 * PIXEL_SIZE
        err_D1, err_D2 = (D1_GT**2 / (F * B)) * (2 * di1 * PIXEL_SIZE), (D2_gt**2 / (F * B)) * (2 * di2 * PIXEL_SIZE)

        for scenario in SCENARIOS:
            i1, i2, D1, D2 = i1_m_gt, i2_m_gt, D1_GT, D2_gt
            if scenario == 'Erreurs_i': i1 -= err_i1; i2 -= err_i2
            elif scenario == 'Erreurs_D': D1 += err_D1; D2 += err_D2
            elif scenario == 'Combine': i1 -= err_i1; i2 -= err_i2; D1 += err_D1; D2 += err_D2

            x1_est = -i1 * D1 / F
            x2_est = -i2 * D2 / F
            
            lateral_diff = (x1_est - x2_est) - (V_REAL * DT)
            
            if abs(lateral_diff) < 1e-7: 
                R_est = np.inf
            else:
                R_est = (D1 * V_REAL * DT) / lateral_diff
                
            v_est = np.sqrt((x1_est - x2_est)**2 + (D1 - D2)**2) / DT
            
            err_v[scenario].append(v_est - V_REAL)
            
            if np.isinf(fixed_R):
                err_R[scenario].append(1.0 / R_est if not np.isinf(R_est) else 0.0)
            else:
                err_R[scenario].append(R_est - fixed_R)
            
    stats_v = {s: (np.median(err_v[s]), np.percentile(err_v[s], 25), np.percentile(err_v[s], 75)) for s in SCENARIOS}
    stats_R = {s: (np.median(err_R[s]), np.percentile(err_R[s], 25), np.percentile(err_R[s], 75)) for s in SCENARIOS}
    return stats_v, stats_R

# =====================================================================
# --- 3. EXÉCUTION ET AFFICHAGE ---
# =====================================================================

# Patch global pour indiquer la légende des marges
marge_legend = mpatches.Patch(color='grey', alpha=0.3, label='Marges d\'erreurs')

for idx_r, current_R in enumerate(R_VALUES):
    print(f"Calculs en cours pour {R_LABELS[idx_r]}...")
    
    data_v_gauss = {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS}
    data_R_gauss = {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS}
    for p in NOISE_LEVELS:
        sv, sr = run_study_1_gaussian(p, current_R)
        for s in SCENARIOS:
            data_v_gauss[s]['med'].append(sv[s][0]); data_v_gauss[s]['q25'].append(sv[s][1]); data_v_gauss[s]['q75'].append(sv[s][2])
            data_R_gauss[s]['med'].append(sr[s][0]); data_R_gauss[s]['q25'].append(sr[s][1]); data_R_gauss[s]['q75'].append(sr[s][2])

    data_v_quant = {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS}
    data_R_quant = {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS}
    for frac in FRACTIONS_PIXEL:
        sv, sr = run_study_2_quantization(frac, current_R)
        for s in SCENARIOS:
            data_v_quant[s]['med'].append(sv[s][0]); data_v_quant[s]['q25'].append(sv[s][1]); data_v_quant[s]['q75'].append(sv[s][2])
            data_R_quant[s]['med'].append(sr[s][0]); data_R_quant[s]['q25'].append(sr[s][1]); data_R_quant[s]['q75'].append(sr[s][2])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.canvas.manager.set_window_title(f"Simulation - {R_LABELS[idx_r]}")
    fig.suptitle(f"Erreur d'estimation : {R_LABELS[idx_r]}", fontsize=18, fontweight='bold')

    for s in SCENARIOS:
        # Courbes médianes
        axes[0,0].plot(NOISE_LEVELS, data_v_gauss[s]['med'], marker=STYLE[s]['marker'], color=STYLE[s]['color'], label=STYLE[s]['label'])
        axes[0,1].plot(NOISE_LEVELS, data_R_gauss[s]['med'], marker=STYLE[s]['marker'], color=STYLE[s]['color'], label=STYLE[s]['label'])
        axes[1,0].plot(FRACTIONS_PIXEL, data_v_quant[s]['med'], marker=STYLE[s]['marker'], color=STYLE[s]['color'], label=STYLE[s]['label'])
        axes[1,1].plot(FRACTIONS_PIXEL, data_R_quant[s]['med'], marker=STYLE[s]['marker'], color=STYLE[s]['color'], label=STYLE[s]['label'])
        
        # Zones 25% - 75%
        axes[0,0].fill_between(NOISE_LEVELS, data_v_gauss[s]['q25'], data_v_gauss[s]['q75'], color=STYLE[s]['color'], alpha=0.15)
        axes[0,1].fill_between(NOISE_LEVELS, data_R_gauss[s]['q25'], data_R_gauss[s]['q75'], color=STYLE[s]['color'], alpha=0.15)
        axes[1,0].fill_between(FRACTIONS_PIXEL, data_v_quant[s]['q25'], data_v_quant[s]['q75'], color=STYLE[s]['color'], alpha=0.15)
        axes[1,1].fill_between(FRACTIONS_PIXEL, data_R_quant[s]['q25'], data_R_quant[s]['q75'], color=STYLE[s]['color'], alpha=0.15)

    # Cosmétique et Ligne de Référence (0)
    axes[0,0].set_title("Vitesse (Gauss)"); axes[0,0].set_ylabel("Erreur norme vitesse (est - reel) [m/s]")
    axes[1,0].set_title("Vitesse (Discrétisation)"); axes[1,0].set_ylabel("Erreur norme vitesse (est - reel) [m/s]")
    
    label_r = "Courbure 1/R [m⁻¹]" if np.isinf(current_R) else "Rayon R [m]"
    axes[0,1].set_title(f"{label_r} (Gauss)"); axes[0,1].set_ylabel(f"Erreur {label_r} (est - reel)")
    axes[1,1].set_title(f"{label_r} (Discrétisation)"); axes[1,1].set_ylabel(f"Erreur {label_r} (est - reel)")

    for ax in axes.flatten():
        ax.axhline(0, color='red', linestyle='--', alpha=0.6) # LIGNE ROUGE À ZÉRO
        ax.grid(True, alpha=0.3)
    
        handles, labels = ax.get_legend_handles_labels()
        if 'Marges d\'erreurs' not in labels:
            handles.append(marge_legend)
            labels.append(marge_legend.get_label())
            
        ax.legend(handles=handles, labels=labels, fontsize='small')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()