import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import matplotlib.ticker as mtick
warnings.filterwarnings("ignore")

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
ERREUR_TRONCATURE_PX = np.linspace(0.0, 0.99, 20) 

# =====================================================================
# --- 2. FONCTIONS DE SIMULATION ---
# =====================================================================

def run_study_1_gaussian(sigma_px, fixed_R):
    n_trials = 2000
    err_v = {s: [] for s in SCENARIOS}
    err_theta = {s: [] for s in SCENARIOS}
    
    for _ in range(n_trials):
        # 1. Calcul des positions PARFAITES (Ground Truth)
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

        noise_i = np.random.normal(0, sigma_px) * PIXEL_SIZE
        noise_D = (D1_GT**2 / (F * B)) * (2 * np.random.normal(0, sigma_px) * PIXEL_SIZE)

        for scenario in SCENARIOS:
       
            i1, i2, D1, D2 = i1_m_gt, i2_m_gt, D1_GT, D2_gt
            
            if scenario == 'Erreurs_i' or scenario == 'Combine': 
                i1 += noise_i
                i2 += noise_i
            if scenario == 'Erreurs_D' or scenario == 'Combine': 
                D1 += noise_D
                D2 += noise_D

            if abs(D2 - D1) < 1e-8:
                theta_est = 0.0
                x1_est, x2_est = -i1 * D1 / F, -i2 * D2 / F
                v_est = np.sqrt((x1_est - x2_est)**2 + (D1 - D2)**2) / DT
            else:
                denom_commun = (i1 * D1 + i2 * D2)**2 + (F**2) * (D2 - D1)**2
                
                if abs(denom_commun) < 1e-12:
                    val_sin = 0.0
                    val_cos = 1.0
                else:
                    num_sin = -2 * F * (D2 - D1) * (i2 * D2 + i1 * D1)
                    val_sin = num_sin / denom_commun
                    num_cos = (i1 * D1 + i2 * D2)**2 - (F**2) * (D2 - D1)**2
                    val_cos = num_cos / denom_commun
                
                theta_est = np.arctan2(val_sin, val_cos)
                
                num_R = (i1 * D1)**2 - (i2 * D2)**2 + (F**2) * (D1**2 - D2**2)
                R_est = num_R / (2 * (F**2) * (D2 - D1))
                
                v_est = abs((R_est / DT) * theta_est)

            err_v[scenario].append(abs(v_est - V_REAL))
            err_theta[scenario].append(abs(np.degrees(theta_est - theta_gt))) 
            
    stats_v = {s: (np.median(err_v[s]), np.percentile(err_v[s], 25), np.percentile(err_v[s], 75)) for s in SCENARIOS}
    stats_theta = {s: (np.median(err_theta[s]), np.percentile(err_theta[s], 25), np.percentile(err_theta[s], 75)) for s in SCENARIOS}
    return stats_v, stats_theta

def run_study_2_quantization(delta_i_px, fixed_R):
    n_trials = 2000
    err_v = {s: [] for s in SCENARIOS}
    err_theta = {s: [] for s in SCENARIOS}
    
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
        err_troncature_m = delta_i_px * PIXEL_SIZE
        err_D_m = (D1_GT**2 / (F * B)) * err_troncature_m

        for scenario in SCENARIOS:
        
            i1, i2, D1, D2 = i1_m_gt, i2_m_gt, D1_GT, D2_gt
            
            if scenario == 'Erreurs_i' or scenario == 'Combine': 
                i1 -= err_troncature_m
                i2 -= err_troncature_m
            if scenario == 'Erreurs_D' or scenario == 'Combine': 
                D1 += err_D_m
                D2 += err_D_m

            if abs(D2 - D1) < 1e-8:
                theta_est = 0.0
                x1_est, x2_est = -i1 * D1 / F, -i2 * D2 / F
                v_est = np.sqrt((x1_est - x2_est)**2 + (D1 - D2)**2) / DT
            else:
                denom_commun = (i1 * D1 + i2 * D2)**2 + (F**2) * (D2 - D1)**2
                
                if abs(denom_commun) < 1e-12:
                    val_sin = 0.0
                    val_cos = 1.0
                else:
                    num_sin = -2 * F * (D2 - D1) * (i2 * D2 + i1 * D1)
                    val_sin = num_sin / denom_commun
                    num_cos = (i1 * D1 + i2 * D2)**2 - (F**2) * (D2 - D1)**2
                    val_cos = num_cos / denom_commun
                
                theta_est = np.arctan2(val_sin, val_cos)
                
                num_R = (i1 * D1)**2 - (i2 * D2)**2 + (F**2) * (D1**2 - D2**2)
                R_est = num_R / (2 * (F**2) * (D2 - D1))
                
                v_est = abs((R_est / DT) * theta_est)
            
            err_v[scenario].append(abs(v_est - V_REAL))
            err_theta[scenario].append(abs(np.degrees(theta_est - theta_gt)))
                    
    stats_v = {s: (np.median(err_v[s]), np.percentile(err_v[s], 25), np.percentile(err_v[s], 75)) for s in SCENARIOS}
    stats_theta = {s: (np.median(err_theta[s]), np.percentile(err_theta[s], 25), np.percentile(err_theta[s], 75)) for s in SCENARIOS}
    return stats_v, stats_theta

# =====================================================================
# --- 3. EXÉCUTION ET AFFICHAGE ---
# =====================================================================

marge_legend = mpatches.Patch(color='grey', alpha=0.3, label='Marges d\'erreurs')

for idx_r, current_R in enumerate(R_VALUES):
    print(f"Calculs en cours pour {R_LABELS[idx_r]}")
    
    data_v_gauss = {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS}
    data_t_gauss = {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS}
    for p in NOISE_LEVELS:
        sv, st = run_study_1_gaussian(p, current_R)
        for s in SCENARIOS:
            data_v_gauss[s]['med'].append(sv[s][0]); data_v_gauss[s]['q25'].append(sv[s][1]); data_v_gauss[s]['q75'].append(sv[s][2])
            data_t_gauss[s]['med'].append(st[s][0]); data_t_gauss[s]['q25'].append(st[s][1]); data_t_gauss[s]['q75'].append(st[s][2])

    data_v_quant = {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS}
    data_t_quant = {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS}
    
    for delta_i in ERREUR_TRONCATURE_PX:
        sv, st = run_study_2_quantization(delta_i, current_R)
        for s in SCENARIOS:
            data_v_quant[s]['med'].append(sv[s][0]); data_v_quant[s]['q25'].append(sv[s][1]); data_v_quant[s]['q75'].append(sv[s][2])
            data_t_quant[s]['med'].append(st[s][0]); data_t_quant[s]['q25'].append(st[s][1]); data_t_quant[s]['q75'].append(st[s][2])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.canvas.manager.set_window_title(f"Simulation Discrète - {R_LABELS[idx_r]}")
    fig.suptitle(f"Erreur d'estimation (Méthode Discrète) : {R_LABELS[idx_r]}", fontsize=18, fontweight='bold')

    for s in SCENARIOS:
        axes[0,0].plot(NOISE_LEVELS, data_v_gauss[s]['med'], marker=STYLE[s]['marker'], color=STYLE[s]['color'], label=STYLE[s]['label'])
        axes[0,1].plot(NOISE_LEVELS, data_t_gauss[s]['med'], marker=STYLE[s]['marker'], color=STYLE[s]['color'], label=STYLE[s]['label'])
        axes[1,0].plot(ERREUR_TRONCATURE_PX, data_v_quant[s]['med'], marker=STYLE[s]['marker'], color=STYLE[s]['color'], label=STYLE[s]['label'])
        axes[1,1].plot(ERREUR_TRONCATURE_PX, data_t_quant[s]['med'], marker=STYLE[s]['marker'], color=STYLE[s]['color'], label=STYLE[s]['label'])
        
        axes[0,0].fill_between(NOISE_LEVELS, data_v_gauss[s]['q25'], data_v_gauss[s]['q75'], color=STYLE[s]['color'], alpha=0.15)
        axes[0,1].fill_between(NOISE_LEVELS, data_t_gauss[s]['q25'], data_t_gauss[s]['q75'], color=STYLE[s]['color'], alpha=0.15)
        axes[1,0].fill_between(ERREUR_TRONCATURE_PX, data_v_quant[s]['q25'], data_v_quant[s]['q75'], color=STYLE[s]['color'], alpha=0.15)
        axes[1,1].fill_between(ERREUR_TRONCATURE_PX, data_t_quant[s]['q25'], data_t_quant[s]['q75'], color=STYLE[s]['color'], alpha=0.15)

    axes[0,0].set_title("Vitesse (Bruit Gaussien)")
    axes[0,0].set_ylabel("Erreur absolue vitesse [m/s]")
    axes[0,0].set_xlabel("Écart-type du bruit [Pixels]") 

    axes[0,1].set_title(f"Orientation \u03B8 (Bruit Gaussien)")
    axes[0,1].set_ylabel(f"Erreur absolue orientation [°]")
    axes[0,1].set_xlabel("Écart-type du bruit [Pixels]") 
    
    axes[1,0].set_title("Vitesse (Discrétisation)")
    axes[1,0].set_ylabel("Erreur absolue vitesse [m/s]")
    axes[1,0].set_xlabel("Erreur de troncature \u03B4i [Pixels]") 

    axes[1,1].set_title(f"Orientation \u03B8 (Discrétisation)")
    axes[1,1].set_ylabel(f"Erreur absolue orientation [°]")
    axes[1,1].set_xlabel("Erreur de troncature \u03B4i [Pixels]") 

    for ax in axes.flatten():
            ax.axhline(0, color='red', linestyle='--', alpha=0.6) 
            ax.grid(True, alpha=0.3)
        
            def clean_format(x, pos):
                if x == 0: return '0'
                return f"{x:.8f}".rstrip('0').rstrip('.')

            ax.yaxis.set_major_formatter(mtick.FuncFormatter(clean_format))

            handles, labels = ax.get_legend_handles_labels()
            if 'Marges d\'erreurs' not in labels:
                handles.append(marge_legend)
                labels.append(marge_legend.get_label())
            ax.legend(handles=handles, labels=labels, fontsize='small')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()