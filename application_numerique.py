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
DT = 0.02            # Pas de temps (s)
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

NOISE_LEVELS = np.linspace(0.0, 0.7, 10)
ERREUR_TRONCATURE_PX = np.linspace(0.0, 0.99, 20) 

# =====================================================================
# --- 2. FONCTIONS DE SIMULATION ---
# =====================================================================

def run_study_1_gaussian(sigma_px, fixed_R):
    n_trials = 2000
    err_v = {s: [] for s in SCENARIOS}
    err_theta = {s: [] for s in SCENARIOS}
    err_p = {s: [] for s in SCENARIOS} 
    
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

        # Vérité Terrain (Vecteur Position)
        x3_gt = x2_gt * np.cos(theta_gt) + D2_gt * np.sin(theta_gt)
        z3_gt = -x2_gt * np.sin(theta_gt) + D2_gt * np.cos(theta_gt)
        Tx_gt = x1_gt - x3_gt
        Tz_gt = D1_GT - z3_gt

        noise_i1 = np.random.normal(0, sigma_px) * PIXEL_SIZE
        noise_i2 = np.random.normal(0, sigma_px) * PIXEL_SIZE
        noise_D1 = (D1_GT**2 / (F * B)) * (2 * np.random.normal(0, sigma_px) * PIXEL_SIZE)
        noise_D2 = (D2_gt**2 / (F * B)) * (2 * np.random.normal(0, sigma_px) * PIXEL_SIZE)

        for scenario in SCENARIOS:
            i1, i2, D1, D2 = i1_m_gt, i2_m_gt, D1_GT, D2_gt
            
            if scenario == 'Erreurs_i' or scenario == 'Combine': 
                i1 += noise_i1
                i2 += noise_i2
            if scenario == 'Erreurs_D' or scenario == 'Combine': 
                D1 += noise_D1
                D2 += noise_D2

            if abs(D2 - D1) < 1e-8:
                theta_est = 0.0
                x1_est, x2_est = -i1 * D1 / F, -i2 * D2 / F
                v_est = np.sqrt((x1_est - x2_est)**2 + (D1 - D2)**2) / DT
                
                Tx_est = v_est * DT
                Tz_est = 0.0
            else:
                denom_commun = (i1 * D1 + i2 * D2)**2 + (F**2) * (D2 - D1)**2
                
                if abs(denom_commun) < 1e-12:
                    val_sin, val_cos = 0.0, 1.0
                    R_est = 1e6
                else:
                    num_sin = -2 * F * (D2 - D1) * (i2 * D2 + i1 * D1)
                    val_sin = num_sin / denom_commun
                    num_cos = (i1 * D1 + i2 * D2)**2 - (F**2) * (D2 - D1)**2
                    val_cos = num_cos / denom_commun
                
                theta_est = np.arctan2(val_sin, val_cos)
                
                num_R = (i1 * D1)**2 - (i2 * D2)**2 + (F**2) * (D1**2 - D2**2)
                denom_R = 2 * (F**2) * (D2 - D1)
                R_est = num_R / denom_R if abs(denom_R) >= 1e-15 else 1e6
                
                # Norme de la vitesse estimée
                v_est = abs((R_est / DT) * theta_est)

                # ================================================================
                # Reconstruction directe de la position via Vitesse et Orientation
                # ================================================================
                distance = v_est * DT
                if abs(theta_est) < 1e-8:
                    Tx_est = distance
                    Tz_est = 0.0
                else:
                    Tx_est = distance * (np.sin(theta_est) / theta_est)
                    Tz_est = distance * ((1 - np.cos(theta_est)) / theta_est)

            err_v[scenario].append(abs(v_est - V_REAL))
            err_theta[scenario].append(abs(np.degrees(theta_est - theta_gt))) 
            err_p[scenario].append(np.sqrt((Tx_est - Tx_gt)**2 + (Tz_est - Tz_gt)**2))
            
    stats_v = {s: (np.median(err_v[s]), np.percentile(err_v[s], 25), np.percentile(err_v[s], 75)) for s in SCENARIOS}
    stats_theta = {s: (np.median(err_theta[s]), np.percentile(err_theta[s], 25), np.percentile(err_theta[s], 75)) for s in SCENARIOS}
    stats_p = {s: (np.median(err_p[s]), np.percentile(err_p[s], 25), np.percentile(err_p[s], 75)) for s in SCENARIOS}
    return stats_v, stats_theta, stats_p

def run_study_2_quantization(delta_i_px, fixed_R):
    n_trials = 2000
    err_v = {s: [] for s in SCENARIOS}
    err_theta = {s: [] for s in SCENARIOS}
    err_p = {s: [] for s in SCENARIOS} 
    
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
        
        # Vérité Terrain (Vecteur Position)
        x3_gt = x2_gt * np.cos(theta_gt) + D2_gt * np.sin(theta_gt)
        z3_gt = -x2_gt * np.sin(theta_gt) + D2_gt * np.cos(theta_gt)
        Tx_gt = x1_gt - x3_gt
        Tz_gt = D1_GT - z3_gt

        noise_i1 = np.random.uniform(-0.5, 0.5) * delta_i_px * PIXEL_SIZE
        noise_i2 = np.random.uniform(-0.5, 0.5) * delta_i_px * PIXEL_SIZE
        noise_D1 = (D1_GT**2 / (F * B)) * (2 * np.random.uniform(-0.5, 0.5) * delta_i_px * PIXEL_SIZE)
        noise_D2 = (D2_gt**2 / (F * B)) * (2 * np.random.uniform(-0.5, 0.5) * delta_i_px * PIXEL_SIZE)

        for scenario in SCENARIOS:
            i1, i2, D1, D2 = i1_m_gt, i2_m_gt, D1_GT, D2_gt
            
            if scenario == 'Erreurs_i' or scenario == 'Combine': 
                i1 += noise_i1
                i2 += noise_i2
            if scenario == 'Erreurs_D' or scenario == 'Combine': 
                D1 += noise_D1
                D2 += noise_D2

            if abs(D2 - D1) < 1e-8:
                theta_est = 0.0
                x1_est, x2_est = -i1 * D1 / F, -i2 * D2 / F
                v_est = np.sqrt((x1_est - x2_est)**2 + (D1 - D2)**2) / DT
                
                Tx_est = v_est * DT
                Tz_est = 0.0
            else:
                denom_commun = (i1 * D1 + i2 * D2)**2 + (F**2) * (D2 - D1)**2
                
                if abs(denom_commun) < 1e-12:
                    val_sin, val_cos = 0.0, 1.0
                    R_est = 1e6
                else:
                    num_sin = -2 * F * (D2 - D1) * (i2 * D2 + i1 * D1)
                    val_sin = num_sin / denom_commun
                    num_cos = (i1 * D1 + i2 * D2)**2 - (F**2) * (D2 - D1)**2
                    val_cos = num_cos / denom_commun
                
                theta_est = np.arctan2(val_sin, val_cos)
                
                num_R = (i1 * D1)**2 - (i2 * D2)**2 + (F**2) * (D1**2 - D2**2)
                denom_R = 2 * (F**2) * (D2 - D1)
                R_est = num_R / denom_R if abs(denom_R) >= 1e-15 else 1e6
                
                # Norme de la vitesse estimée
                v_est = abs((R_est / DT) * theta_est)

                # ================================================================
                # Reconstruction directe de la position via Vitesse et Orientation
                # ================================================================
                distance = v_est * DT
                if abs(theta_est) < 1e-8:
                    Tx_est = distance
                    Tz_est = 0.0
                else:
                    Tx_est = distance * (np.sin(theta_est) / theta_est)
                    Tz_est = distance * ((1 - np.cos(theta_est)) / theta_est)

            err_v[scenario].append(abs(v_est - V_REAL))
            err_theta[scenario].append(abs(np.degrees(theta_est - theta_gt)))
            err_p[scenario].append(np.sqrt((Tx_est - Tx_gt)**2 + (Tz_est - Tz_gt)**2))
                    
    stats_v = {s: (np.median(err_v[s]), np.percentile(err_v[s], 25), np.percentile(err_v[s], 75)) for s in SCENARIOS}
    stats_theta = {s: (np.median(err_theta[s]), np.percentile(err_theta[s], 25), np.percentile(err_theta[s], 75)) for s in SCENARIOS}
    stats_p = {s: (np.median(err_p[s]), np.percentile(err_p[s], 25), np.percentile(err_p[s], 75)) for s in SCENARIOS}
    return stats_v, stats_theta, stats_p

# =====================================================================
# --- 3. GÉNÉRATION DES DONNÉES ---
# =====================================================================

print("Début des simulations... Cela peut prendre quelques instants.")

data_all_R = {}

for current_R, r_label in zip(R_VALUES, R_LABELS):
    print(f"-> Calculs en cours pour {r_label}")
    res = {
        'v_gauss': {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS},
        't_gauss': {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS},
        'p_gauss': {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS},
        'v_quant': {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS},
        't_quant': {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS},
        'p_quant': {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS}
    }
    
    # 1. Bruit Gaussien
    for p in NOISE_LEVELS:
        sv, st, sp = run_study_1_gaussian(p, current_R)
        for s in SCENARIOS:
            res['v_gauss'][s]['med'].append(sv[s][0]); res['v_gauss'][s]['q25'].append(sv[s][1]); res['v_gauss'][s]['q75'].append(sv[s][2])
            res['t_gauss'][s]['med'].append(st[s][0]); res['t_gauss'][s]['q25'].append(st[s][1]); res['t_gauss'][s]['q75'].append(st[s][2])
            res['p_gauss'][s]['med'].append(sp[s][0]); res['p_gauss'][s]['q25'].append(sp[s][1]); res['p_gauss'][s]['q75'].append(sp[s][2])

    # 2. Bruit de Discrétisation
    for delta_i in ERREUR_TRONCATURE_PX:
        sv, st, sp = run_study_2_quantization(delta_i, current_R)
        for s in SCENARIOS:
            res['v_quant'][s]['med'].append(sv[s][0]); res['v_quant'][s]['q25'].append(sv[s][1]); res['v_quant'][s]['q75'].append(sv[s][2])
            res['t_quant'][s]['med'].append(st[s][0]); res['t_quant'][s]['q25'].append(st[s][1]); res['t_quant'][s]['q75'].append(st[s][2])
            res['p_quant'][s]['med'].append(sp[s][0]); res['p_quant'][s]['q25'].append(sp[s][1]); res['p_quant'][s]['q75'].append(sp[s][2])
            
    data_all_R[current_R] = res

# =====================================================================
# --- 4. AFFICHAGE DES 6 GRANDES FIGURES ---
# =====================================================================

marge_legend = mpatches.Patch(color='grey', alpha=0.3, label='Marges d\'erreurs')

configs_figures = [
    ("1 - Erreur Vitesse (Bruit Gaussien)", 'v_gauss', NOISE_LEVELS, "Écart-type du bruit stochastique [Pixels]", "Erreur absolue vitesse [m/s]"),
    ("2 - Erreur Orientation (Bruit Gaussien)", 't_gauss', NOISE_LEVELS, "Écart-type du bruit stochastique [Pixels]", "Erreur absolue orientation [°]"),
    ("3 - Erreur Position (Bruit Gaussien)", 'p_gauss', NOISE_LEVELS, "Écart-type du bruit stochastique [Pixels]", "Erreur absolue position [m]"),
    ("4 - Erreur Vitesse (Discrétisation)", 'v_quant', ERREUR_TRONCATURE_PX, "Erreur de discrétisation δi [Pixels]", "Erreur absolue vitesse [m/s]"),
    ("5 - Erreur Orientation (Discrétisation)", 't_quant', ERREUR_TRONCATURE_PX, "Erreur de discrétisation δi [Pixels]", "Erreur absolue orientation [°]"),
    ("6 - Erreur Position (Discrétisation)", 'p_quant', ERREUR_TRONCATURE_PX, "Erreur de discrétisation δi [Pixels]", "Erreur absolue position [m]")
]

for title, data_key, x_axis, x_label, y_label in configs_figures:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.canvas.manager.set_window_title(title)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for ax, current_R, r_label in zip(axes.flatten(), R_VALUES, R_LABELS):
        data = data_all_R[current_R][data_key]
        
        for s in SCENARIOS:
            ax.plot(x_axis, data[s]['med'], marker=STYLE[s]['marker'], color=STYLE[s]['color'], label=STYLE[s]['label'])
            ax.fill_between(x_axis, data[s]['q25'], data[s]['q75'], color=STYLE[s]['color'], alpha=0.15)
            
        ax.set_title(r_label, fontsize=12, fontweight='bold')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.axhline(0, color='red', linestyle='--', alpha=0.6) 
        ax.grid(True, alpha=0.3)
        
        if "Position" in title:
            ax.set_ylim(0, 1.6 if "Discrétisation" in title else 3.5)
            
        def clean_format(x, pos):
            if x == 0: return '0'
            return f"{x:.2f}".rstrip('0').rstrip('.') if "Position" in title else f"{x:.8f}".rstrip('0').rstrip('.')
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(clean_format))
        
        handles, labels = ax.get_legend_handles_labels()
        if 'Marges d\'erreurs' not in labels:
            handles.append(marge_legend)
            labels.append(marge_legend.get_label())
        ax.legend(handles=handles, labels=labels, fontsize='small', loc='upper left')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()