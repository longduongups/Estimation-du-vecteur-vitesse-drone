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
DT = 0.5            # Pas de temps (s)
F = 0.035            # Focale de la caméra (m)
B = 0.20             # Baseline stéréoscopique (m)
PIXEL_SIZE = 5e-6    # Taille d'un pixel sur le capteur (m)
D1_GT = 10.0         # Distance initiale à la façade (m)

R_VALUES = [10, 50, 500, np.inf] 
R_LABELS = ["Virage Serré (R=10m)", "Virage Fluide (R=50m)", "Quasi-Ligne Droite (R=500m)", "Ligne Droite Pure (R \u2192 \u221E)"]

SCENARIOS = ['Erreurs_i', 'Erreurs_D', 'Combine']
STYLE = {
    'Erreurs_i': {'color': 'blue',   'marker': 'o', 'label': 'Erreurs sur $i_1, i_2$'},
    'Erreurs_D': {'color': 'orange', 'marker': 's', 'label': 'Erreurs sur $D_1, D_2$'},
    'Combine':   {'color': 'black',  'marker': 'X', 'label': 'Erreurs combinées'}
}

NOISE_LEVELS = np.linspace(0.0, 0.7, 10)
ERREUR_TRONCATURE_PX = np.linspace(0.0, 1.0, 20) 

# =====================================================================
# --- 2. FONCTIONS DE SIMULATION (MODÈLE 2 FACTORISÉ ASTEK) ---
# =====================================================================

def run_study_1_gaussian(sigma_px, fixed_R):
    n_trials = 2000
    err_p = {s: [] for s in SCENARIOS}
    
    for _ in range(n_trials):
        # --- Vérité Terrain ---
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

        # -> VECTEUR VRAI (Point d'arrivée GT depuis le repère 1)
        x3_gt = x2_gt * np.cos(theta_gt) + D2_gt * np.sin(theta_gt)
        z3_gt = -x2_gt * np.sin(theta_gt) + D2_gt * np.cos(theta_gt)
        Tx_gt = x1_gt - x3_gt
        Tz_gt = D1_GT - z3_gt

        # --- Bruits stochastiques ---
        noise_i1 = np.random.normal(0, sigma_px) * PIXEL_SIZE
        noise_i2 = np.random.normal(0, sigma_px) * PIXEL_SIZE
        noise_D1 = (D1_GT**2 / (F * B)) * (2 * np.random.normal(0, sigma_px) * PIXEL_SIZE)
        noise_D2 = (D2_gt**2 / (F * B)) * (2 * np.random.normal(0, sigma_px) * PIXEL_SIZE)

        for scenario in SCENARIOS:
            i1, i2, D1, D2 = i1_m_gt, i2_m_gt, D1_GT, D2_gt
            
            if scenario == 'Erreurs_i' or scenario == 'Combine': 
                i1 += noise_i1; i2 += noise_i2
            if scenario == 'Erreurs_D' or scenario == 'Combine': 
                D1 += noise_D1; D2 += noise_D2

            # -> VECTEUR ESTIMÉ (FORMULE FACTORISÉE ASTEK - image_f20a98.png)
            denom_commun = (i1 * D1 + i2 * D2)**2 + (F**2) * (D2 - D1)**2
            
            if abs(denom_commun) < 1e-12:
                Tx_est = 0.0
                Tz_est = 0.0
            else:
                # Calcul du grand facteur commun K
                K = (D2**2 - D1**2) * (F**2) + (D2**2) * (i2**2) - (D1**2) * (i1**2)
                
                # Application de la formule pour (x3 - x1) et (z3 - z1)
                x3_minus_x1 = -(1.0 / F) * (D2 * i2 + D1 * i1) * (K / denom_commun)
                z3_minus_z1 = -(D2 - D1) * (K / denom_commun)
                
                # Notre vecteur de translation est l'inverse : (x1 - x3) et (z1 - z3)
                Tx_est = -x3_minus_x1
                Tz_est = -z3_minus_z1
            
            # -> ERREUR ABSOLUE DE POSITION
            erreur_position = np.sqrt((Tx_est - Tx_gt)**2 + (Tz_est - Tz_gt)**2)
            err_p[scenario].append(erreur_position)
            
    stats_p = {s: (np.median(err_p[s]), np.percentile(err_p[s], 25), np.percentile(err_p[s], 75)) for s in SCENARIOS}
    return stats_p

def run_study_2_quantization(delta_i_px, fixed_R):
    n_trials = 2000
    err_p = {s: [] for s in SCENARIOS}
    
    for _ in range(n_trials):
        # --- Vérité Terrain ---
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
        
        # -> VECTEUR VRAI
        x3_gt = x2_gt * np.cos(theta_gt) + D2_gt * np.sin(theta_gt)
        z3_gt = -x2_gt * np.sin(theta_gt) + D2_gt * np.cos(theta_gt)
        Tx_gt = x1_gt - x3_gt
        Tz_gt = D1_GT - z3_gt
        
        # --- Bruits de Discrétisation ---
        noise_i1 = np.random.uniform(-0.5, 0.5) * delta_i_px * PIXEL_SIZE
        noise_i2 = np.random.uniform(-0.5, 0.5) * delta_i_px * PIXEL_SIZE
        noise_D1 = (D1_GT**2 / (F * B)) * (2 * np.random.uniform(-0.5, 0.5) * delta_i_px * PIXEL_SIZE)
        noise_D2 = (D2_gt**2 / (F * B)) * (2 * np.random.uniform(-0.5, 0.5) * delta_i_px * PIXEL_SIZE)

        for scenario in SCENARIOS:
            i1, i2, D1, D2 = i1_m_gt, i2_m_gt, D1_GT, D2_gt
            
            if scenario == 'Erreurs_i' or scenario == 'Combine': 
                i1 += noise_i1; i2 += noise_i2
            if scenario == 'Erreurs_D' or scenario == 'Combine': 
                D1 += noise_D1; D2 += noise_D2

            # -> VECTEUR ESTIMÉ (FORMULE FACTORISÉE ASTEK - image_f20a98.png)
            denom_commun = (i1 * D1 + i2 * D2)**2 + (F**2) * (D2 - D1)**2
            
            if abs(denom_commun) < 1e-12:
                Tx_est = 0.0
                Tz_est = 0.0
            else:
                # Calcul du grand facteur commun K
                K = (D2**2 - D1**2) * (F**2) + (D2**2) * (i2**2) - (D1**2) * (i1**2)
                
                # Application de la formule pour (x3 - x1) et (z3 - z1)
                x3_minus_x1 = -(1.0 / F) * (D2 * i2 + D1 * i1) * (K / denom_commun)
                z3_minus_z1 = -(D2 - D1) * (K / denom_commun)
                
                # Notre vecteur de translation est l'inverse : (x1 - x3) et (z1 - z3)
                Tx_est = -x3_minus_x1
                Tz_est = -z3_minus_z1
            
            # -> ERREUR ABSOLUE DE POSITION
            erreur_position = np.sqrt((Tx_est - Tx_gt)**2 + (Tz_est - Tz_gt)**2)
            err_p[scenario].append(erreur_position)
                    
    stats_p = {s: (np.median(err_p[s]), np.percentile(err_p[s], 25), np.percentile(err_p[s], 75)) for s in SCENARIOS}
    return stats_p

# =====================================================================
# --- 3. GÉNÉRATION DES DONNÉES ---
# =====================================================================

print("Début des simulations (Modèle 2 : Formule Astek Factorisée)...")
data_all_R = {}

for current_R, r_label in zip(R_VALUES, R_LABELS):
    print(f"-> Calculs en cours pour {r_label}")
    res = {
        'p_gauss': {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS},
        'p_quant': {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS}
    }
    
    for p in NOISE_LEVELS:
        sp = run_study_1_gaussian(p, current_R)
        for s in SCENARIOS:
            res['p_gauss'][s]['med'].append(sp[s][0]); res['p_gauss'][s]['q25'].append(sp[s][1]); res['p_gauss'][s]['q75'].append(sp[s][2])

    for delta_i in ERREUR_TRONCATURE_PX:
        sp = run_study_2_quantization(delta_i, current_R)
        for s in SCENARIOS:
            res['p_quant'][s]['med'].append(sp[s][0]); res['p_quant'][s]['q25'].append(sp[s][1]); res['p_quant'][s]['q75'].append(sp[s][2])
            
    data_all_R[current_R] = res

# =====================================================================
# --- 4. AFFICHAGE ---
# =====================================================================

marge_legend = mpatches.Patch(color='grey', alpha=0.3, label='Marges d\'erreurs')

configs_figures = [
    ("1 - Erreur Modèle 2 (Bruit Gaussien)", 'p_gauss', NOISE_LEVELS, "Écart-type du bruit stochastique [Pixels]"),
    ("2 - Erreur Modèle 2 (Discrétisation)", 'p_quant', ERREUR_TRONCATURE_PX, "Erreur de discrétisation $\delta i$ [Pixels]")
]

for title, data_key, x_axis, x_label in configs_figures:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.canvas.manager.set_window_title(title)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for ax, current_R, r_label in zip(axes.flatten(), R_VALUES, R_LABELS):
        data = data_all_R[current_R][data_key]
        
        for s in SCENARIOS:
            y_med = np.array(data[s]['med'])
            y_q25 = np.array(data[s]['q25'])
            y_q75 = np.array(data[s]['q75'])
            
            linewidth = 2.5 if s == 'Erreurs_i' else 1.5
            ax.plot(x_axis, y_med, marker=STYLE[s]['marker'], color=STYLE[s]['color'], label=STYLE[s]['label'], linewidth=linewidth)
            ax.fill_between(x_axis, y_q25, y_q75, color=STYLE[s]['color'], alpha=0.15)
            
        ax.set_title(r_label, fontsize=12, fontweight='bold')
        ax.set_xlabel(x_label)
        ax.set_ylabel("Erreur de position absolue [m]")
        
        if "Discrétisation" in title:
            ax.set_xlim(0, 1.0)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        elif "Gaussien" in title:
            ax.set_xlim(0, 0.7)
            
        ax.grid(True, alpha=0.3)
        
        def clean_format(x, pos):
            if x == 0: return '0'
            return f"{x:.2f}".rstrip('0').rstrip('.')
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(clean_format))
        
        handles, labels = ax.get_legend_handles_labels()
        if 'Marges d\'erreurs' not in labels:
            handles.append(marge_legend)
            labels.append(marge_legend.get_label())
        ax.legend(handles=handles, labels=labels, fontsize='small', loc='upper left')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()