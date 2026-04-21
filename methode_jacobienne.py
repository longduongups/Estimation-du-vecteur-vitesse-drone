import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =====================================================================
# --- 1. CONFIGURATION GÉNÉRALE (Paramètres Physiques) ---
# =====================================================================
V_REAL = 2.78        # Vitesse réelle de référence (m/s)
DT = 0.02            # Pas de temps (s)
F = 0.035            # Focale de la caméra (m)
B = 0.20             # Baseline stéréoscopique (m)
PIXEL_SIZE = 5e-6    # Taille d'un pixel sur le capteur (m)
D1_GT = 10.0         # Distance initiale à la façade (m)

# Nombre de points d'intérêts suivis (pour le multi-équations)
N_POINTS = 500        

R_VALUES = [10, 50, 500, np.inf] 
R_LABELS = ["Virage Serré (R=10m)", "Virage Fluide (R=50m)", "Quasi-Ligne Droite (R=500m)", "Ligne Droite Pure (R \u2192 \u221E)"]

SCENARIOS = ['Erreurs_i', 'Erreurs_D', 'Combine']
STYLE = {
    'Erreurs_i': {'color': 'blue',   'marker': 'o', 'label': 'Erreurs sur $i$ uniquement'},
    'Erreurs_D': {'color': 'orange', 'marker': 's', 'label': 'Erreurs sur $D$ uniquement'},
    'Combine':   {'color': 'black',  'marker': 'X', 'label': 'Erreurs sur toutes les grandeurs'}
}

NOISE_LEVELS = np.linspace(0.0, 1.0, 10)
FRACTIONS_PIXEL = np.linspace(0.0, 0.99, 20)

# =====================================================================
# --- 2. FONCTIONS DE SIMULATION (MÉTHODE JACOBIENNE) ---
# =====================================================================

def run_study_1_gaussian(sigma_px, fixed_R):
    n_trials = 2000
    err_v = {s: [] for s in SCENARIOS}
    err_theta = {s: [] for s in SCENARIOS}
    
    for _ in range(n_trials):
        # 1. Génération de N points répartis sur le capteur
        i1_px_gt = np.random.uniform(-400, 400, N_POINTS)
        i1_m_gt = i1_px_gt * PIXEL_SIZE
        D1_gt = np.full(N_POINTS, D1_GT) 

        # 2. Cinématique vraie du drone
        V_z_gt = V_REAL
        V_x_gt = 0.0
        omega_y_gt = 0.0 if np.isinf(fixed_R) else V_REAL / fixed_R
        theta_gt = omega_y_gt * DT

        # 3. Vitesse apparente exacte des pixels (Flux Optique Continu)
        i_dot_gt = (F / D1_gt) * V_x_gt - (i1_m_gt / D1_gt) * V_z_gt + (F + (i1_m_gt**2)/F) * omega_y_gt

        # Bruits gaussiens
        noise_i1 = np.random.normal(0, sigma_px, N_POINTS) * PIXEL_SIZE
        noise_i2 = np.random.normal(0, sigma_px, N_POINTS) * PIXEL_SIZE
        noise_D = (D1_GT**2 / (F * B)) * (2 * np.random.normal(0, sigma_px, N_POINTS) * PIXEL_SIZE)

        for scenario in SCENARIOS:
            i1_meas = i1_m_gt.copy()
            D_meas = D1_gt.copy()
            delta_i_gt = i_dot_gt * DT

            if scenario == 'Erreurs_i' or scenario == 'Combine':
                i1_meas += noise_i1
                # Le delta_i mesuré subit le bruit de la frame 1 et de la frame 2
                delta_i_meas = (i1_m_gt + delta_i_gt + noise_i2) - (i1_m_gt + noise_i1)
            else:
                delta_i_meas = delta_i_gt

            if scenario == 'Erreurs_D' or scenario == 'Combine':
                D_meas += noise_D

            # Vitesse apparente mesurée
            i_dot_meas = delta_i_meas / DT

            #  MATRICE JACOBIENNE (L)
            L = np.zeros((N_POINTS, 3))
            L[:, 0] = F / D_meas
            L[:, 1] = -i1_meas / D_meas
            L[:, 2] = F + (i1_meas**2) / F

            # Résolution par Pseudo-Inverse (Moindres Carrés)
            try:
                V_est_vec = np.linalg.pinv(L) @ i_dot_meas
                # V_est_vec contient [V_x, V_z, omega_y]
                v_est = np.sqrt(V_est_vec[0]**2 + V_est_vec[1]**2) # Vitesse globale
                theta_est = V_est_vec[2] * DT                    # Intégration de l'angle
            except:
                v_est, theta_est = 0.0, 0.0
            # -----------------------------------------------

            err_v[scenario].append(abs(v_est - V_REAL))
            err_theta[scenario].append(abs(np.degrees(theta_est - theta_gt))) 
            
    stats_v = {s: (np.median(err_v[s]), np.percentile(err_v[s], 25), np.percentile(err_v[s], 75)) for s in SCENARIOS}
    stats_theta = {s: (np.median(err_theta[s]), np.percentile(err_theta[s], 25), np.percentile(err_theta[s], 75)) for s in SCENARIOS}
    return stats_v, stats_theta

def run_study_2_quantization(fraction_px, fixed_R):
    n_trials = 2000
    err_v = {s: [] for s in SCENARIOS}
    err_theta = {s: [] for s in SCENARIOS}
    
    for _ in range(n_trials):
        base_pixels = np.random.randint(-400, 400, N_POINTS)
        i1_px_gt = base_pixels + fraction_px 
        i1_m_gt = i1_px_gt * PIXEL_SIZE
        D1_gt = np.full(N_POINTS, D1_GT)

        V_z_gt = V_REAL
        V_x_gt = 0.0
        omega_y_gt = 0.0 if np.isinf(fixed_R) else V_REAL / fixed_R
        theta_gt = omega_y_gt * DT

        i_dot_gt = (F / D1_gt) * V_x_gt - (i1_m_gt / D1_gt) * V_z_gt + (F + (i1_m_gt**2)/F) * omega_y_gt
        delta_i_m_gt = i_dot_gt * DT
        i2_m_gt = i1_m_gt + delta_i_m_gt
        i2_px_gt = i2_m_gt / PIXEL_SIZE

        # Erreurs de discrétisation matricielles
        di1 = np.abs(i1_px_gt) - np.floor(np.abs(i1_px_gt))
        di2 = np.abs(i2_px_gt) - np.floor(np.abs(i2_px_gt))
        sign1 = np.where(i1_px_gt != 0, np.sign(i1_px_gt), 1)
        sign2 = np.where(i2_px_gt != 0, np.sign(i2_px_gt), 1)

        err_i1 = sign1 * di1 * PIXEL_SIZE
        err_i2 = sign2 * di2 * PIXEL_SIZE
        err_D = (D1_GT**2 / (F * B)) * (2 * di1 * PIXEL_SIZE)

        for scenario in SCENARIOS:
            i1_meas = i1_m_gt.copy()
            D_meas = D1_gt.copy()
            i2_meas = i2_m_gt.copy()

            if scenario == 'Erreurs_i' or scenario == 'Combine':
                i1_meas -= err_i1
                i2_meas -= err_i2
            if scenario == 'Erreurs_D' or scenario == 'Combine':
                D_meas += err_D

            delta_i_meas = i2_meas - i1_meas
            i_dot_meas = delta_i_meas / DT

            #  MATRICE JACOBIENNE (L)
            L = np.zeros((N_POINTS, 3))
            L[:, 0] = F / D_meas
            L[:, 1] = -i1_meas / D_meas
            L[:, 2] = F + (i1_meas**2) / F

            try:
                V_est_vec = np.linalg.pinv(L) @ i_dot_meas
                v_est = np.sqrt(V_est_vec[0]**2 + V_est_vec[1]**2)
                theta_est = V_est_vec[2] * DT
            except:
                v_est, theta_est = 0.0, 0.0
            # -----------------------------------------------
            
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
    print(f"Calculs en cours pour {R_LABELS[idx_r]} (Jacobienne sur {N_POINTS} points)...")
    
    data_v_gauss = {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS}
    data_t_gauss = {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS}
    for p in NOISE_LEVELS:
        sv, st = run_study_1_gaussian(p, current_R)
        for s in SCENARIOS:
            data_v_gauss[s]['med'].append(sv[s][0]); data_v_gauss[s]['q25'].append(sv[s][1]); data_v_gauss[s]['q75'].append(sv[s][2])
            data_t_gauss[s]['med'].append(st[s][0]); data_t_gauss[s]['q25'].append(st[s][1]); data_t_gauss[s]['q75'].append(st[s][2])

    data_v_quant = {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS}
    data_t_quant = {s: {'med':[], 'q25':[], 'q75':[]} for s in SCENARIOS}
    for frac in FRACTIONS_PIXEL:
        sv, st = run_study_2_quantization(frac, current_R)
        for s in SCENARIOS:
            data_v_quant[s]['med'].append(sv[s][0]); data_v_quant[s]['q25'].append(sv[s][1]); data_v_quant[s]['q75'].append(sv[s][2])
            data_t_quant[s]['med'].append(st[s][0]); data_t_quant[s]['q25'].append(st[s][1]); data_t_quant[s]['q75'].append(st[s][2])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.canvas.manager.set_window_title(f"Simulation Jacobienne - {R_LABELS[idx_r]}")
    fig.suptitle(f"Erreur d'estimation (Matrice Jacobienne, N={N_POINTS}) : {R_LABELS[idx_r]}", fontsize=18, fontweight='bold')

    for s in SCENARIOS:
        # Courbes
        axes[0,0].plot(NOISE_LEVELS, data_v_gauss[s]['med'], marker=STYLE[s]['marker'], color=STYLE[s]['color'], label=STYLE[s]['label'])
        axes[0,1].plot(NOISE_LEVELS, data_t_gauss[s]['med'], marker=STYLE[s]['marker'], color=STYLE[s]['color'], label=STYLE[s]['label'])
        axes[1,0].plot(FRACTIONS_PIXEL, data_v_quant[s]['med'], marker=STYLE[s]['marker'], color=STYLE[s]['color'], label=STYLE[s]['label'])
        axes[1,1].plot(FRACTIONS_PIXEL, data_t_quant[s]['med'], marker=STYLE[s]['marker'], color=STYLE[s]['color'], label=STYLE[s]['label'])
        
        # Marges
        axes[0,0].fill_between(NOISE_LEVELS, data_v_gauss[s]['q25'], data_v_gauss[s]['q75'], color=STYLE[s]['color'], alpha=0.15)
        axes[0,1].fill_between(NOISE_LEVELS, data_t_gauss[s]['q25'], data_t_gauss[s]['q75'], color=STYLE[s]['color'], alpha=0.15)
        axes[1,0].fill_between(FRACTIONS_PIXEL, data_v_quant[s]['q25'], data_v_quant[s]['q75'], color=STYLE[s]['color'], alpha=0.15)
        axes[1,1].fill_between(FRACTIONS_PIXEL, data_t_quant[s]['q25'], data_t_quant[s]['q75'], color=STYLE[s]['color'], alpha=0.15)

    axes[0,0].set_title("Vitesse (Bruit Gaussien)")
    axes[0,0].set_ylabel("Erreur absolue vitesse [m/s]")
    axes[0,0].set_xlabel("Écart-type du bruit [Pixels]") 

    axes[0,1].set_title(f"Orientation \u03B8 (Bruit Gaussien)")
    axes[0,1].set_ylabel(f"Erreur absolue orientation [°]")
    axes[0,1].set_xlabel("Écart-type du bruit [Pixels]")
    
    axes[1,0].set_title("Vitesse (Discrétisation)")
    axes[1,0].set_ylabel("Erreur absolue vitesse [m/s]")
    axes[1,0].set_xlabel("Décalage fractionnaire [Fraction de Pixel]")

    axes[1,1].set_title(f"Orientation \u03B8 (Discrétisation)")
    axes[1,1].set_ylabel(f"Erreur absolue orientation [°]")
    axes[1,1].set_xlabel("Décalage fractionnaire [Fraction de Pixel]") 

    for ax in axes.flatten():
        ax.axhline(0, color='red', linestyle='--', alpha=0.6)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if 'Marges d\'erreurs' not in labels:
            handles.append(marge_legend)
            labels.append(marge_legend.get_label())
        ax.legend(handles=handles, labels=labels, fontsize='small')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()