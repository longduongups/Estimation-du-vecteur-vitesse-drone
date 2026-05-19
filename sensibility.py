
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)  # Reproductibilité

# =====================================================================
# 1. CONFIGURATION DES PARAMÈTRES GLOBAUX
# =====================================================================
F = 0.010              # Focale par défaut (10 mm)
B = 0.20               # Baseline stéréoscopique (m)
PIXEL_SIZE = 5e-6      # Taille physique d'un pixel (5 µm)
D1_GT = 10.0           # Distance initiale vérité terrain (m)
DT = 0.02              # Pas de temps (s)
V_DRONE = 2.78         # Vitesse drone (m/s)
D_DRONE = V_DRONE * DT # Déplacement réel par pas de temps (0.0556 m)

I_FIXE = 0.005         # Position pixellique de référence (5 mm = θ_FOV ≈ 26.57°)
THETA_FOV_REF = np.degrees(np.arctan(I_FIXE / F))  # ≈ 26.57°

N_TRIALS = 2000        # Nombre de tirages Monte-Carlo

# Bruit max de profondeur pour 1 pixel à D = 10 m, f = 10 mm
BRUIT_MAX_M_DEFAUT = (D1_GT**2 / (F * B)) * (1.0 * PIXEL_SIZE)  

# =====================================================================
# 2. FONCTIONS DES DEUX MODÈLES
# =====================================================================
def calc_modele_1(i1, i2, D1, D2, f_cam=F):
    """Modèle cinématique : estime R, θ, v puis déplacement Tx, Tz."""
    if np.all(D1 == D2):
        scalar_input = np.isscalar(D1)
        if scalar_input:
            return -D_DRONE, 0.0
        return -np.full_like(np.asarray(D1, dtype=float), D_DRONE), \
               np.zeros_like(np.asarray(D1, dtype=float))

    R = (i2**2 * D2**2 - i1**2 * D1**2 + (f_cam**2) * (D2**2 - D1**2)) \
        / (2 * f_cam**2 * (D2 - D1))
    denom_theta = (i1 * D1 + i2 * D2)**2 + (f_cam**2) * (D2 - D1)**2
    sin_theta = (2 * f_cam * (D2 - D1) * (i2 * D2 + i1 * D1)) / denom_theta
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    theta = np.arcsin(sin_theta)
    d = R * theta
    Tx_est = d * (np.sin(theta) / theta)
    Tz_est = d * ((1 - np.cos(theta)) / theta)
    return -Tx_est, -Tz_est

def calc_modele_2(i1, i2, D1, D2, f_cam=F):
    """Modèle géométrique direct : retourne (Tx, Tz) du mouvement de la scène.
       Pas de singularité D1 ≈ D2."""
    denom = (i1 * D1 + i2 * D2)**2 + (f_cam**2) * (D2 - D1)**2
    K = (D2**2 - D1**2) * (f_cam**2) + (D2**2) * (i2**2) - (D1**2) * (i1**2)
    Tx_est = -(1.0 / f_cam) * (D2 * i2 + D1 * i1) * (K / denom)
    Tz_est = -(D2 - D1) * (K / denom)
    return Tx_est, Tz_est

# =====================================================================
# 3. GÉNÉRATION DE LA VÉRITÉ TERRAIN PAR SCÉNARIO DE TRAJECTOIRE
# =====================================================================
def verite_terrain(scenario, i1=I_FIXE, D1=D1_GT, f=F, d=D_DRONE):
    """Génère la vérité terrain pour un scénario donné."""
    X_M = -i1 * D1 / f  

    if scenario == 'rect_0':  
        TX_drone, TZ_drone = d, 0.0
        x_M_C2 = X_M - TX_drone
        z_M_C2 = D1 - TZ_drone
        D2 = z_M_C2
        i2 = -x_M_C2 * f / D2
        return i2, D2, -TX_drone, -TZ_drone

    elif scenario in ('rect_5', 'rect_30'):
            TX_local = d
            TZ_local = 0
            
            x_M_C2 = X_M - TX_local
            D2 = D1 
            i2 = -x_M_C2 * f / D2
            return i2, D2, -TX_local, -TZ_local

    elif scenario.startswith('R'):
        R = float(scenario[1:])
        theta = d / R
        TX_drone = R * np.sin(theta)
        TZ_drone = R * (1 - np.cos(theta))
        dx = X_M - TX_drone
        dz = D1 - TZ_drone
        x_M_C2 = np.cos(theta) * dx + np.sin(theta) * dz
        z_M_C2 = -np.sin(theta) * dx + np.cos(theta) * dz
        D2 = z_M_C2
        i2 = -x_M_C2 * f / D2
        return i2, D2, -TX_drone, -TZ_drone

    raise ValueError(f"Scénario inconnu : {scenario}")

# =====================================================================
# FIGURE 3 : SENSIBILITÉ AU BRUIT 
# =====================================================================
print("\n=== Génération Figure 3 : Sensibilité au bruit (Modèle 2) ===")

echelles_bruit = np.linspace(0.0, 1.0, 25)
i2_ref = -((-I_FIXE * D1_GT / F) - D_DRONE) * F / D1_GT  # rectiligne colinéaire

m2_med, m2_q25, m2_q75, m2_max = [], [], [], []

for scale in echelles_bruit:
    bruit_px_1 = np.random.uniform(-0.5*scale, 0.5*scale, N_TRIALS)
    bruit_px_2 = np.random.uniform(-0.5*scale, 0.5*scale, N_TRIALS)
    #pire cas asymétrique 
    bruit_px_1[0], bruit_px_2[0] = -0.5*scale, 0.5*scale

    D1_sim = D1_GT + (D1_GT**2 / (F * B)) * (bruit_px_1 * PIXEL_SIZE)
    D2_sim = D1_GT + (D1_GT**2 / (F * B)) * (bruit_px_2 * PIXEL_SIZE)

    tx2, tz2 = calc_modele_2(I_FIXE, i2_ref, D1_sim, D2_sim)
    err2 = np.sqrt((tx2 - (-D_DRONE))**2 + (tz2 - 0)**2)
    m2_med.append(np.nanmedian(err2));  m2_q25.append(np.nanpercentile(err2, 25))
    m2_q75.append(np.nanpercentile(err2, 75));  m2_max.append(np.nanmax(err2))

fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(echelles_bruit, m2_q25, m2_q75, color='#ff7f0e', alpha=0.25,
                label='Intervalle interquartile (50 % central)')
ax.plot(echelles_bruit, m2_med, color='#ff7f0e', lw=3.5,
        label='Médiane M2 (2000 tirages)')
ax.plot(echelles_bruit, m2_max, color='#d62728', lw=3.5,
        label='Pire cas absolu M2')

ax.set_xlabel(r"Amplitude maximale d'erreur stéréoscopique $\delta d_{max}$ [pixels]")
ax.set_ylabel(r"Erreur de position $\varepsilon_{pos}$ [m]")
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 0.7)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax.annotate('~ 0,62 m', xy=(1.0, m2_max[-1]),
            xytext=(0.72, 0.48),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.tight_layout()
plt.savefig('figure_3_bruit_M2.png', dpi=150, bbox_inches='tight')
plt.show()

# =====================================================================
# FIGURE 4 : SENSIBILITÉ AUX SCÉNARIOS DE TRAJECTOIRE
# =====================================================================
print("\n=== Génération Figure 4 : Sensibilité aux scénarios de trajectoire ===")

SCENARIOS = ['rect_0', 'rect_5', 'rect_30', 'R500', 'R50', 'R10']
SCENARIO_INFO = {
    'rect_0':  ('Rectiligne colinéaire (α = 0°)',  '#d62728'),
    'rect_5':  ('Rectiligne oblique (α = 5°)',      '#ff7f0e'),
    'rect_30': ('Rectiligne oblique (α = 30°)',     '#2ca02c'),
    'R500':    ('Courbe R = 500 m',                 '#1f77b4'),
    'R50':     ('Courbe R = 50 m',                  '#9467bd'),
    'R10':     ('Courbe R = 10 m',                  '#8c564b'),
}

medianes_scenarios = {s: [] for s in SCENARIOS}

for scale in echelles_bruit:
    for scenario in SCENARIOS:
        i2_gt, D2_gt, Tx_vrai, Tz_vrai = verite_terrain(scenario)

        bruit_px_1 = np.random.uniform(-0.5*scale, 0.5*scale, N_TRIALS)
        bruit_px_2 = np.random.uniform(-0.5*scale, 0.5*scale, N_TRIALS)

        D1_sim = D1_GT + (D1_GT**2 / (F * B)) * (bruit_px_1 * PIXEL_SIZE)
        D2_sim = D2_gt + (D2_gt**2 / (F * B)) * (bruit_px_2 * PIXEL_SIZE)

        tx, tz = calc_modele_2(I_FIXE, i2_gt, D1_sim, D2_sim)
        err = np.sqrt((tx - Tx_vrai)**2 + (tz - Tz_vrai)**2)
        medianes_scenarios[scenario].append(np.nanmedian(err))

fig, ax = plt.subplots(figsize=(10, 6))
for scenario in SCENARIOS:
    label, color = SCENARIO_INFO[scenario]
    ax.plot(echelles_bruit, medianes_scenarios[scenario],
            color=color, lw=2.5, label=label)


ax.set_xlabel(r"Amplitude maximale d'erreur stéréoscopique $\delta d_{max}$ [pixels]")
ax.set_ylabel(r"Erreur de position médiane $\varepsilon_{pos}$ [m]")
ax.set_xlim(0, 1.0)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.savefig('figure_4_scenarios_trajectoire.png', dpi=150, bbox_inches='tight')
plt.show()

# =====================================================================
# Calcul pire cas pour figures 5-7 (cas rectiligne, M2 uniquement)
# =====================================================================
def pire_cas_M2(i1, i2, D1_gt, D2_gt, bruit_max_d, Tx_vrai, Tz_vrai):
    D1_sim = D1_gt - bruit_max_d / 2.0
    D2_sim = D2_gt + bruit_max_d / 2.0
    tx, tz = calc_modele_2(i1, i2, D1_sim, D2_sim)
    return np.sqrt((tx - Tx_vrai)**2 + (tz - Tz_vrai)**2)

# =====================================================================
# FIGURE 5 : SENSIBILITÉ À L'ANGLE D'OBSERVATION θ_FOV
# =====================================================================
print("\n=== Génération Figure 5 : Sensibilité à θ_FOV ===")

i_values = np.linspace(0.0005, 0.0096, 100)
theta_fov_deg = np.degrees(np.arctan(i_values / F))

err_theta = []
for i1_var in i_values:
    X_M = -i1_var * D1_GT / F
    X_M_C2 = X_M - D_DRONE
    i2_var = -X_M_C2 * F / D1_GT
    
    err = pire_cas_M2(i1_var, i2_var, D1_GT, D1_GT, BRUIT_MAX_M_DEFAUT,
                      Tx_vrai=-D_DRONE, Tz_vrai=0)
    err_theta.append(err)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta_fov_deg, err_theta, color='#d62728', lw=3.5,
        label='Pire cas absolu M2 (±0,5 px par image)')


ax.set_xlabel(r"Angle d'observation $\theta_{FOV}$ par rapport à l'axe optique [°]")
ax.set_ylabel(r"Erreur de position $\varepsilon_{pos}$ (pire cas) [m]")
ax.set_xlim(0, 45)
ax.set_ylim(0, 5.0)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper right', fontsize=10)

ax.scatter([5.71, 26.57], [2.44, 0.62], color='black', zorder=5)
ax.annotate(r'~ 2,44 m ($\theta_{FOV}$ = 5,71°)',
            xy=(5.71, 2.44), xytext=(10, 3.2),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate(r'~ 0,62 m ($\theta_{FOV}$ = 26,57°)',
            xy=(26.57, 0.62), xytext=(30, 1.5),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.tight_layout()
plt.savefig('figure_5_sensibilite_theta_fov.png', dpi=150, bbox_inches='tight')
plt.show()

# =====================================================================
# FIGURE 6 : SENSIBILITÉ À LA DISTANCE FOCALE f 
# =====================================================================
print("\n=== Génération Figure 6 : Sensibilité à la focale ===")

f_values = np.linspace(0.005, 0.050, 100)
err_focale = []

I_MAX_CAPTEUR = 0.0096  
F_LIMIT = I_MAX_CAPTEUR / np.tan(np.radians(THETA_FOV_REF)) # en mètres
F_LIMIT_MM = F_LIMIT * 1000 

for f_cam in f_values:
    bruit_d_actuel = (D1_GT**2 / (f_cam * B)) * (1.0 * PIXEL_SIZE)
    i1_var = f_cam * np.tan(np.radians(THETA_FOV_REF))
    X_M = -i1_var * D1_GT / f_cam
    X_M_C2 = X_M - D_DRONE
    i2_var = -X_M_C2 * f_cam / D1_GT

    D1_sim = D1_GT - bruit_d_actuel / 2.0
    D2_sim = D1_GT + bruit_d_actuel / 2.0
    tx, tz = calc_modele_2(i1_var, i2_var, D1_sim, D2_sim, f_cam=f_cam)
    err_focale.append(np.sqrt((tx - (-D_DRONE))**2 + (tz - 0)**2))

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(f_values * 1000, err_focale, color='#d62728', lw=3.5, zorder=3,
        label='Pire cas absolu M2 (±0,5 px par image)')
ax.axvspan(F_LIMIT_MM, 50, color='gray', alpha=0.2, hatch='//', edgecolor='none', zorder=1,
           label='Zone hors-champ (Point > Bord du capteur 4K)')
ax.axvline(x=F_LIMIT_MM, color='black', linestyle='--', lw=2, zorder=4)

ax.set_xlabel(r"Distance focale de la caméra $f$ [mm]")
ax.set_ylabel(r"Erreur de position $\varepsilon_{pos}$ (pire cas) [m]")
ax.set_xlim(5, 50)
ax.set_ylim(0, 1.5)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper right', fontsize=10)

# Annotations existantes et nouvelles
ax.scatter([10.0], [0.62], color='black', zorder=5)
ax.annotate('~ 0,62 m (f = 10 mm)',
            xy=(10.0, 0.62), xytext=(6.0, 1.0),
            arrowprops=dict(facecolor='black', arrowstyle='->'))

ax.annotate(f'Limite\n(f = {F_LIMIT_MM:.1f} mm)',
            xy=(F_LIMIT_MM, 1.3), xytext=(F_LIMIT_MM + 2, 1.35),
            fontweight='bold', color='black',
            arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.tight_layout()
plt.savefig('figure_6_sensibilite_focale.png', dpi=150, bbox_inches='tight')
plt.show()

# =====================================================================
# FIGURE 7 : SENSIBILITÉ À LA DISTANCE AU MUR D
# =====================================================================
print("\n=== Génération Figure 7 : Sensibilité à la distance D ===")

D_values = np.linspace(2.0, 15.0, 100)
err_D = []

for D in D_values:
    bruit_d_actuel = (D**2 / (F * B)) * (1.0 * PIXEL_SIZE)
    X_M = -I_FIXE * D1_GT / F  
    i1_var = -X_M * F / D
    X_M_C2 = X_M - D_DRONE
    i2_var = -X_M_C2 * F / D

    D1_sim = D - bruit_d_actuel / 2.0
    D2_sim = D + bruit_d_actuel / 2.0
    tx, tz = calc_modele_2(i1_var, i2_var, D1_sim, D2_sim)
    err_D.append(np.sqrt((tx - (-D_DRONE))**2 + (tz - 0)**2))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(D_values, err_D, color='#d62728', lw=3.5,
        label='Pire cas absolu M2 (±0,5 px par image)')


ax.set_xlabel(r"Distance au mur $D$ [m]")
ax.set_ylabel(r"Erreur de position $\varepsilon_{pos}$ (pire cas) [m]")
ax.set_xlim(2, 15)
ax.set_ylim(0, max(err_D) * 1.1)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left', fontsize=10)
ax.scatter([10.0], [err_D[np.argmin(np.abs(D_values - 10))]], color='black', zorder=5)
ax.annotate(f"~ 0,62 m (D = 10 m)",
            xy=(10.0, err_D[np.argmin(np.abs(D_values - 10))]),
            xytext=(5, 1.0),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.tight_layout()
plt.savefig('figure_7_sensibilite_distance.png', dpi=150, bbox_inches='tight')
plt.show()

# =====================================================================
# FIGURE 8 : IMPACT DU NOMBRE N DE CAMÉRAS 
# =====================================================================
print("\n=== Génération Figure 8 : Impact du nombre N de caméras ===")

# --- Hypothèse physique ---
B_TOT = 0.20   # Longueur totale 
N_values = np.arange(2, 17)

# Gain de précision en exploitant TOUTES les paires de caméras possibles.
G_N = np.sqrt(N_values**2 * (N_values + 1) / (12 * (N_values - 1)))

ERREUR_REF_N2 = m2_max[-1]     

err_N = ERREUR_REF_N2 / G_N

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(N_values, err_N, color='#1f77b4', lw=3.5, marker='o', markersize=8,
        label="Erreur projetée (exploitation de toutes les paires de caméras)")


ax.set_xlabel(r"Nombre $N$ de caméras alignées")
ax.set_ylabel(r"Erreur de position $\varepsilon_{pos}$ projetée [m]")
ax.set_xticks(N_values)
ax.set_xlim(2, 16)
ax.set_ylim(0, ERREUR_REF_N2 * 1.1)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper right', fontsize=10)

# Annotations des points clés
ax.scatter([2, 4, 10], [err_N[0], err_N[2], err_N[8]], color='black', zorder=5)
ax.annotate(f"~ {err_N[0]:.2f} m (N = 2)",
            xy=(2, err_N[0]), xytext=(3.0, err_N[0] * 0.80),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate(f"~ {err_N[2]:.2f} m (N = 4)",
            xy=(4, err_N[2]), xytext=(5.5, err_N[2] + 0.10),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate(f"~ {err_N[8]:.2f} m (N = 10)",
            xy=(10, err_N[8]), xytext=(11.5, err_N[8] + 0.10),
            arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.tight_layout()
plt.savefig('figure_8_impact_N_cameras.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  Gain G(3)  = {G_N[1]:.2f}  -> erreur ≈ {err_N[1]:.3f} m")
print(f"  Gain G(4)  = {G_N[2]:.2f}  -> erreur ≈ {err_N[2]:.3f} m")
print(f"  Gain G(10) = {G_N[8]:.2f}  -> erreur ≈ {err_N[8]:.3f} m")

print("\n=== Génération terminée ===")