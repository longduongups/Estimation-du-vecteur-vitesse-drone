"""
Simulation MERAK 551 — Modèle de bruit final, simple et cohérent :
À chaque image acquise (t1, t2), une erreur pixellique unique δk ∈ [-0.5, +0.5] px
qui se propage SIMULTANÉMENT à la position image i_k ET à la profondeur D_k
(via la loi de triangulation différentielle δD = D²·δd/(f·b)).

Pire cas : δ1 = -0.5, δ2 = +0.5 → erreur totale de disparité = 1 px.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import product
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

# =====================================================================
# 1. PARAMÈTRES
# =====================================================================
F = 0.010; B = 0.20; PIXEL_SIZE = 5e-6
D1_GT = 10.0; DT = 0.02; V_DRONE = 2.78
D_DRONE = V_DRONE * DT
I_FIXE = 0.005
THETA_FOV_REF = np.degrees(np.arctan(I_FIXE / F))
N_TRIALS = 2000

# =====================================================================
# 2. MODÈLE 2
# =====================================================================
def calc_modele_2(i1, i2, D1, D2, f=F):
    denom = (i1*D1 + i2*D2)**2 + f**2 * (D2-D1)**2
    K = (D2**2 - D1**2)*f**2 + D2**2*i2**2 - D1**2*i1**2
    Tx = -(1.0/f) * (D2*i2 + D1*i1) * K / denom
    Tz = -(D2 - D1) * K / denom
    return Tx, Tz

# =====================================================================
# 3. INJECTION DE BRUIT (modèle final : δk affecte i_k ET D_k)
# =====================================================================
def inputs(i1g, i2g, D1g, D2g, d1_px, d2_px, *, mode='both', f=F, b=B):
    """
    d1_px, d2_px = erreurs pixelliques sur image 1 et image 2 (en pixels).
    mode='both' : chacune affecte à la fois i et D (cas physique)
    mode='i'    : affecte uniquement i (cas A)
    mode='D'    : affecte uniquement D (cas B)
    """
    if mode in ('both', 'i'):
        i1n = i1g + d1_px * PIXEL_SIZE
        i2n = i2g + d2_px * PIXEL_SIZE
    else:
        i1n, i2n = i1g, i2g
    if mode in ('both', 'D'):
        D1n = D1g + (D1g**2/(f*b)) * d1_px * PIXEL_SIZE
        D2n = D2g + (D2g**2/(f*b)) * d2_px * PIXEL_SIZE
    else:
        D1n, D2n = D1g, D2g
    return i1n, i2n, D1n, D2n

def err_pos(tx, tz, txv, tzv):
    return np.sqrt((tx-txv)**2 + (tz-tzv)**2)

def monte_carlo(i1g, i2g, D1g, D2g, txv, tzv, sc=1.0, *, mode='both',
                n=N_TRIALS, f=F, b=B):
    d1 = np.random.uniform(-0.5*sc, 0.5*sc, n)
    d2 = np.random.uniform(-0.5*sc, 0.5*sc, n)
    i1n,i2n,D1n,D2n = inputs(i1g,i2g,D1g,D2g, d1, d2, mode=mode, f=f, b=b)
    tx, tz = calc_modele_2(i1n, i2n, D1n, D2n, f=f)
    return err_pos(tx, tz, txv, tzv)

def worst_case(i1g, i2g, D1g, D2g, txv, tzv, sc=1.0, *, mode='both', f=F, b=B):
    """4 combinaisons de signes (±0.5·sc, ±0.5·sc)."""
    s=0.5*sc; max_e=0
    for a, b_ in product([-s,+s], repeat=2):
        i1n,i2n,D1n,D2n = inputs(i1g,i2g,D1g,D2g, a, b_, mode=mode, f=f, b=b)
        tx, tz = calc_modele_2(i1n,i2n,D1n,D2n, f=f)
        e = np.sqrt((tx-txv)**2 + (tz-tzv)**2)
        if e>max_e: max_e=e
    return max_e

# =====================================================================
# 4. VÉRITÉ TERRAIN PAR SCÉNARIO
# =====================================================================
def verite_terrain(scenario, i1=I_FIXE, D1=D1_GT, f=F, d=D_DRONE):
    X_M = -i1 * D1 / f
    if scenario == 'rect_0':
        TX, TZ = d, 0.0
        x_M_C2 = X_M - TX;  D2 = D1 - TZ
        i2 = -x_M_C2 * f / D2
        return i2, D2, -TX, -TZ
    elif scenario in ('rect_5', 'rect_30'):
        TX = d;  TZ = 0
        x_M_C2 = X_M - TX;  D2 = D1
        i2 = -x_M_C2 * f / D2
        return i2, D2, -TX, -TZ
    elif scenario.startswith('R'):
        R = float(scenario[1:]);  theta = d / R
        TX = R*np.sin(theta);  TZ = R*(1 - np.cos(theta))
        dx = X_M - TX;  dz = D1 - TZ
        x_M_C2 =  np.cos(theta)*dx + np.sin(theta)*dz
        z_M_C2 = -np.sin(theta)*dx + np.cos(theta)*dz
        D2 = z_M_C2
        i2 = -x_M_C2 * f / D2
        return i2, D2, -TX, -TZ
    raise ValueError(scenario)

# Référence rectiligne colinéaire
_X1 = -I_FIXE * D1_GT / F
_X2 = _X1 - D_DRONE
I2_REF = -_X2 * F / D1_GT
TX_REF, TZ_REF = -D_DRONE, 0.0

# =====================================================================
# FIGURE 3 — sensibilité bruit (Scindée en 2 images distinctes)
# =====================================================================
print("\n=== Figure 3 ===")
echelles = np.linspace(0.0, 1.0, 25)
resultats = {m: {'med':[], 'q25':[], 'q75':[], 'wc':[]} for m in ('i','D','both')}

for sc in echelles:
    for m in ('i','D','both'):
        e = monte_carlo(I_FIXE, I2_REF, D1_GT, D1_GT, TX_REF, TZ_REF, sc, mode=m)
        resultats[m]['med'].append(np.nanmedian(e))
        resultats[m]['q25'].append(np.nanpercentile(e, 25))
        resultats[m]['q75'].append(np.nanpercentile(e, 75))
        resultats[m]['wc' ].append(worst_case(I_FIXE, I2_REF, D1_GT, D1_GT,
                                              TX_REF, TZ_REF, sc, mode=m))

COULEURS = {'i': '#2ca02c', 'D': '#ff7f0e', 'both': '#d62728'}
LABELS = {'i': 'Erreur sur i seul (i₁, i₂)',
          'D': 'Erreur sur D seul (disparités)',
          'both': 'Erreurs combinées (i + D)'}

# ---------------------------------------------------------
# FIGURE 3A : Impact dominant (D et Combiné) en MÈTRES
# ---------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 6))

for m in ('D', 'both'):
    c = COULEURS[m]
    ax1.fill_between(echelles, resultats[m]['q25'], resultats[m]['q75'],
                     color=c, alpha=0.15, linewidth=0)
    ax1.plot(echelles, resultats[m]['med'], color=c, lw=2.5,
             label=f"Médiane — {LABELS[m]}")
    ax1.plot(echelles, resultats[m]['wc'], color=c, lw=2.0, linestyle='--',
             label=f"Pire cas — {LABELS[m]}")

handles1, labels1 = ax1.get_legend_handles_labels()
handles1.append(mpatches.Patch(color='gray', alpha=0.3, label='Intervalle interquartile (Q1-Q3)'))

ax1.set_xlabel(r"Amplitude maximale d'erreur pixellique $\delta d_{max}$ [pixels]")
ax1.set_ylabel(r"Erreur de position $\varepsilon_{pos}$ [m]")
ax1.set_xlim(0, 1.0)
ax1.set_ylim(0, 0.7)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(handles=handles1, loc='upper left', fontsize=9, framealpha=0.95)

ax1.annotate(f"~ {resultats['both']['wc'][-1]:.2f} m",
             xy=(1.0, resultats['both']['wc'][-1]),
             xytext=(0.72, 0.48),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.tight_layout()
plt.savefig('figure_3a_bruit_dominant.png', dpi=150, bbox_inches='tight')
plt.close(fig1)

# ---------------------------------------------------------
# FIGURE 3B : Impact négligeable (i seul) en MILLIMÈTRES
# ---------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 6))
m = 'i'
c = COULEURS[m]

# Conversion des données de mètres vers MILLIMÈTRES
med_mm = np.array(resultats[m]['med']) * 1000
q25_mm = np.array(resultats[m]['q25']) * 1000
q75_mm = np.array(resultats[m]['q75']) * 1000
wc_mm  = np.array(resultats[m]['wc']) * 1000

ax2.fill_between(echelles, q25_mm, q75_mm, color=c, alpha=0.15, linewidth=0)
ax2.plot(echelles, med_mm, color=c, lw=2.5, label=f"Médiane — {LABELS[m]}")
ax2.plot(echelles, wc_mm, color=c, lw=2.0, linestyle='--', label=f"Pire cas — {LABELS[m]}")

handles2, labels2 = ax2.get_legend_handles_labels()
handles2.append(mpatches.Patch(color='gray', alpha=0.3, label='Intervalle interquartile (Q1-Q3)'))

ax2.set_xlabel(r"Amplitude maximale d'erreur pixellique $\delta d_{max}$ [pixels]")
ax2.set_ylabel(r"Erreur de position $\varepsilon_{pos}$ [mm]")
ax2.set_xlim(0, 1.0)
ax2.set_ylim(0, max(wc_mm) * 1.2) # Échelle dynamique adaptée
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(handles=handles2, loc='upper left', fontsize=9, framealpha=0.95)

ax2.annotate(f"~ {wc_mm[-1]:.1f} mm",
             xy=(1.0, wc_mm[-1]),
             xytext=(0.70, wc_mm[-1] * 0.7),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.tight_layout()
plt.savefig('figure_3b_bruit_negligeable.png', dpi=150, bbox_inches='tight')
plt.close(fig2)

print(f"  i={resultats['i']['wc'][-1]*1000:.1f}mm  "
      f"D={resultats['D']['wc'][-1]:.3f}m  "
      f"combiné={resultats['both']['wc'][-1]:.3f}m")

# =====================================================================
# FIGURE 4 — scénarios trajectoire
# =====================================================================
print("\n=== Figure 4 ===")
SCENARIOS = ['rect_0','rect_5','rect_30','R500','R50','R10']
SC_INFO = {'rect_0':  ('Rectiligne colinéaire (α = 0°)',  '#d62728'),
           'rect_5':  ('Rectiligne oblique (α = 5°)',     '#ff7f0e'),
           'rect_30': ('Rectiligne oblique (α = 30°)',    '#2ca02c'),
           'R500':    ('Courbe R = 500 m',                '#1f77b4'),
           'R50':     ('Courbe R = 50 m',                 '#9467bd'),
           'R10':     ('Courbe R = 10 m',                 '#8c564b')}
med_sc = {s: [] for s in SCENARIOS}
q25_sc = {s: [] for s in SCENARIOS}; q75_sc = {s: [] for s in SCENARIOS}
for sc in echelles:
    for s_name in SCENARIOS:
        i2_gt, D2_gt, txv, tzv = verite_terrain(s_name)
        e = monte_carlo(I_FIXE, i2_gt, D1_GT, D2_gt, txv, tzv, sc, mode='both')
        med_sc[s_name].append(np.nanmedian(e))
        q25_sc[s_name].append(np.nanpercentile(e, 25))
        q75_sc[s_name].append(np.nanpercentile(e, 75))
fig, ax = plt.subplots(figsize=(10, 6))
for s_name in SCENARIOS:
    _, c = SC_INFO[s_name]
    ax.fill_between(echelles, q25_sc[s_name], q75_sc[s_name],
                    color=c, alpha=0.04, linewidth=0)
for s_name in SCENARIOS:
    label, c = SC_INFO[s_name]
    ax.plot(echelles, med_sc[s_name], color=c, lw=2.5, label=label)
handles, _ = ax.get_legend_handles_labels()
handles.append(mpatches.Patch(color='gray', alpha=0.3,
                              label='Intervalle interquartile (Q1–Q3)'))
ax.set_xlabel(r"Amplitude maximale d'erreur pixellique $\delta d_{max}$ [pixels]")
ax.set_ylabel(r"Erreur de position $\varepsilon_{pos}$ [m]")
ax.set_xlim(0, 1.0); ax.set_ylim(bottom=0)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(handles=handles, loc='upper left', fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.savefig('figure_4_scenarios.png', dpi=150, bbox_inches='tight')
plt.close()

# =====================================================================
# FIGURE 5 — θ_FOV
# =====================================================================
print("\n=== Figure 5 — θ_FOV ===")
i_vals = np.linspace(0.0005, 0.0796, 100)
theta_vals = np.degrees(np.arctan(i_vals / F))
err_theta = []
for i1 in i_vals:
    X_M = -i1 * D1_GT / F; X_M2 = X_M - D_DRONE
    i2 = -X_M2 * F / D1_GT
    err_theta.append(worst_case(i1, i2, D1_GT, D1_GT, -D_DRONE, 0, sc=1.0))
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta_vals, err_theta, color='#d62728', lw=3.0,
        label='Pire cas combiné (i + D, ±0,5 px par image)')
ax.set_xlabel(r"Angle d'observation $\theta_{FOV}$ [°]")
ax.set_ylabel(r"Erreur de position $\varepsilon_{pos}$ (pire cas) [m]")
ax.set_xlim(0, 90); ax.set_ylim(0, 5.0)
ax.grid(True, linestyle='--', alpha=0.7); ax.legend(loc='upper right', fontsize=10)
for th_show in (5.71, 26.57):
    i1 = F*np.tan(np.radians(th_show))
    X_M = -i1*D1_GT/F; X_M2 = X_M-D_DRONE
    i2 = -X_M2*F/D1_GT
    e = worst_case(i1, i2, D1_GT, D1_GT, -D_DRONE, 0)
    ax.scatter([th_show], [e], color='black', zorder=5)
    xtxt, ytxt = (10, 3.2) if th_show == 5.71 else (30, 1.5)
    ax.annotate(f'~ {e:.2f} m ($\\theta_{{FOV}}$ = {th_show:.2f}°)',
                xy=(th_show, e), xytext=(xtxt, ytxt),
                arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.tight_layout()
plt.savefig('figure_5_theta_fov.png', dpi=150, bbox_inches='tight')
plt.close()

# =====================================================================
# FIGURE 6 — focale
# =====================================================================
print("=== Figure 6 — focale ===")
f_vals = np.linspace(0.005, 0.050, 100)
I_MAX = 0.0096
F_LIMIT_MM = (I_MAX / np.tan(np.radians(THETA_FOV_REF))) * 1000
err_f = []
for fc in f_vals:
    i1 = fc * np.tan(np.radians(THETA_FOV_REF))
    X_M = -i1*D1_GT/fc; X_M2 = X_M-D_DRONE
    i2 = -X_M2*fc/D1_GT
    err_f.append(worst_case(i1, i2, D1_GT, D1_GT, -D_DRONE, 0, sc=1.0, f=fc))
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(f_vals*1000, err_f, color='#d62728', lw=3.0, zorder=3,
        label='Pire cas combiné (i + D, ±0,5 px par image)')
ax.axvspan(F_LIMIT_MM, 50, color='gray', alpha=0.2, hatch='//',
           edgecolor='none', zorder=1,
           label='Zone hors-champ (capteur 4K)')
ax.axvline(x=F_LIMIT_MM, color='black', linestyle='--', lw=2, zorder=4)
ax.set_xlabel(r"Distance focale $f$ [mm]")
ax.set_ylabel(r"Erreur de position $\varepsilon_{pos}$ (pire cas) [m]")
ax.set_xlim(5, 50); ax.set_ylim(0, 1.5)
ax.grid(True, linestyle='--', alpha=0.7); ax.legend(loc='upper right', fontsize=10)
e_10 = err_f[np.argmin(np.abs(f_vals-0.010))]
ax.scatter([10.0], [e_10], color='black', zorder=5)
ax.annotate(f'~ {e_10:.2f} m (f = 10 mm)', xy=(10.0, e_10), xytext=(6.0, 1.0),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate(f'Limite\n(f = {F_LIMIT_MM:.1f} mm)',
            xy=(F_LIMIT_MM, 1.3), xytext=(F_LIMIT_MM+2, 1.35),
            fontweight='bold', arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.tight_layout()
plt.savefig('figure_6_focale.png', dpi=150, bbox_inches='tight')
plt.close()

# =====================================================================
# FIGURE 7 — distance D
# =====================================================================
print("=== Figure 7 — distance D ===")
D_vals = np.linspace(2.0, 20.0, 100)
err_D = []
for D in D_vals:
    i1 = I_FIXE
    X_M = -i1*D/F; X_M2 = X_M-D_DRONE
    i2 = -X_M2*F/D
    err_D.append(worst_case(i1, i2, D, D, -D_DRONE, 0, sc=1.0))
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(D_vals, err_D, color='#d62728', lw=3.0,
        label='Pire cas combiné (i + D, ±0,5 px par image)')
ax.set_xlabel(r"Distance au mur $D$ [m]")
ax.set_ylabel(r"Erreur de position $\varepsilon_{pos}$ (pire cas) [m]")
ax.set_xlim(2, 20); ax.set_ylim(0, max(err_D)*1.1)
ax.grid(True, linestyle='--', alpha=0.7); ax.legend(loc='upper left', fontsize=10)
e_10 = err_D[np.argmin(np.abs(D_vals-10))]
e_20 = err_D[-1]
ax.scatter([10, 20], [e_10, e_20], color='black', zorder=5)
ax.annotate(f'~ {e_10:.2f} m (D = 10 m)', xy=(10, e_10), xytext=(4, e_10+0.3),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate(f'~ {e_20:.2f} m (D = 20 m)', xy=(20, e_20), xytext=(13, e_20-0.4),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.tight_layout()
plt.savefig('figure_7_distance.png', dpi=150, bbox_inches='tight')
plt.close()

# =====================================================================
# FIGURE 8 — Impact du nombre N de caméras (Théorie vs Simulation MC)
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

print("\n=== Figure 8 — N caméras (Théorie Exacte vs Vraie Simulation) ===")

# Paramètres du système
F = 0.010; B = 0.20; PIXEL_SIZE = 5e-6
D1_GT = 10.0; DT = 0.02; V_DRONE = 2.78
D_DRONE = V_DRONE * DT
I_FIXE = 0.005
N_TRIALS = 10000  # 10 000 tirages comme spécifié dans le texte

N_vals = np.arange(2, 17)

def calc_modele_2(i1, i2, D1, D2, f=F):
    denom = (i1*D1 + i2*D2)**2 + f**2 * (D2-D1)**2
    K = (D2**2 - D1**2)*f**2 + D2**2*i2**2 - D1**2*i1**2
    Tx = -(1.0/f) * (D2*i2 + D1*i1) * K / denom
    Tz = -(D2 - D1) * K / denom
    return Tx, Tz

def simuler_fusion_N_cameras_MC(X_scene, Z_scene, N):
    X_cams = np.linspace(0, B, N)
    i_gt = -F * (X_scene - X_cams) / Z_scene
    
    # Bruit uniforme de +- 0.5 pixel sur chaque capteur indépendant
    bruit = np.random.uniform(-0.5, 0.5, (N, N_TRIALS))
    i_mes = i_gt[:, None] + bruit * PIXEL_SIZE

    sum_num = np.zeros(N_TRIALS)
    sum_den = 0.0
    for u in range(N):
        for v in range(u+1, N):
            b_uv = X_cams[v] - X_cams[u]
            d_uv_mes = i_mes[v] - i_mes[u] 
            
            sum_num += b_uv * d_uv_mes
            sum_den += b_uv**2

    D_est = (F * sum_den) / sum_num
    i_ref = i_mes[0] 
    
    return D_est, i_ref

# --- 1. LANCEMENT DE LA SIMULATION ---
median_simule = []
lower_bound = []
upper_bound = []

X1_scene = -I_FIXE * D1_GT / F
X2_scene = X1_scene - D_DRONE

for N in N_vals:
    D1_est, i1_est = simuler_fusion_N_cameras_MC(X1_scene, D1_GT, N)
    D2_est, i2_est = simuler_fusion_N_cameras_MC(X2_scene, D1_GT, N)

    tx, tz = calc_modele_2(i1_est, i2_est, D1_est, D2_est)
    err = np.sqrt((tx - (-D_DRONE))**2 + (tz - 0)**2)
    
    # Extraction de la médiane et de la marge d'erreur (Percentiles 5% et 95%)
    median_simule.append(np.median(err))
    lower_bound.append(np.percentile(err, 5))
    upper_bound.append(np.percentile(err, 95))

# --- 2. CALCUL DES THÉORIES COHÉRENTES AVEC LA MÉDIANE ---
ERR_REF_N2 = median_simule[0]  # Référence prise sur la médiane à N=2 (~0.125m)

# Théorie "Naïve" (rouge en pointillé dans le texte)
G_N_naif = np.sqrt(N_vals**2 * (N_vals + 1) / (12 * (N_vals - 1)))
err_N_naif = ERR_REF_N2 / G_N_naif

# Théorie "Exacte" (verte en pointillé dans le texte)
G_N_vrai = np.sqrt(N_vals * (N_vals + 1) / (6 * (N_vals - 1)))
err_N_vrai = ERR_REF_N2 / G_N_vrai

# --- 3. TRACÉ DU GRAPHIQUE ---
fig, ax = plt.subplots(figsize=(10, 6))

# Zone ombrée : Marge d'erreur de la simulation (dispersion à 90% des tirages)
ax.fill_between(N_vals, lower_bound, upper_bound, color='blue', alpha=0.12, zorder=1,
                label="Marge d'erreur de la simulation (Percentiles 5-95)")

# Courbe rouge en pointillé : Prédiction naïve
ax.plot(N_vals, err_N_naif, color='red', lw=2.0, linestyle='--', zorder=2,
        label=r'Courbe rouge : Prédiction naïve $\varepsilon_{pos}(N=2) / G_{naif}(N)$')

# Courbe verte en pointillé : Projection théorique exacte
ax.plot(N_vals, err_N_vrai, color='green', lw=2.0, linestyle='--', zorder=3,
        label=r'Courbe verte : Théorie exacte $\varepsilon_{pos}(N=2) / G_{vrai}(N)$')

# Courbe bleue continue : Médiane Monte-Carlo
ax.plot(N_vals, median_simule, color='blue', lw=2.5, marker='o', markersize=6, zorder=4,
        label='Courbe bleue : Médiane Monte-Carlo simulée')

# Configuration des axes
ax.set_xlabel(r"Nombre $N$ de caméras alignées")
ax.set_ylabel(r"Erreur de position $\varepsilon_{pos}$ [m]")
ax.set_xticks(N_vals)
ax.set_xlim(2, 16)
ax.set_ylim(0, max(upper_bound) * 1.05)  # Adapté pour voir la marge d'erreur haute à N=2
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper right', fontsize=10)

# Annotations (sur la courbe bleue de la simulation médiane)
for n_show in (2, 4, 10):
    idx = n_show - 2
    ax.annotate(f"~ {median_simule[idx]:.3f} m (N = {n_show})",
                xy=(n_show, median_simule[idx]), 
                xytext=(n_show + 0.8, median_simule[idx] + 0.02),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2))

plt.tight_layout()
plt.savefig('figure_8_N_cameras_theorie_vs_pratique.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  Médiane Simulée N=2  : {median_simule[0]:.3f} m  (Cas de référence)")
print(f"  Médiane Simulée N=4  : {median_simule[2]:.3f} m  (Conforme au texte: ~0.118 m)")
print(f"  Médiane Simulée N=10 : {median_simule[8]:.3f} m  (Conforme au texte: ~0.087 m)")