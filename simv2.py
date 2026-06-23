"""
Simulation d'odométrie par stéréovision - Master Code Complet
--------------------------------------------------------------
PHASE 0 : Vérification géométrique (Sans Bruit)
PHASE 1 : Analyse des sources de bruit (i, D, Combiné)
PHASE 2 : Analyse de l'influence de K (1, 100, 500, 1500)
PHASE 3 : Analyse de l'influence de N (2, 4, 8, 16)
--------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ============================================================
# PARAMÈTRES GÉNÉRAUX
# ============================================================
DOSSIER_SORTIE = os.getcwd()

FPS = 50
DT = 1.0 / FPS
V_MS = 10.0 / 3.6
FOCALE = 0.01
BASELINE_TOT = 0.20   # b_tot : encombrement total fixe
PIXEL_SIZE = 5e-6
BRUIT_MAX_PX = 0.5
DIST_OBJECTIF = 500.0
INTENSITE_DELTA = 0.002
EPSILON_ANGLE = 1e-12

D_NOMINAL = 10.0
SIGMA_FACADE = 0.3
RATIO_FACADE = 0.8
ANGLE_MAX_DEG = 60.0
ANGLE_MIN_DEG = 3.0

# Configurations Phases 1 et 2
K_FIXE_POUR_BRUIT = 1
MC_POUR_BRUIT = 2000

CONFIGS_K = {
    1: 2000,
    100: 200,
    500: 50,
    1500: 20
}

# Configuration Phase 3 (N caméras avec K fixé)
K_FIXE_POUR_N = 1
CONFIGS_N = {
    2: 2000,
    4: 2000,
    8: 2000,
    16: 2000
}

# ============================================================
# 1. MODÈLE MATHÉMATIQUE
# ============================================================
def modele_2_vec(i1, i2, D1, D2, f=FOCALE):
    denom = ((i1 * D1 + i2 * D2) ** 2 + (f ** 2) * (D2 - D1) ** 2)
    Knum = ((D2 ** 2 - D1 ** 2) * (f ** 2) + D2 ** 2 * i2 ** 2 - D1 ** 2 * i1 ** 2)
    Tx = np.where(denom > 1e-15, -(1.0 / f) * (D2 * i2 + D1 * i1) * Knum / denom, 0.0)
    Tz = np.where(denom > 1e-15, -(D2 - D1) * Knum / denom, 0.0)
    return Tx, Tz

def calc_odo(i1_obs, i2_obs, D1_obs, D2_obs, phi_prev, x_prev, z_prev):
    Tx_pts, Tz_pts = modele_2_vec(i1_obs, i2_obs, D1_obs, D2_obs)
    Tx, Tz = np.mean(Tx_pts, axis=1), np.mean(Tz_pts, axis=1)
    theta = np.where(np.abs(Tx) > 1e-12, -2.0 * np.arctan2(Tz, Tx), 0.0)
    phi_new = phi_prev + theta
    vx = -Tx * np.cos(phi_prev) - Tz * np.sin(phi_prev)
    vz = -Tx * np.sin(phi_prev) + Tz * np.cos(phi_prev)
    return x_prev + vx, z_prev + vz, phi_new

# ============================================================
# 2. FUSION MULTI-CAMÉRAS SOUS H2 (INVERSE-VARIANCE)
# ============================================================
def mesure_profondeur_multi_cam(D_true, N, n_mc, f=FOCALE, b_tot=BASELINE_TOT):
    """
    Simule la mesure de profondeur avec N caméras alignées sur b_tot fixe,
    sous l'hypothèse H2 (capteurs physiquement partagés) avec estimateur
    par inverse-variance, conformément à la NE Comparaison numérique [2].
    
    D_p_est = (f * S_B2) / Sum_(i<j) B_ij * d_ij_obs
    avec d_ij_obs = d_ij_true + (delta_p_i - delta_p_j)
    
    Parameters
    ----------
    D_true : ndarray, shape (K_valid,)
        Profondeurs vraies des K_valid points
    N      : int >= 2
        Nombre de caméras alignées
    n_mc   : int
        Nombre de tirages Monte-Carlo
    
    Returns
    -------
    D_obs  : ndarray, shape (n_mc, K_valid)
        Profondeurs estimées
    """
    K_valid = len(D_true)
    b0 = b_tot / (N - 1)
    
    # Bruit pixellique INDÉPENDANT sur chaque capteur k, chaque point p, chaque tirage MC
    delta_p = np.random.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX, (n_mc, N, K_valid)) * PIXEL_SIZE
    
    # Construction des coefficients :
    # Sum_(i<j) B_ij * (delta_p_i - delta_p_j) = Sum_k coef_k * delta_p_k
    S_B2 = 0.0
    coef_k = np.zeros(N)
    for i in range(N):
        for j in range(i + 1, N):
            B_ij = (j - i) * b0
            S_B2 += B_ij ** 2
            coef_k[i] += B_ij
            coef_k[j] -= B_ij
    
    # Estimateur : D_obs = (f * S_B2) / [(f * S_B2)/D_true + Sum_k coef_k * delta_p_k]
    noise_q = np.einsum('k,mkp->mp', coef_k, delta_p)
    denom_est = (f * S_B2) / D_true[None, :] + noise_q
    D_obs = (f * S_B2) / denom_est
    
    return D_obs

# ============================================================
# 3. VÉRITÉ TERRAIN ET ENVIRONNEMENT
# ============================================================
print("\n" + "="*60)
print("ÉTAPE 1 : GÉNÉRATION DU MONDE ET DE LA VÉRITÉ TERRAIN")
print("="*60)
x_d, z_d, phi_d = [0.0], [10.0], [0.0]

while x_d[-1] < DIST_OBJECTIF:
    phi = phi_d[-1]
    valide = False
    while not valide:
        dphi = np.random.uniform(-INTENSITE_DELTA, INTENSITE_DELTA)
        if z_d[-1] > 12 and phi > 0: dphi = np.random.uniform(-INTENSITE_DELTA, 0)
        if z_d[-1] < 8 and phi < 0: dphi = np.random.uniform(0, INTENSITE_DELTA)

        if abs(dphi) < EPSILON_ANGLE:
            dx, dz = V_MS * DT * np.cos(phi), V_MS * DT * np.sin(phi)
        else:
            R = (V_MS * DT) / dphi
            dx = R * (np.sin(phi + dphi) - np.sin(phi))
            dz = R * (np.cos(phi) - np.cos(phi + dphi))

        if dx > 0:
            valide = True
            x_d.append(x_d[-1] + dx)
            z_d.append(z_d[-1] + dz)
            phi_d.append(phi + dphi)

x_d, z_d, phi_d = np.array(x_d), np.array(z_d), np.array(phi_d)
N_PAS = len(x_d) - 1
print(f" -> Trajectoire globale générée : {N_PAS} itérations.\n")

def genere_profondeur(K):
    if K == 1: return np.array([D_NOMINAL])
    K_facade = int(K * RATIO_FACADE)
    K_saillie = K - K_facade
    z_facade = np.random.normal(0, SIGMA_FACADE, K_facade)
    masque = (z_facade < -5) | (z_facade > 5)
    while masque.any():
        z_facade[masque] = np.random.normal(0, SIGMA_FACADE, masque.sum())
        masque = (z_facade < -5) | (z_facade > 5)
    z_saillie = np.random.uniform(-5, 5, K_saillie)
    z_wall = np.concatenate([z_facade, z_saillie])
    np.random.shuffle(z_wall)
    return D_NOMINAL + z_wall

# ============================================================
# 4. LE MOTEUR CENTRAL (accepte N_cam)
# ============================================================
def run_simulation(K, n_mc, mode_bruit, injecter_bruit=True, N_cam=2):
    """
    N_cam : nombre de caméras (>= 2)
        - N_cam = 2 : paire stéréo simple avec b_tot
        - N_cam > 2 : fusion multi-baseline par inverse-variance sous H2
    """
    np.random.seed(42 + K + 1000 * N_cam)
    
    x_c = np.zeros((n_mc, N_PAS + 1))
    z_c = np.zeros((n_mc, N_PAS + 1))
    phi_c = np.zeros((n_mc, N_PAS + 1))
    x_c[:, 0], z_c[:, 0] = 0.0, 10.0
    err_c = np.zeros((n_mc, N_PAS + 1))

    for n in range(N_PAS):
        x1, z1, phi1 = x_d[n], z_d[n], phi_d[n]
        x2, z2, phi2 = x_d[n + 1], z_d[n + 1], phi_d[n + 1]

        # Génération des K angles avec exclusion centrale
        angles = np.random.uniform(-np.deg2rad(ANGLE_MAX_DEG), np.deg2rad(ANGLE_MAX_DEG), K)
        mask = np.abs(angles) < np.deg2rad(ANGLE_MIN_DEG)
        while np.any(mask):
            angles[mask] = np.random.uniform(-np.deg2rad(ANGLE_MAX_DEG), np.deg2rad(ANGLE_MAX_DEG), np.sum(mask))
            mask = np.abs(angles) < np.deg2rad(ANGLE_MIN_DEG)

        # Génération des profondeurs (distribution mixte 80/20)
        D_world = genere_profondeur(K)
        Xw = x1 + D_world * np.cos(phi1 + angles)
        Zw = z1 + D_world * np.sin(phi1 + angles)

        # Projection dans le repère caméra pour les 2 instants
        dx1, dz1 = Xw - x1, Zw - z1
        Xc1 = dx1 * np.cos(phi1) + dz1 * np.sin(phi1)
        D1_true = dx1 * np.sin(phi1) - dz1 * np.cos(phi1)
        dx2, dz2 = Xw - x2, Zw - z2
        Xc2 = dx2 * np.cos(phi2) + dz2 * np.sin(phi2)
        D2_true = dx2 * np.sin(phi2) - dz2 * np.cos(phi2)

        # Filtrage de validité
        valid = (np.abs(D1_true) > 0.5) & (np.abs(D2_true) > 0.5)
        Xc1, Xc2 = Xc1[valid], Xc2[valid]
        D1_true, D2_true = D1_true[valid], D2_true[valid]
        K_valid = len(D1_true)

        if K_valid < 1:
            x_c[:, n+1], z_c[:, n+1], phi_c[:, n+1] = x_c[:, n], z_c[:, n], phi_c[:, n]
            err_c[:, n+1] = err_c[:, n]
            continue

        i1_true = -FOCALE * Xc1 / D1_true
        i2_true = -FOCALE * Xc2 / D2_true

        # Bruit sur les coordonnées image
        if injecter_bruit:
            bruit_i1 = np.random.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX, (n_mc, K_valid)) * PIXEL_SIZE
            bruit_i2 = np.random.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX, (n_mc, K_valid)) * PIXEL_SIZE
        else:
            bruit_i1, bruit_i2 = 0.0, 0.0

        # Mesure de profondeur : choix entre 2-cam et N-cam
        if N_cam == 2:
            # Mode 2-caméras classique : baseline = b_tot
            if injecter_bruit:
                bruit_d1 = (np.random.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX, (n_mc, K_valid)) -
                            np.random.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX, (n_mc, K_valid))) * PIXEL_SIZE
                bruit_d2 = (np.random.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX, (n_mc, K_valid)) -
                            np.random.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX, (n_mc, K_valid))) * PIXEL_SIZE
            else:
                bruit_d1, bruit_d2 = 0.0, 0.0
            delta_D1 = (D1_true**2 / (FOCALE * BASELINE_TOT)) * bruit_d1
            delta_D2 = (D2_true**2 / (FOCALE * BASELINE_TOT)) * bruit_d2
            D1_noisy = D1_true[None, :] + delta_D1
            D2_noisy = D2_true[None, :] + delta_D2
        else:
            # Mode N-caméras : fusion inverse-variance sous H2
            if injecter_bruit:
                D1_noisy = mesure_profondeur_multi_cam(D1_true, N_cam, n_mc)
                D2_noisy = mesure_profondeur_multi_cam(D2_true, N_cam, n_mc)
            else:
                D1_noisy = np.tile(D1_true, (n_mc, 1))
                D2_noisy = np.tile(D2_true, (n_mc, 1))

        # Sélection du mode de bruit
        if mode_bruit == 'i':
            i1_obs, i2_obs = i1_true[None, :] + bruit_i1, i2_true[None, :] + bruit_i2
            D1_obs, D2_obs = np.tile(D1_true, (n_mc, 1)), np.tile(D2_true, (n_mc, 1))
        elif mode_bruit == 'D':
            i1_obs, i2_obs = np.tile(i1_true, (n_mc, 1)), np.tile(i2_true, (n_mc, 1))
            D1_obs, D2_obs = D1_noisy, D2_noisy
        else:  # 'combine'
            i1_obs, i2_obs = i1_true[None, :] + bruit_i1, i2_true[None, :] + bruit_i2
            D1_obs, D2_obs = D1_noisy, D2_noisy

        x_c[:, n+1], z_c[:, n+1], phi_c[:, n+1] = calc_odo(
            i1_obs, i2_obs, D1_obs, D2_obs, phi_c[:, n], x_c[:, n], z_c[:, n])
        err_c[:, n+1] = np.sqrt((x_c[:, n+1] - x_d[n+1])**2 + (z_c[:, n+1] - z_d[n+1])**2)

    return x_c, z_c, err_c

# Utilitaires pour les graphes
def filepath(n):
    return os.path.join(DOSSIER_SORTIE, n)
iterations = np.arange(N_PAS + 1)
dist_parcourue = iterations * V_MS * DT


# ==============================================================================
# PHASE 0 : SANS BRUIT (Vérification géométrique)
# ==============================================================================
print("\n" + "="*60)
print("PHASE 0 : SANS BRUIT (Vérification géométrique Modèle 2)")
print("="*60)

t0 = time.time()
x0, z0, err0 = run_simulation(K=1, n_mc=1, mode_bruit='combine', injecter_bruit=False, N_cam=2)
x0, z0, err0 = x0[0], z0[0], err0[0]
print(f" -> Terminé en {time.time()-t0:.1f}s.")
print(f" -> Erreur finale  : {err0[-1]:.3e} m")
print(f" -> Erreur max     : {err0.max():.3e} m")

# Figure 0A : Trajectoire
fig, axA = plt.subplots(figsize=(12, 5))
axA.plot(x_d, z_d, color='black', lw=2.5, label='Vérité terrain')
axA.plot(x0, z0, color='red', lw=1.5, linestyle='--', label='Estimée (Modèle 2)')
axA.set_xlabel('X [m]'); axA.set_ylabel('Z [m]')
axA.set_title(f"SANS bruit — Vérification géométrique (Erreur finale = {err0[-1]:.2e} m)", fontweight='bold')
axA.legend(); axA.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(filepath('Fig0a_Sans_Bruit_Trajectoire.png'), dpi=120, bbox_inches='tight')
plt.close()

# Figure 0B : Erreur (échelle log)
fig, axB = plt.subplots(figsize=(11, 4.5))
axB.plot(dist_parcourue, err0, color='green', lw=1.5)
axB.set_xlabel('Distance parcourue [m]'); axB.set_ylabel(r'Erreur $\varepsilon_{pos}$ [m]')
axB.set_title(f"SANS bruit — Erreur résiduelle numérique (Max = {err0.max():.2e} m)", fontweight='bold')
axB.set_yscale('log')
axB.grid(True, linestyle='--', alpha=0.7, which='both')
plt.tight_layout()
plt.savefig(filepath('Fig0b_Sans_Bruit_Erreur.png'), dpi=120, bbox_inches='tight')
plt.close()


# ==============================================================================
# PHASE 1 : ANALYSE DES SOURCES DE BRUIT
# ==============================================================================
print("\n" + "="*60)
print("PHASE 1 : ANALYSE DES SOURCES DE BRUIT (i vs D)")
print("="*60)

stats_bruit = {}
for cas in ['i', 'D', 'combine']:
    print(f" -> Calcul du cas : {cas}...")
    x_e, z_e, err_e = run_simulation(K=K_FIXE_POUR_BRUIT, n_mc=MC_POUR_BRUIT,
                                      mode_bruit=cas, injecter_bruit=True, N_cam=2)
    stats_bruit[cas] = {
        'x': x_e, 'z': z_e, 'err': err_e,
        'med': np.median(err_e, axis=0),
        'q25': np.percentile(err_e, 25, axis=0),
        'q75': np.percentile(err_e, 75, axis=0)
    }

idx_med = np.argsort(stats_bruit['combine']['err'][:, -1])[MC_POUR_BRUIT // 2]

# Figure 1a : Bruit i seul
fig, axes = plt.subplots(2, 1, figsize=(13, 10))
ax1 = axes[0]
ax1.fill_between(iterations, stats_bruit['i']['q25'], stats_bruit['i']['q75'], color='#4daf4a', alpha=0.20)
ax1.plot(iterations, stats_bruit['i']['med'], color='#4daf4a', lw=2.0)
ax1.set_title(f"Bruit sur la coordonnée image 'i' seul (Dérive finale = {stats_bruit['i']['med'][-1]:.2f}m)")
ax1.set_ylabel("Erreur [m]"); ax1.grid(True)
ax2 = axes[1]
ax2.plot(x_d, z_d, 'k', lw=2.0, label="Vérité")
ax2.plot(stats_bruit['i']['x'][idx_med, :], stats_bruit['i']['z'][idx_med, :],
         color='#4daf4a', lw=2.0, linestyle='--', label="Estimée (Médiane)")
ax2.set_xlabel("X [m]"); ax2.set_ylabel("Z [m]"); ax2.grid(True); ax2.legend()
plt.tight_layout()
plt.savefig(filepath("Fig1a_Bruit_i.png"), dpi=150)
plt.close()

# Figure 1b : Bruit D et combiné
fig, axes = plt.subplots(2, 1, figsize=(13, 10))
ax1 = axes[0]
ax1.fill_between(iterations, stats_bruit['D']['q25'], stats_bruit['D']['q75'], color='#e41a1c', alpha=0.15)
ax1.plot(iterations, stats_bruit['D']['med'], color='#e41a1c', lw=4.0, alpha=0.7,
         label=f"Bruit D seul (Erreur: {stats_bruit['D']['med'][-1]:.2f}m)")
ax1.plot(iterations, stats_bruit['combine']['med'], color='#377eb8', lw=2.0, linestyle='--',
         label=f"Bruit Combiné (Erreur: {stats_bruit['combine']['med'][-1]:.2f}m)")
ax1.set_title("Superposition : Le bruit sur D et le bruit combiné (i+D)")
ax1.set_ylabel("Erreur [m]"); ax1.grid(True); ax1.legend(loc='upper left')
ax2 = axes[1]
ax2.plot(x_d, z_d, 'k', lw=2.0, label="Vérité terrain")
ax2.plot(stats_bruit['D']['x'][idx_med, :], stats_bruit['D']['z'][idx_med, :],
         color='#e41a1c', lw=4.0, alpha=0.7, label="Trajectoire D seul")
ax2.plot(stats_bruit['combine']['x'][idx_med, :], stats_bruit['combine']['z'][idx_med, :],
         color='#377eb8', lw=2.0, linestyle='--', label="Trajectoire Combinée")
ax2.set_xlabel("X [m]"); ax2.set_ylabel("Z [m]"); ax2.grid(True); ax2.legend()
plt.tight_layout()
plt.savefig(filepath("Fig1b_Bruit_D_vs_Combine.png"), dpi=150)
plt.close()


# ==============================================================================
# PHASE 2 : INFLUENCE DE K
# ==============================================================================
print("\n" + "="*60)
print("PHASE 2 : INFLUENCE DE K (Bruit Combiné, N=2)")
print("="*60)

resultats_K = {}
colors_K = {1: "#e41a1c", 100: "#d95f02", 500: "#1b9e77", 1500: "#377eb8"}

for K, NMC in CONFIGS_K.items():
    if K == K_FIXE_POUR_BRUIT and NMC == MC_POUR_BRUIT:
        print(f" -> Récupération directe de la PHASE 1 pour K = {K} ({NMC} MC)...")
        x_e = stats_bruit['combine']['x']
        z_e = stats_bruit['combine']['z']
        err = stats_bruit['combine']['err']
    else:
        print(f" -> Calcul pour K = {K} ({NMC} MC)...")
        x_e, z_e, err = run_simulation(K, NMC, mode_bruit='combine',
                                        injecter_bruit=True, N_cam=2)
    
    med = np.median(err, axis=0)
    q25 = np.percentile(err, 25, axis=0)
    q75 = np.percentile(err, 75, axis=0)
    idx = np.argsort(err[:, -1])[len(err)//2]
    resultats_K[K] = {"med": med, "q25": q25, "q75": q75, "x": x_e[idx], "z": z_e[idx]}

# Graphe erreur en fonction de K
plt.figure(figsize=(12, 7))
for K in [1, 100, 500, 1500]:
    r = resultats_K[K]
    plt.fill_between(dist_parcourue, r["q25"], r["q75"], alpha=0.15, color=colors_K[K])
    plt.plot(dist_parcourue, r["med"], lw=2.5, color=colors_K[K], label=f"K = {K}")
plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
plt.xlabel("Distance parcourue [m]"); plt.ylabel("Erreur [m]")
plt.title("Convergence statistique de l'erreur selon K")
plt.tight_layout(); plt.savefig(filepath("Fig2a_Analyse_K_Erreur.png"), dpi=150); plt.close()

# Graphe trajectoire en fonction de K
plt.figure(figsize=(12, 7))
plt.plot(x_d, z_d, 'k', lw=3, label='Vérité terrain')
for K in [1, 100, 500, 1500]:
    r = resultats_K[K]
    plt.plot(r["x"], r["z"], '--', lw=2, color=colors_K[K], label=f"K = {K}")
plt.scatter([x_d[0]], [z_d[0]], color='green', s=100)
plt.scatter([x_d[-1]], [z_d[-1]], color='orange', s=100)
plt.grid(True); plt.legend(); plt.xlabel("X [m]"); plt.ylabel("Z [m]")
plt.title("Impact de K sur la dérive de la trajectoire")
plt.tight_layout(); plt.savefig(filepath("Fig2b_Analyse_K_Trajectoire.png"), dpi=150); plt.close()


# ==============================================================================
# PHASE 3 : INFLUENCE DE N (CAMÉRAS)
# ==============================================================================
print("\n" + "="*60)
print(f"PHASE 3 : INFLUENCE DE N (Bruit Combiné, K = {K_FIXE_POUR_N})")
print("="*60)

resultats_N = {}
colors_N = {2: "#e41a1c", 4: "#d95f02", 8: "#1b9e77", 16: "#377eb8"}

for N_val, NMC in CONFIGS_N.items():
    print(f" -> Calcul pour N = {N_val} ({NMC} MC)...")
    t0 = time.time()
    x_e, z_e, err = run_simulation(K=K_FIXE_POUR_N, n_mc=NMC, mode_bruit='combine',
                                    injecter_bruit=True, N_cam=N_val)
    print(f"    Temps : {time.time()-t0:.1f}s, Err finale médiane = {np.median(err[:,-1]):.3f} m")
    
    med = np.median(err, axis=0)
    q25 = np.percentile(err, 25, axis=0)
    q75 = np.percentile(err, 75, axis=0)
    idx = np.argsort(err[:, -1])[len(err) // 2]
    resultats_N[N_val] = {"med": med, "q25": q25, "q75": q75, "x": x_e[idx], "z": z_e[idx]}

# Graphe erreur en fonction de N
plt.figure(figsize=(12, 7))
for N_val in [2, 4, 8, 16]:
    r = resultats_N[N_val]
    plt.fill_between(dist_parcourue, r["q25"], r["q75"], alpha=0.15, color=colors_N[N_val])
    plt.plot(dist_parcourue, r["med"], lw=2.5, color=colors_N[N_val],
             label=f"N = {N_val} caméras (err finale = {r['med'][-1]:.2f} m)")
plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
plt.xlabel("Distance parcourue [m]")
plt.ylabel("Erreur de position [m]")
plt.title(f"Influence du nombre N de caméras (K = {K_FIXE_POUR_N}, encombrement fixe b_tot = {BASELINE_TOT} m)")
plt.tight_layout()
plt.savefig(filepath("Fig3a_Analyse_N_Erreur.png"), dpi=150)
plt.close()

# Graphe trajectoires en fonction de N
plt.figure(figsize=(12, 7))
plt.plot(x_d, z_d, 'k', lw=3, label='Vérité terrain')
for N_val in [2, 4, 8, 16]:
    r = resultats_N[N_val]
    plt.plot(r["x"], r["z"], '--', lw=2, color=colors_N[N_val], label=f"N = {N_val} caméras")
plt.scatter([x_d[0]], [z_d[0]], color='green', s=100)
plt.scatter([x_d[-1]], [z_d[-1]], color='orange', s=100)
plt.grid(True); plt.legend(); plt.xlabel("X [m]"); plt.ylabel("Z [m]")
plt.title(f"Impact de N sur la dérive (K = {K_FIXE_POUR_N})")
plt.tight_layout()
plt.savefig(filepath("Fig3b_Analyse_N_Trajectoire.png"), dpi=150)
plt.close()


# ==============================================================================
# COMPARAISON GAIN THÉORIQUE H2 vs OBSERVÉ
# ==============================================================================
print("\n" + "="*60)
print("VÉRIFICATION : Gain théorique H2 vs observé")
print("="*60)

def gain_H2_MC(N):
    """Gain théorique sous H2 par inverse-variance (NE Comparaison [2])."""
    return np.sqrt(N * (N + 1) / (6.0 * (N - 1)))

err_N2 = resultats_N[2]["med"][-1]
print(f"\n  Erreur finale médiane à N=2 : {err_N2:.3f} m (référence)\n")
print(f"  {'N':>4} | {'G_th (N/2)':>10} | {'Err théo':>10} | {'Err obs':>10} | {'Ratio obs/th':>12}")
print(f"  {'-'*4} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*12}")
for N_val in [2, 4, 8, 16]:
    G_th = gain_H2_MC(N_val) / gain_H2_MC(2)
    err_th = err_N2 / G_th
    err_obs = resultats_N[N_val]["med"][-1]
    ratio = err_obs / err_th
    print(f"  {N_val:>4} | {G_th:>10.3f} | {err_th:>10.3f} | {err_obs:>10.3f} | {ratio:>12.3f}")

print(f"\n--- TERMINÉ ! Figures générées dans : {DOSSIER_SORTIE} ---")