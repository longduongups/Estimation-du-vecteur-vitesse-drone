"""
Simulation d'odométrie par stéréovision — Analyse de l'influence de N
----------------------------------------------------------------------
PHASE UNIQUE : Influence du nombre N de caméras (2, 4, 8, 16)
               à K = 1500 fixé, bruit combiné (i + D)

D_world = D_NOMINAL = 10 m pour TOUS les points : neutralise
l'hétérogénéité des variances et permet une comparaison directe
au gain théorique G_H2,MC(N) sous H2.

Ancrage physique strict sur N_REF=16 positions de référence.
----------------------------------------------------------------------
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
BASELINE_TOT = 0.20      # b_tot : encombrement total fixe
PIXEL_SIZE = 5e-6
BRUIT_MAX_PX = 0.5
DIST_OBJECTIF = 100
INTENSITE_DELTA = 0.002
EPSILON_ANGLE = 1e-12

D_NOMINAL = 10
ANGLE_MAX_DEG = 26.0
ANGLE_MIN_DEG = 25.0

# Nombre de positions physiques de référence (ancrage de bruit)
N_REF = 16

# Configuration : K fixé, N balayé
K_FIXE = 100
N_LIST = [2, 4, 8, 16]
N_MC = 1000

# ============================================================
# 1. MODÈLE MATHÉMATIQUE
# ============================================================
def modele_2_vec(i1, i2, D1, D2, f=FOCALE):
    denom = ((i1 * D1 + i2 * D2) ** 2 + (f ** 2) * (D2 - D1) ** 2)
    Knum = ((D2 ** 2 - D1 ** 2) * (f ** 2) + D2 ** 2 * i2 ** 2 - D1 ** 2 * i1 ** 2)
    Tx = np.where(denom > 1e-15,
                  -(1.0 / f) * (D2 * i2 + D1 * i1) * Knum / denom, 0.0)
    Tz = np.where(denom > 1e-15,
                  -(D2 - D1) * Knum / denom, 0.0)
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
# 2. ESTIMATEUR INVERSE-VARIANCE SOUS H2
# ============================================================
def precalc_coefs_H2(N, b_tot=BASELINE_TOT):
    b0 = b_tot / (N - 1)
    S_B2 = 0.0
    coef_k = np.zeros(N)
    for i in range(N):
        for j in range(i + 1, N):
            B_ij = (j - i) * b0
            S_B2 += B_ij ** 2
            coef_k[i] += B_ij
            coef_k[j] -= B_ij
    return S_B2, coef_k

def mesure_profondeur_H2(D_true, delta_p, S_B2, coef_k, f=FOCALE):
    noise_q = np.einsum('k,mkp->mp', coef_k, delta_p)
    denom_est = (f * S_B2) / D_true[None, :] + noise_q
    return (f * S_B2) / denom_est

# ============================================================
# 3. VÉRITÉ TERRAIN
# ============================================================
print("\n" + "="*60)
print("ÉTAPE 1 : GÉNÉRATION DU MONDE ET DE LA VÉRITÉ TERRAIN")
print("="*60)

np.random.seed(42)
x_d, z_d, phi_d = [0.0], [10.0], [0.0]

while x_d[-1] < DIST_OBJECTIF:
    phi = phi_d[-1]
    valide = False
    while not valide:
        dphi = np.random.uniform(-INTENSITE_DELTA, INTENSITE_DELTA)
        if z_d[-1] > 11 and phi > 0:
            dphi = np.random.uniform(-INTENSITE_DELTA, 0)
        if z_d[-1] < 9 and phi < 0:
            dphi = np.random.uniform(0, INTENSITE_DELTA)

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

def genere_profondeur(K, rng_geo):
    """
    PROFONDEUR UNIFORME : tous les K points à D = D_NOMINAL.
    Cette configuration neutralise l'hétérogénéité des variances de D
    (var(D_obs) ∝ D⁴), et permet une comparaison directe au gain
    théorique G_H2,MC(N) sous H2.
    """
    return D_NOMINAL * np.ones(K)

# ============================================================
# 4. SIMULATION MUTUALISÉE
# ============================================================
def run_simulation_mutualisee(K, n_mc, N_list, N_ref=N_REF):
    rng_geo = np.random.RandomState(42 + K)
    rng_noise = np.random.RandomState(100 + K)

    x_c   = {N: np.zeros((n_mc, N_PAS + 1)) for N in N_list}
    z_c   = {N: np.zeros((n_mc, N_PAS + 1)) for N in N_list}
    phi_c = {N: np.zeros((n_mc, N_PAS + 1)) for N in N_list}
    err_c = {N: np.zeros((n_mc, N_PAS + 1)) for N in N_list}
    for N in N_list:
        x_c[N][:, 0], z_c[N][:, 0] = 0.0, 10.0

    idx_phys_dict = {N: np.linspace(0, N_ref - 1, N).round().astype(int)
                     for N in N_list}
    coefs_dict = {N: precalc_coefs_H2(N) for N in N_list}

    t_start = time.time()
    for n in range(N_PAS):
        if n > 0 and n % 1000 == 0:
            elapsed = time.time() - t_start
            eta = elapsed * (N_PAS - n) / n
            print(f"    Iter {n}/{N_PAS} — élapsé {elapsed:.0f}s, ETA {eta:.0f}s")

        x1, z1, phi1 = x_d[n], z_d[n], phi_d[n]
        x2, z2, phi2 = x_d[n + 1], z_d[n + 1], phi_d[n + 1]

        angles = rng_geo.uniform(-np.deg2rad(ANGLE_MAX_DEG),
                                  np.deg2rad(ANGLE_MAX_DEG), K)
        mask = np.abs(angles) < np.deg2rad(ANGLE_MIN_DEG)
        while np.any(mask):
            angles[mask] = rng_geo.uniform(-np.deg2rad(ANGLE_MAX_DEG),
                                            np.deg2rad(ANGLE_MAX_DEG),
                                            np.sum(mask))
            mask = np.abs(angles) < np.deg2rad(ANGLE_MIN_DEG)

        D_world = genere_profondeur(K, rng_geo)
        Xw = x1 + D_world * np.cos(phi1 + angles)
        Zw = z1 + D_world * np.sin(phi1 + angles)

        dx1, dz1 = Xw - x1, Zw - z1
        Xc1 = dx1 * np.cos(phi1) + dz1 * np.sin(phi1)
        D1_true = dx1 * np.sin(phi1) - dz1 * np.cos(phi1)
        dx2, dz2 = Xw - x2, Zw - z2
        Xc2 = dx2 * np.cos(phi2) + dz2 * np.sin(phi2)
        D2_true = dx2 * np.sin(phi2) - dz2 * np.cos(phi2)

        valid = (np.abs(D1_true) > 0.5) & (np.abs(D2_true) > 0.5)
        Xc1, Xc2 = Xc1[valid], Xc2[valid]
        D1_true, D2_true = D1_true[valid], D2_true[valid]
        K_valid = len(D1_true)

        if K_valid < 1:
            for N in N_list:
                x_c[N][:, n+1] = x_c[N][:, n]
                z_c[N][:, n+1] = z_c[N][:, n]
                phi_c[N][:, n+1] = phi_c[N][:, n]
                err_c[N][:, n+1] = err_c[N][:, n]
            continue

        i1_true = -FOCALE * Xc1 / D1_true
        i2_true = -FOCALE * Xc2 / D2_true

        bruit_full_1 = rng_noise.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX,
                                          (n_mc, N_ref, K_valid)) * PIXEL_SIZE
        bruit_full_2 = rng_noise.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX,
                                          (n_mc, N_ref, K_valid)) * PIXEL_SIZE

        bruit_i1 = bruit_full_1[:, 0, :]
        bruit_i2 = bruit_full_2[:, N_ref - 1, :]
        i1_obs = i1_true[None, :] + bruit_i1
        i2_obs = i2_true[None, :] + bruit_i2

        for N in N_list:
            idx_phys = idx_phys_dict[N]
            delta_p1 = bruit_full_1[:, idx_phys, :]
            delta_p2 = bruit_full_2[:, idx_phys, :]

            if N == 2:
                bruit_d1 = delta_p1[:, 0, :] - delta_p1[:, -1, :]
                bruit_d2 = delta_p2[:, 0, :] - delta_p2[:, -1, :]
                delta_D1 = (D1_true**2 / (FOCALE * BASELINE_TOT)) * bruit_d1
                delta_D2 = (D2_true**2 / (FOCALE * BASELINE_TOT)) * bruit_d2
                D1_obs = D1_true[None, :] + delta_D1
                D2_obs = D2_true[None, :] + delta_D2
            else:
                S_B2, coef_k = coefs_dict[N]
                D1_obs = mesure_profondeur_H2(D1_true, delta_p1, S_B2, coef_k)
                D2_obs = mesure_profondeur_H2(D2_true, delta_p2, S_B2, coef_k)

            x_c[N][:, n+1], z_c[N][:, n+1], phi_c[N][:, n+1] = calc_odo(
                i1_obs, i2_obs, D1_obs, D2_obs,
                phi_c[N][:, n], x_c[N][:, n], z_c[N][:, n])
            err_c[N][:, n+1] = np.sqrt(
                (x_c[N][:, n+1] - x_d[n+1])**2 +
                (z_c[N][:, n+1] - z_d[n+1])**2)

    return {N: {'x': x_c[N], 'z': z_c[N], 'err': err_c[N]} for N in N_list}

def filepath(n):
    return os.path.join(DOSSIER_SORTIE, n)

iterations = np.arange(N_PAS + 1)
dist_parcourue = iterations * V_MS * DT

# ==============================================================================
# VÉRIFICATION DE L'ANCRAGE PHYSIQUE
# ==============================================================================
print("="*60)
print("VÉRIFICATION : Cohérence positionnelle entre configurations N")
print("="*60)
for N_val in N_LIST:
    idx = np.linspace(0, N_REF - 1, N_val).round().astype(int)
    positions = idx * (BASELINE_TOT / (N_REF - 1))
    print(f"  N = {N_val:>2}  →  indices physiques {list(idx)}")
    print(f"          positions (m) {[f'{p:.4f}' for p in positions]}")
print(f"\n  Profondeur des K = {K_FIXE} points : "
      f"UNIFORME à D = {D_NOMINAL} m (homogénéité des variances)\n")

# ==============================================================================
# EXÉCUTION
# ==============================================================================
print("="*60)
print(f"SIMULATION MUTUALISÉE  (K = {K_FIXE}, MC = {N_MC}, D uniforme = {D_NOMINAL} m)")
print("="*60)

t_start = time.time()
res = run_simulation_mutualisee(K=K_FIXE, n_mc=N_MC, N_list=N_LIST)
print(f"\n -> Simulation complète terminée en {time.time()-t_start:.1f}s")

resultats_N = {}
for N_val in N_LIST:
    err = res[N_val]['err']
    med = np.median(err, axis=0)
    q25 = np.percentile(err, 25, axis=0)
    q75 = np.percentile(err, 75, axis=0)
    idx = np.argsort(err[:, -1])[len(err) // 2]
    resultats_N[N_val] = {"med": med, "q25": q25, "q75": q75,
                           "x": res[N_val]['x'][idx], "z": res[N_val]['z'][idx]}
    print(f"  N = {N_val:>2} : err finale médiane = {med[-1]:.3f} m  "
          f"[Q1 = {q25[-1]:.3f} ; Q3 = {q75[-1]:.3f}]")

# ==============================================================================
# FIGURES
# ==============================================================================
colors_N = {2: "#e41a1c", 4: "#d95f02", 8: "#1b9e77", 16: "#377eb8"}

plt.figure(figsize=(12, 7))
for N_val in N_LIST:
    r = resultats_N[N_val]
    plt.fill_between(dist_parcourue, r["q25"], r["q75"],
                     alpha=0.15, color=colors_N[N_val])
    plt.plot(dist_parcourue, r["med"], lw=2.5, color=colors_N[N_val],
             label=f"N = {N_val} caméras (err finale = {r['med'][-1]:.2f} m)")
plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
plt.xlabel("Distance parcourue [m]")
plt.ylabel(r"Erreur de position $\varepsilon_{pos}(t)$ [m]")
plt.title(f"Évolution temporelle de la dérive — K = {K_FIXE}, "
          f"D uniforme = {D_NOMINAL} m, b_tot = {BASELINE_TOT} m")
plt.tight_layout()
plt.savefig(filepath("Fig_a_Erreur_temporelle_par_N.png"), dpi=150)
plt.close()

err_med = [resultats_N[N]["med"][-1] for N in N_LIST]
err_q25 = [resultats_N[N]["q25"][-1] for N in N_LIST]
err_q75 = [resultats_N[N]["q75"][-1] for N in N_LIST]

plt.figure(figsize=(10, 6))
plt.errorbar(N_LIST, err_med,
             yerr=[np.array(err_med) - np.array(err_q25),
                   np.array(err_q75) - np.array(err_med)],
             fmt='o-', capsize=5, lw=2, markersize=8, color='#377eb8',
             label="Médiane ± IQR [Q1; Q3]")
plt.xscale('log', base=2)
plt.xticks(N_LIST, [str(n) for n in N_LIST])
plt.xlabel("Nombre de caméras N")
plt.ylabel(r"Erreur finale $\varepsilon_{pos}(t_f)$ [m]")
plt.title(f"Critère principal — Erreur finale vs N "
          f"(K = {K_FIXE}, D uniforme = {D_NOMINAL} m)")
plt.grid(True, which='both', linestyle='--', alpha=0.7); plt.legend()
plt.tight_layout()
plt.savefig(filepath("Fig_b_Erreur_finale_vs_N.png"), dpi=150)
plt.close()

plt.figure(figsize=(12, 7))
plt.plot(x_d, z_d, 'k', lw=3, label='Vérité terrain')
for N_val in N_LIST:
    r = resultats_N[N_val]
    plt.plot(r["x"], r["z"], '--', lw=2, color=colors_N[N_val],
             label=f"N = {N_val} caméras")
plt.scatter([x_d[0]], [z_d[0]], color='green', s=100, label='Départ')
plt.scatter([x_d[-1]], [z_d[-1]], color='orange', s=100, label='Arrivée')
plt.grid(True); plt.legend(); plt.xlabel("X [m]"); plt.ylabel("Z [m]")
plt.title(f"Trajectoires estimées en fonction de N (K = {K_FIXE}, D uniforme)")
plt.tight_layout()
plt.savefig(filepath("Fig_c_Trajectoires_par_N.png"), dpi=150)
plt.close()

# ==============================================================================
# COMPARAISON GAIN THÉORIQUE vs OBSERVÉ
# ==============================================================================
print("\n" + "="*60)
print("VÉRIFICATION : Gain théorique H2,MC vs observé (D uniforme)")
print("="*60)

def gain_H2_MC(N):
    return np.sqrt(N * (N + 1) / (6.0 * (N - 1)))

err_N2 = resultats_N[2]["med"][-1]
print(f"\n  Erreur finale médiane à N=2 : {err_N2:.3f} m (référence)\n")
print(f"  {'N':>4} | {'G_th (N/2)':>10} | {'Err théo':>10} | "
      f"{'Err obs':>10} | {'Ratio obs/th':>12}")
print(f"  {'-'*4} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*12}")
for N_val in N_LIST:
    G_th = gain_H2_MC(N_val) / gain_H2_MC(2)
    err_th = err_N2 / G_th
    err_obs = resultats_N[N_val]["med"][-1]
    ratio = err_obs / err_th
    print(f"  {N_val:>4} | {G_th:>10.3f} | {err_th:>10.3f} | "
          f"{err_obs:>10.3f} | {ratio:>12.3f}")

print(f"\n--- TERMINÉ ! Figures générées dans : {DOSSIER_SORTIE} ---")