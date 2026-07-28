"""
Simulation d'odometrie multi-cameras par recalage stereoscopique
================================================================
Fichier dédié : Calcul et affichage UNIQUEMENT de la superposition 
des trajectoires pour la stratégie w_{i,j} = B^2_{i,j} (baseline seule).
Correction : Ajustement dynamique des axes pour afficher toutes les trajectoires.
"""

import os
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from tqdm import tqdm

# ==================================================================
# PARAMETRES PHYSIQUES DU DISPOSITIF
# ==================================================================
DOSSIER_SORTIE = os.getcwd()

# Acquisition
FPS = 50
DT = 1.0 / FPS
V_MS = 10.0 / 3.6
FOCALE = 0.01
PIXEL_SIZE = 5e-6
BASELINE_TOT = 0.20
BRUIT_MAX_PX = 0.5
DIST_OBJECTIF = 500.0
EPSILON_ANGLE = 1e-12

# Trajectoire
INTENSITE_DELTA = 0.003
Z_DRONE_INIT = 10.0
THETA_MAX = np.pi / 2
Z_DRONE_MIN = 8.0
Z_DRONE_MAX = 12.0

# Generation des K points
D_NOMINAL = 10.0
SIGMA_FACADE = 0.3
RATIO_FACADE = 0.8
ANGLE_MAX_DEG = 70.0
ANGLE_MIN_DEG = 5.0
ZD_MAX_REJECT = 20.0

# Configurations
K_REF = 1000
N_MC_REF = 150
CONFIGS_N = [2, 4, 8, 16]

# Optimisation
LOT_MC = 100
N_JOBS_CONFIG = 4

# Seeds
SEED_TRAJ = 42
SEED_GEO = 100
SEED_NOISE = 200


# ==================================================================
# 1. MODELE ET FUSION (STRATEGIE B^2 SEULE, COMPILE NUMBA)
# ==================================================================
@njit(cache=True, fastmath=True)
def median_1d(arr):
    """Mediane manuelle compatible Numba."""
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    if n % 2 == 1:
        return sorted_arr[n // 2]
    else:
        return 0.5 * (sorted_arr[n // 2 - 1] + sorted_arr[n // 2])


@njit(cache=True, fastmath=True)
def estimer_lot_numba(i_pk_1, i_pk_2, delta_1, delta_2, c, sum_B2, f):
    """Strategie : ponderation B^2 seule.

    Fusion GM lineaire : 1/D_hat = Sum_k c_k * i_k / (-f * sum_B2)
    Modele 2 applique une seule fois par point.
    """
    n_mc = delta_1.shape[0]
    N = delta_1.shape[1]
    K = delta_1.shape[2]

    Tx_final = np.zeros(n_mc)
    Tz_final = np.zeros(n_mc)

    for m in range(n_mc):
        Tx_pts = np.zeros(K)
        Tz_pts = np.zeros(K)

        for p in range(K):
            i_ref_1 = 0.0
            i_ref_2 = 0.0
            num_1 = 0.0
            num_2 = 0.0

            for k in range(N):
                i_obs_1_kp = i_pk_1[k, p] + delta_1[m, k, p]
                i_obs_2_kp = i_pk_2[k, p] + delta_2[m, k, p]
                i_ref_1 += i_obs_1_kp
                i_ref_2 += i_obs_2_kp
                num_1 += c[k] * i_obs_1_kp
                num_2 += c[k] * i_obs_2_kp

            i_ref_1 /= N
            i_ref_2 /= N

            inv_D_1 = num_1 / (-f * sum_B2)
            inv_D_2 = num_2 / (-f * sum_B2)

            if abs(inv_D_1) > 1e-15:
                D_hat_1 = 1.0 / inv_D_1
            else:
                D_hat_1 = ZD_MAX_REJECT

            if abs(inv_D_2) > 1e-15:
                D_hat_2 = 1.0 / inv_D_2
            else:
                D_hat_2 = ZD_MAX_REJECT

            denom = (i_ref_1 * D_hat_1 + i_ref_2 * D_hat_2) ** 2 + (f**2) * (
                D_hat_2 - D_hat_1
            ) ** 2
            if denom < 1e-18:
                Tx_pts[p] = 0.0
                Tz_pts[p] = 0.0
            else:
                Knum = (D_hat_2**2 - D_hat_1**2) * (f**2) + D_hat_2**2 * i_ref_2**2 - D_hat_1**2 * i_ref_1**2
                Tx_pts[p] = -(1.0 / f) * (D_hat_2 * i_ref_2 + D_hat_1 * i_ref_1) * Knum / denom
                Tz_pts[p] = -(D_hat_2 - D_hat_1) * Knum / denom

        Tx_final[m] = median_1d(Tx_pts)
        Tz_final[m] = median_1d(Tz_pts)

    return Tx_final, Tz_final


# ==================================================================
# 2. GENERATION DE LA TRAJECTOIRE VERITE TERRAIN
# ==================================================================
def generer_trajectoire_VT(seed=SEED_TRAJ):
    rng = np.random.RandomState(seed)
    n_pas_max = int(DIST_OBJECTIF / (V_MS * DT)) + 100

    x_d = np.zeros(n_pas_max)
    z_d = np.zeros(n_pas_max)
    theta_d = np.zeros(n_pas_max)
    z_d[0] = Z_DRONE_INIT

    n = 0
    while x_d[n] < DIST_OBJECTIF and n < n_pas_max - 1:
        valide = False
        while not valide:
            th_n = theta_d[n]
            dtheta = rng.uniform(-INTENSITE_DELTA, INTENSITE_DELTA)
            if z_d[n] > Z_DRONE_MAX and th_n > 0:
                dtheta = rng.uniform(-INTENSITE_DELTA, 0)
            if z_d[n] < Z_DRONE_MIN and th_n < 0:
                dtheta = rng.uniform(0, INTENSITE_DELTA)

            th_new = th_n + dtheta

            if abs(dtheta) < EPSILON_ANGLE:
                dx = V_MS * DT * np.cos(th_n)
                dz = V_MS * DT * np.sin(th_n)
            else:
                R = (V_MS * DT) / dtheta
                dx = R * (np.sin(th_new) - np.sin(th_n))
                dz = R * (np.cos(th_n) - np.cos(th_new))

            if dx > 0:
                x_d[n + 1] = x_d[n] + dx
                z_d[n + 1] = z_d[n] + dz
                theta_d[n + 1] = th_new
                valide = True

        n += 1

    n_pas_final = n + 1
    return x_d[:n_pas_final], z_d[:n_pas_final], theta_d[:n_pas_final]


# ==================================================================
# 3. GENERATION DES K POINTS D'INTERET
# ==================================================================
def genere_profondeur(K, rng):
    if K == 1:
        return np.array([D_NOMINAL])

    K_facade = int(K * RATIO_FACADE)
    K_saillie = K - K_facade

    z_facade = rng.normal(0, SIGMA_FACADE, K_facade)
    masque = (z_facade < -2.0) | (z_facade > 2.0)
    while masque.any():
        z_facade[masque] = rng.normal(0, SIGMA_FACADE, masque.sum())
        masque = (z_facade < -2.0) | (z_facade > 2.0)

    z_saillie = rng.uniform(-2.0, 2.0, K_saillie)
    z_wall = np.concatenate([z_facade, z_saillie])
    rng.shuffle(z_wall)

    return D_NOMINAL + z_wall


def generer_K_points(x_drone, z_drone, theta, K, rng):
    alpha_max = np.deg2rad(ANGLE_MAX_DEG)
    alpha_min = np.deg2rad(ANGLE_MIN_DEG)

    angles = rng.uniform(-alpha_max, alpha_max, K)
    masque = np.abs(angles) < alpha_min
    while masque.any():
        angles[masque] = rng.uniform(-alpha_max, alpha_max, masque.sum())
        masque = np.abs(angles) < alpha_min

    D_world = genere_profondeur(K, rng)
    Xw = x_drone + D_world * np.cos(theta + angles)
    Zw = z_drone + D_world * np.sin(theta + angles)
    return Xw, Zw


# ==================================================================
# 4. PROJECTION SUR N CAMERAS ET POIDS ALGEBRIQUES
# ==================================================================
def projeter_N_cameras(Xw, Zw, x_drone, z_drone, theta, N, f=FOCALE):
    dx = Xw - x_drone
    dz = Zw - z_drone
    Xc = dx * np.cos(theta) + dz * np.sin(theta)
    D = dx * np.sin(theta) - dz * np.cos(theta)

    if N > 1:
        b_0 = BASELINE_TOT / (N - 1)
        X_ck = np.arange(N) * b_0 - BASELINE_TOT / 2
    else:
        X_ck = np.array([0.0])

    i_pk = -f * (Xc[None, :] - X_ck[:, None]) / D[None, :]
    return i_pk, Xc, D


def calculer_poids_paires(N):
    if N < 2:
        return np.zeros(N), 0.0

    b_0 = BASELINE_TOT / (N - 1)
    X_ck = np.arange(N) * b_0 - BASELINE_TOT / 2

    c = np.zeros(N)
    sum_B2 = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            B_ij = X_ck[j] - X_ck[i]
            c[i] += B_ij
            c[j] -= B_ij
            sum_B2 += B_ij * B_ij
    return c, sum_B2


# ==================================================================
# 5. MISE A JOUR ODOMETRIQUE
# ==================================================================
def maj_position(Tx, Tz, x_prev, z_prev, theta_prev):
    theta_inc = np.where(np.abs(Tx) > 1e-12, -2.0 * np.arctan2(Tz, Tx), 0.0)
    theta_new = theta_prev + theta_inc
    vx = -Tx * np.cos(theta_prev) - Tz * np.sin(theta_prev)
    vz = -Tx * np.sin(theta_prev) + Tz * np.cos(theta_prev)
    return x_prev + vx, z_prev + vz, theta_new


# ==================================================================
# 6. SIMULATION POUR UNE CONFIGURATION N (B^2 UNIQUEMENT)
# ==================================================================
def run_simulation_config_B2(x_d, z_d, theta_d, K, N, n_mc, lot_size):
    """Simule l'odometrie pour une configuration N, uniquement avec w_{i,j} = B^2."""
    N_PAS = len(x_d) - 1
    c, sum_B2 = calculer_poids_paires(N)
    rng_geo = np.random.RandomState(SEED_GEO + N)

    n_lots = (n_mc + lot_size - 1) // lot_size
    n_mc_effectif = n_lots * lot_size

    err_pos = np.zeros((n_mc_effectif, N_PAS + 1))
    x_est_final = np.zeros((n_mc_effectif, N_PAS + 1))
    z_est_final = np.zeros((n_mc_effectif, N_PAS + 1))
    x_est_final[:, 0] = x_d[0]
    z_est_final[:, 0] = z_d[0]

    x_est_lots = [np.full(lot_size, x_d[0]) for _ in range(n_lots)]
    z_est_lots = [np.full(lot_size, z_d[0]) for _ in range(n_lots)]
    theta_est_lots = [np.full(lot_size, theta_d[0]) for _ in range(n_lots)]

    pbar = tqdm(total=N_PAS, desc=f"N={N:2d} [B^2]", ncols=100)

    for n in range(N_PAS):
        x_n, z_n, th_n = x_d[n], z_d[n], theta_d[n]
        x_np1, z_np1, th_np1 = x_d[n + 1], z_d[n + 1], theta_d[n + 1]

        Xw, Zw = generer_K_points(x_n, z_n, th_n, K, rng_geo)
        i_pk_1, _, D1 = projeter_N_cameras(Xw, Zw, x_n, z_n, th_n, N)
        i_pk_2, _, D2 = projeter_N_cameras(Xw, Zw, x_np1, z_np1, th_np1, N)

        valid = (np.abs(D1) > 0.5) & (np.abs(D2) > 0.5)
        if not valid.all():
            i_pk_1 = i_pk_1[:, valid]
            i_pk_2 = i_pk_2[:, valid]
        K_valid = i_pk_1.shape[1]

        if K_valid < 1:
            for lot in range(n_lots):
                idx_deb = lot * lot_size
                idx_fin = (lot + 1) * lot_size
                x_est_final[idx_deb:idx_fin, n + 1] = x_est_lots[lot]
                z_est_final[idx_deb:idx_fin, n + 1] = z_est_lots[lot]
                err_pos[idx_deb:idx_fin, n + 1] = err_pos[idx_deb:idx_fin, n]
            pbar.update(1)
            continue

        for lot in range(n_lots):
            idx_deb = lot * lot_size
            idx_fin = (lot + 1) * lot_size

            rng_noise = np.random.default_rng(
                SEED_NOISE + N * 100000 + n * 1000 + lot
            )
            delta_1 = (
                rng_noise.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX, (lot_size, N, K_valid))
                * PIXEL_SIZE
            )
            delta_2 = (
                rng_noise.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX, (lot_size, N, K_valid))
                * PIXEL_SIZE
            )

            Tx_lot, Tz_lot = estimer_lot_numba(
                i_pk_1, i_pk_2, delta_1, delta_2, c, sum_B2, FOCALE
            )

            x_new, z_new, theta_new = maj_position(
                Tx_lot,
                Tz_lot,
                x_est_lots[lot],
                z_est_lots[lot],
                theta_est_lots[lot],
            )

            err_pos[idx_deb:idx_fin, n + 1] = np.sqrt(
                (x_new - x_np1) ** 2 + (z_new - z_np1) ** 2
            )

            x_est_final[idx_deb:idx_fin, n + 1] = x_new
            z_est_final[idx_deb:idx_fin, n + 1] = z_new

            x_est_lots[lot] = x_new
            z_est_lots[lot] = z_new
            theta_est_lots[lot] = theta_new

        pbar.update(1)

    pbar.close()

    return {
        "N": N,
        "err_pos": err_pos[:n_mc, :],
        "x_est": x_est_final[:n_mc, :],
        "z_est": z_est_final[:n_mc, :],
    }


# ==================================================================
# 7. UTILITAIRES
# ==================================================================
def filepath(name):
    return os.path.join(DOSSIER_SORTIE, name)


def selectionner_traj_representative(err_pos):
    """Sélectionne l'essai Monte-Carlo le plus proche de la médiane finale."""
    med_final = np.median(err_pos[:, -1])
    idx = np.argmin(np.abs(err_pos[:, -1] - med_final))
    return idx


# ==================================================================
# 8. EXECUTION PRINCIPALE
# ==================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SIMULATION - SUPERPOSITION DES TRAJECTOIRES (STRATEGIE B^2 SEULE)")
    print("=" * 70)
    print(f"Parametres : FPS = {FPS} Hz, Distance = {DIST_OBJECTIF} m")
    print(f"Configs N = {CONFIGS_N}, N_mc = {N_MC_REF}, K = {K_REF}")

    # ------------------------------------------------------------------
    # PRE-COMPILATION NUMBA
    # ------------------------------------------------------------------
    print("\nPre-compilation Numba...")
    t0 = time.time()
    N_test = 2
    c_t, sum_B2_t = calculer_poids_paires(N_test)
    _ = estimer_lot_numba(
        np.zeros((N_test, 5)),
        np.zeros((N_test, 5)),
        np.zeros((3, N_test, 5)),
        np.zeros((3, N_test, 5)),
        c_t,
        sum_B2_t,
        FOCALE,
    )
    print(f"  Compilation terminee en {time.time() - t0:.1f} s")

    # ------------------------------------------------------------------
    # PHASE 0 : Trajectoire verite terrain
    # ------------------------------------------------------------------
    print("\nGeneration trajectoire verite terrain...")
    x_d, z_d, theta_d = generer_trajectoire_VT()
    print(f"  Trajectoire générée ({len(x_d)-1} pas).")

    # ------------------------------------------------------------------
    # PHASE 1 : Simulation pour B^2 seule
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LANCEMENT DES SIMULATIONS (Ponderation B^2 seule)")
    print("=" * 70)
    t_start = time.time()
    results_B2 = Parallel(n_jobs=N_JOBS_CONFIG, backend="threading")(
        delayed(run_simulation_config_B2)(
            x_d, z_d, theta_d, K_REF, N_val, N_MC_REF, LOT_MC
        )
        for N_val in CONFIGS_N
    )
    resultats_B2 = {r["N"]: r for r in results_B2}
    print(
        f"\nTemps total de simulation : {(time.time() - t_start)/60:.1f} min"
    )

    # ------------------------------------------------------------------
    # PHASE 2 : Generation de la figure unique avec axes dynamiques
    # ------------------------------------------------------------------
    print("\nGénération de la figure de superposition des trajectoires...")
    couleurs_N = {2: "#e41a1c", 4: "#d95f02", 8: "#1b9e77", 16: "#377eb8"}
    nom_fichier = "Fig_trajectoires_B2_uniquement.png"

    fig, ax = plt.subplots(figsize=(14, 6))

    # Trace de la vérité terrain et initialisation des limites Z
    ax.plot(x_d, z_d, "k", lw=3, label=r"Verite terrain")
    z_min_list = [np.min(z_d)]
    z_max_list = [np.max(z_d)]

    # Trace des trajectoires estimées et calcul des bornes atteintes
    for N_val in CONFIGS_N:
        r = resultats_B2[N_val]
        idx = selectionner_traj_representative(r["err_pos"])
        z_traj = r["z_est"][idx, :]
        ax.plot(
            r["x_est"][idx, :],
            z_traj,
            "--",
            lw=2,
            color=couleurs_N[N_val],
            label=fr"Trajectoire estimée (N = {N_val})",
        )
        z_min_list.append(np.min(z_traj))
        z_max_list.append(np.max(z_traj))

    # AJUSTEMENT DYNAMIQUE DE L'AXE VERTICAL (avec marge de sécurité de 10 %)
    z_min = min(z_min_list)
    z_max = max(z_max_list)
    marge = max((z_max - z_min) * 0.1, 1.0)  # Au moins 1 mètre de marge
    ax.set_ylim(z_min - marge, z_max + marge)

    ax.set_xlabel(r"Coordonnee longitudinale $X_m$ [m]", fontsize=12)
    ax.set_ylabel(r"Distance au mur $Z_m$ [m]", fontsize=12)
    ax.set_title(
        r"Superposition des trajectoires - Ponderation $B^2$ seule",
        fontsize=14,
    )
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, ls="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(filepath(nom_fichier), dpi=150)
    plt.close()

    print(f"\n=== TERMINÉ AVEC SUCCÈS ===")
    print(f"Figure exportée sous : {filepath(nom_fichier)}")