"""
Simulation d'odometrie multi-cameras par recalage stereoscopique
================================================================
Version v4 - Deux strategies de ponderation

Strategie 1 : w_{i,j} = B^2_{i,j}  (baseline seule, actuel)
Strategie 2 : w_{i,j,p} = B^2_{i,j} * sin^2(2*alpha_{p,i,j})  (composite)

L'angle alpha_{p,i,j} est calcule depuis le centre de la paire (i,j).

CORRECTION Option A : bug d'angle corrige uniquement.
Avant : tan_alpha = -i_ref_1 / f  (constant pour toutes les paires)
Apres : tan_alpha = (X_p - X_cij_val) / D1_ij  (depend de la paire)

Parametres : identiques au code v4 fourni.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from numba import njit, prange
from joblib import Parallel, delayed
from tqdm import tqdm


# ==================================================================
# PARAMETRES PHYSIQUES DU DISPOSITIF
# ==================================================================
DOSSIER_SORTIE = os.getcwd()

# Acquisition
FPS              = 50
DT               = 1.0 / FPS
V_MS             = 10.0 / 3.6
FOCALE           = 0.01
PIXEL_SIZE       = 5e-6
BASELINE_TOT     = 0.20
BRUIT_MAX_PX     = 0.5
DIST_OBJECTIF    = 500.0
EPSILON_ANGLE    = 1e-12

# Trajectoire
INTENSITE_DELTA  = 0.003
Z_DRONE_INIT     = 10.0
THETA_MAX        = np.pi / 2
Z_DRONE_MIN      = 8.0
Z_DRONE_MAX      = 12.0

# Generation des K points
D_NOMINAL        = 10.0
SIGMA_FACADE     = 0.3
RATIO_FACADE     = 0.8
ANGLE_MAX_DEG    = 90.0
ANGLE_MIN_DEG    = 0.0
ZD_MAX_REJECT    = 20.0

# Configurations (identiques au code v3 final)
K_REF            = 1500
N_MC_REF         = 2000
CONFIGS_N        = [2, 4, 8, 16]

# Optimisation
LOT_MC           = 100
N_JOBS_CONFIG    = 4

# Seeds
SEED_TRAJ        = 42
SEED_GEO         = 100
SEED_NOISE       = 200


# ==================================================================
# 1. MODELE 2 + FUSION (STRATEGIE 1 : B^2 SEUL, COMPILE NUMBA)
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
    """
    Strategie 1 : ponderation B^2 seule.
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

            denom = (i_ref_1 * D_hat_1 + i_ref_2 * D_hat_2) ** 2 + \
                    (f ** 2) * (D_hat_2 - D_hat_1) ** 2
            if denom < 1e-18:
                Tx_pts[p] = 0.0
                Tz_pts[p] = 0.0
            else:
                Knum = (D_hat_2 ** 2 - D_hat_1 ** 2) * (f ** 2) + \
                       D_hat_2 ** 2 * i_ref_2 ** 2 - D_hat_1 ** 2 * i_ref_1 ** 2
                Tx_pts[p] = -(1.0 / f) * (D_hat_2 * i_ref_2 +
                                           D_hat_1 * i_ref_1) * Knum / denom
                Tz_pts[p] = -(D_hat_2 - D_hat_1) * Knum / denom

        Tx_final[m] = median_1d(Tx_pts)
        Tz_final[m] = median_1d(Tz_pts)

    return Tx_final, Tz_final


# ==================================================================
# 1bis. FUSION PAIRE PAR PAIRE + POIDS COMPOSITE
# ==================================================================
@njit(cache=True, fastmath=True)
def estimer_lot_numba_pondere(i_pk_1, i_pk_2, delta_1, delta_2,
                                X_ck, X_cij, B_ij_flat, f, N_paires):
    """
    Strategie 2 : ponderation composite w_{i,j,p} = B^2_{i,j} * sin^2(2*alpha_{p,i,j})

    Pour chaque point p et chaque paire (i,j) :
      1. Triangulation : D_{i,j,p} = -f * B_{i,j} / (i_{p,i} - i_{p,j})
      2. Modele 2 : (Tx, Tz)_{i,j,p}
      3. Angle alpha_{p,i,j} depuis le centre de la paire (X_cij_val)
      4. Poids : w = B^2 * sin^2(2*alpha)
    Fusion : moyenne ponderee sur toutes les paires (pour chaque point)
    Puis mediane sur les K points.
    """
    n_mc = delta_1.shape[0]
    N = delta_1.shape[1]
    K = delta_1.shape[2]

    Tx_final = np.zeros(n_mc)
    Tz_final = np.zeros(n_mc)

    Tx_pts = np.zeros(K)
    Tz_pts = np.zeros(K)

    for m in range(n_mc):
        for p in range(K):
            # Reference commune : moyenne des mesures sur toutes les cameras
            i_ref_1 = 0.0
            i_ref_2 = 0.0
            for k in range(N):
                i_ref_1 += i_pk_1[k, p] + delta_1[m, k, p]
                i_ref_2 += i_pk_2[k, p] + delta_2[m, k, p]
            i_ref_1 /= N
            i_ref_2 /= N

            Tx_pond_num = 0.0
            Tz_pond_num = 0.0
            w_total = 0.0

            # Iteration sur toutes les paires (i,j)
            idx_paire = 0
            for i in range(N):
                for j in range(i + 1, N):
                    B_ij = B_ij_flat[idx_paire]
                    X_cij_val = X_cij[idx_paire]
                    idx_paire += 1

                    i_obs_i_1 = i_pk_1[i, p] + delta_1[m, i, p]
                    i_obs_j_1 = i_pk_1[j, p] + delta_1[m, j, p]
                    i_obs_i_2 = i_pk_2[i, p] + delta_2[m, i, p]
                    i_obs_j_2 = i_pk_2[j, p] + delta_2[m, j, p]

                    d1 = i_obs_i_1 - i_obs_j_1
                    d2 = i_obs_i_2 - i_obs_j_2
                    if abs(d1) < 1e-15 or abs(d2) < 1e-15:
                        continue

                    D1_ij = -f * B_ij / d1
                    D2_ij = -f * B_ij / d2

                    i1 = i_ref_1
                    i2 = i_ref_2

                    denom = (i1 * D1_ij + i2 * D2_ij) ** 2 + \
                            (f ** 2) * (D2_ij - D1_ij) ** 2
                    if denom < 1e-18:
                        continue

                    Knum = (D2_ij ** 2 - D1_ij ** 2) * (f ** 2) + \
                           D2_ij ** 2 * i2 ** 2 - D1_ij ** 2 * i1 ** 2
                    Tx_ijp = -(1.0 / f) * (D2_ij * i2 + D1_ij * i1) * Knum / denom
                    Tz_ijp = -(D2_ij - D1_ij) * Knum / denom

                    # === CORRECTION Option A : angle depuis CENTRE DE LA PAIRE ===
                    # Position du point p dans le repere drone (via reference commune)
                    X_p = -i_ref_1 * D1_ij / f
                    # tan(alpha_{p,i,j}) = (X_p - X_cij) / Z_d
                    # DEPEND MAINTENANT de X_cij_val (donc de la paire i,j)
                    if abs(D1_ij) > 1e-9:
                        tan_alpha = (X_p - X_cij_val) / D1_ij
                    else:
                        tan_alpha = 0.0
                    # Identite : sin(2*alpha) = 2*tan / (1 + tan^2)
                    sin_2alpha = 2.0 * tan_alpha / (1.0 + tan_alpha * tan_alpha)
                    sin2_2alpha = sin_2alpha * sin_2alpha

                    # Poids composite : B^2 * sin^2(2*alpha)
                    w_ijp = B_ij * B_ij * sin2_2alpha
                    if w_ijp < 1e-20:
                        continue

                    Tx_pond_num += w_ijp * Tx_ijp
                    Tz_pond_num += w_ijp * Tz_ijp
                    w_total += w_ijp

            if w_total > 1e-20:
                Tx_pts[p] = Tx_pond_num / w_total
                Tz_pts[p] = Tz_pond_num / w_total
            else:
                Tx_pts[p] = 0.0
                Tz_pts[p] = 0.0

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
                x_d[n+1] = x_d[n] + dx
                z_d[n+1] = z_d[n] + dz
                theta_d[n+1] = th_new
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
# 4. PROJECTION SUR N CAMERAS
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


# ==================================================================
# 5. POIDS ALGEBRIQUES ET CENTRES DES PAIRES
# ==================================================================
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


def calculer_paires_composite(N):
    """
    Precalcule les positions des cameras, des centres de paires, et des baselines.
    """
    if N < 2:
        return np.array([0.0]), np.array([]), np.array([])

    b_0 = BASELINE_TOT / (N - 1)
    X_ck = np.arange(N) * b_0 - BASELINE_TOT / 2

    N_paires = N * (N - 1) // 2
    X_cij = np.zeros(N_paires)
    B_ij_flat = np.zeros(N_paires)

    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            X_cij[idx] = 0.5 * (X_ck[i] + X_ck[j])
            B_ij_flat[idx] = X_ck[j] - X_ck[i]
            idx += 1

    return X_ck, X_cij, B_ij_flat


# ==================================================================
# 6. MISE A JOUR ODOMETRIQUE
# ==================================================================
def maj_position(Tx, Tz, x_prev, z_prev, theta_prev):
    theta_inc = np.where(np.abs(Tx) > 1e-12, -2.0 * np.arctan2(Tz, Tx), 0.0)
    theta_new = theta_prev + theta_inc
    vx = -Tx * np.cos(theta_prev) - Tz * np.sin(theta_prev)
    vz = -Tx * np.sin(theta_prev) + Tz * np.cos(theta_prev)
    return x_prev + vx, z_prev + vz, theta_new


# ==================================================================
# 7. SIMULATION POUR UNE CONFIGURATION N (avec choix de strategie)
# ==================================================================
def run_simulation_config(x_d, z_d, theta_d, K, N, n_mc, lot_size,
                           strategie='B2'):
    """
    Simule l'odometrie pour une configuration N.
    strategie : 'B2' pour ponderation baseline seule
                'composite' pour ponderation B^2 * sin^2(2*alpha)
    """
    N_PAS = len(x_d) - 1
    c, sum_B2 = calculer_poids_paires(N)
    X_ck, X_cij, B_ij_flat = calculer_paires_composite(N)
    N_paires = len(B_ij_flat)

    rng_geo = np.random.RandomState(SEED_GEO + N)

    n_lots = (n_mc + lot_size - 1) // lot_size
    n_mc_effectif = n_lots * lot_size

    err_pos = np.zeros((n_mc_effectif, N_PAS + 1))
    err_cum = np.zeros((n_mc_effectif, N_PAS + 1))
    x_est_final = np.zeros((n_mc_effectif, N_PAS + 1))
    z_est_final = np.zeros((n_mc_effectif, N_PAS + 1))
    x_est_final[:, 0] = x_d[0]
    z_est_final[:, 0] = z_d[0]

    x_est_lots = [np.full(lot_size, x_d[0]) for _ in range(n_lots)]
    z_est_lots = [np.full(lot_size, z_d[0]) for _ in range(n_lots)]
    theta_est_lots = [np.full(lot_size, theta_d[0]) for _ in range(n_lots)]
    err_cum_lots = [np.zeros(lot_size) for _ in range(n_lots)]

    pbar = tqdm(total=N_PAS, desc=f"N={N:2d} [{strategie}]", ncols=100)

    for n in range(N_PAS):
        x_n, z_n, th_n = x_d[n], z_d[n], theta_d[n]
        x_np1, z_np1, th_np1 = x_d[n+1], z_d[n+1], theta_d[n+1]

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
                x_est_final[idx_deb:idx_fin, n+1] = x_est_lots[lot]
                z_est_final[idx_deb:idx_fin, n+1] = z_est_lots[lot]
                err_pos[idx_deb:idx_fin, n+1] = err_pos[idx_deb:idx_fin, n]
                err_cum[idx_deb:idx_fin, n+1] = err_cum_lots[lot]
            pbar.update(1)
            continue

        for lot in range(n_lots):
            idx_deb = lot * lot_size
            idx_fin = (lot + 1) * lot_size

            rng_noise = np.random.default_rng(
                SEED_NOISE + N * 100000 + n * 1000 + lot
            )
            delta_1 = rng_noise.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX,
                                         (lot_size, N, K_valid)) * PIXEL_SIZE
            delta_2 = rng_noise.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX,
                                         (lot_size, N, K_valid)) * PIXEL_SIZE

            if strategie == 'B2':
                Tx_lot, Tz_lot = estimer_lot_numba(
                    i_pk_1, i_pk_2, delta_1, delta_2, c, sum_B2, FOCALE
                )
            else:  # composite
                Tx_lot, Tz_lot = estimer_lot_numba_pondere(
                    i_pk_1, i_pk_2, delta_1, delta_2,
                    X_ck, X_cij, B_ij_flat, FOCALE, N_paires
                )

            x_new, z_new, theta_new = maj_position(
                Tx_lot, Tz_lot,
                x_est_lots[lot], z_est_lots[lot], theta_est_lots[lot]
            )

            err_pos[idx_deb:idx_fin, n+1] = np.sqrt(
                (x_new - x_np1) ** 2 + (z_new - z_np1) ** 2
            )

            vx_est = x_new - x_est_lots[lot]
            vz_est = z_new - z_est_lots[lot]
            vx_vt = x_np1 - x_n
            vz_vt = z_np1 - z_n
            norme_delta = np.sqrt((vx_est - vx_vt) ** 2 +
                                  (vz_est - vz_vt) ** 2)
            err_cum_lots[lot] = err_cum_lots[lot] + norme_delta
            err_cum[idx_deb:idx_fin, n+1] = err_cum_lots[lot]

            x_est_final[idx_deb:idx_fin, n+1] = x_new
            z_est_final[idx_deb:idx_fin, n+1] = z_new

            x_est_lots[lot] = x_new
            z_est_lots[lot] = z_new
            theta_est_lots[lot] = theta_new

        pbar.update(1)

    pbar.close()

    return {
        'N': N,
        'strategie': strategie,
        'err_pos': err_pos[:n_mc, :],
        'err_cum': err_cum[:n_mc, :],
        'x_est': x_est_final[:n_mc, :],
        'z_est': z_est_final[:n_mc, :]
    }


# ==================================================================
# 8. UTILITAIRES
# ==================================================================
def filepath(name):
    return os.path.join(DOSSIER_SORTIE, name)


def stats_MC(err):
    mediane = np.median(err, axis=0)
    q1 = np.percentile(err, 25, axis=0)
    q3 = np.percentile(err, 75, axis=0)
    return mediane, q1, q3


def selectionner_traj_representative(err_pos):
    med_final = np.median(err_pos[:, -1])
    idx = np.argmin(np.abs(err_pos[:, -1] - med_final))
    return idx


# ==================================================================
# 9. EXECUTION PRINCIPALE
# ==================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SIMULATION v4 - COMPARAISON DES STRATEGIES DE PONDERATION")
    print("CORRECTION Option A : bug d'angle corrige")
    print("=" * 70)
    print(f"Parametres :")
    print(f"  FPS = {FPS} Hz, V = {V_MS:.3f} m/s")
    print(f"  Distance = {DIST_OBJECTIF} m")
    print(f"  ANGLE = [{ANGLE_MIN_DEG}, {ANGLE_MAX_DEG}] deg")
    print(f"  K = {K_REF}, N_mc = {N_MC_REF}, Configs N = {CONFIGS_N}")
    print(f"  Strategies : (1) B^2 seule, (2) B^2 * sin^2(2*alpha_p)")

    # ------------------------------------------------------------------
    # PRE-COMPILATION NUMBA
    # ------------------------------------------------------------------
    print("\nPre-compilation Numba (peut prendre 30-60s)...")
    t0 = time.time()
    N_test = 2
    c_t, sum_B2_t = calculer_poids_paires(N_test)
    X_ck_t, X_cij_t, B_ij_t = calculer_paires_composite(N_test)
    N_p_t = len(B_ij_t)
    _ = estimer_lot_numba(np.zeros((N_test, 5)), np.zeros((N_test, 5)),
                          np.zeros((3, N_test, 5)), np.zeros((3, N_test, 5)),
                          c_t, sum_B2_t, FOCALE)
    _ = estimer_lot_numba_pondere(np.zeros((N_test, 5)), np.zeros((N_test, 5)),
                                    np.zeros((3, N_test, 5)),
                                    np.zeros((3, N_test, 5)),
                                    X_ck_t, X_cij_t, B_ij_t, FOCALE, N_p_t)
    print(f"  Compilation terminee en {time.time() - t0:.1f} s")

    # ------------------------------------------------------------------
    # PHASE 0 : Trajectoire verite terrain
    # ------------------------------------------------------------------

    print("\nGeneration trajectoire verite terrain...")
    t0 = time.time()
    x_d, z_d, theta_d = generer_trajectoire_VT()
    N_PAS = len(x_d) - 1
    dist_curviligne = np.zeros(N_PAS + 1)
    for n in range(N_PAS):
        dist_curviligne[n+1] = dist_curviligne[n] + \
            np.sqrt((x_d[n+1] - x_d[n]) ** 2 + (z_d[n+1] - z_d[n]) ** 2)
    print(f"  Trajectoire : {N_PAS} pas en {time.time() - t0:.1f} s")

    # ------------------------------------------------------------------
    # PHASE 1 : Strategie 1 (B^2 seule)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STRATEGIE 1 : PONDERATION B^2 SEULE")
    print("=" * 70)
    t_strat1 = time.time()
    results_B2 = Parallel(n_jobs=N_JOBS_CONFIG, backend='threading')(
        delayed(run_simulation_config)(
            x_d, z_d, theta_d, K_REF, N_val, N_MC_REF, LOT_MC, 'B2'
        )
        for N_val in CONFIGS_N
    )
    resultats_B2 = {r['N']: r for r in results_B2}
    print(f"\nTemps strategie B^2 : {(time.time() - t_strat1)/60:.1f} min")

    # ------------------------------------------------------------------
    # PHASE 2 : Strategie 2 (B^2 * sin^2(2*alpha))
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STRATEGIE 2 : PONDERATION COMPOSITE B^2 * sin^2(2*alpha_p)")
    print("=" * 70)
    t_strat2 = time.time()
    results_comp = Parallel(n_jobs=N_JOBS_CONFIG, backend='threading')(
        delayed(run_simulation_config)(
            x_d, z_d, theta_d, K_REF, N_val, N_MC_REF, LOT_MC, 'composite'
        )
        for N_val in CONFIGS_N
    )
    resultats_comp = {r['N']: r for r in results_comp}
    print(f"\nTemps strategie composite : {(time.time() - t_strat2)/60:.1f} min")

    # ------------------------------------------------------------------
    # PHASE 3 : Statistiques comparatives
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STATISTIQUES COMPARATIVES (erreur cumulee finale)")
    print("=" * 70)
    print(f"\n{'N':>4} | {'B^2 med':>10} | {'B^2 Q1':>10} | {'B^2 Q3':>10} "
          f"| {'Comp med':>10} | {'Comp Q1':>10} | {'Comp Q3':>10} "
          f"| {'Gain %':>8}")
    print("-" * 100)
    for N_val in CONFIGS_N:
        r_b = resultats_B2[N_val]
        r_c = resultats_comp[N_val]
        med_b, q1_b, q3_b = stats_MC(r_b['err_cum'][:, -1:].flatten())
        med_c, q1_c, q3_c = stats_MC(r_c['err_cum'][:, -1:].flatten())
        gain = (med_b - med_c) / med_b * 100 if med_b > 0 else 0
        print(f"{N_val:>4} | {med_b:>10.4f} | {q1_b:>10.4f} | {q3_b:>10.4f} "
              f"| {med_c:>10.4f} | {q1_c:>10.4f} | {q3_c:>10.4f} "
              f"| {gain:>+7.1f}%")

    # ------------------------------------------------------------------
    # PHASE 4 : Figures
    # ------------------------------------------------------------------
    print("\nGeneration des 6 figures...")

    couleurs_N = {2: '#e41a1c', 4: '#d95f02', 8: '#1b9e77', 16: '#377eb8'}

    def tracer_figure(resultats, ylabel, titre_suffixe, nom_fichier,
                       metrique='err_pos'):
        fig, ax = plt.subplots(figsize=(12, 6))
        for N_val in CONFIGS_N:
            r = resultats[N_val]
            med, q1, q3 = stats_MC(r[metrique])
            ax.plot(dist_curviligne, med, color=couleurs_N[N_val], lw=2,
                    label=f'N = {N_val} (mediane finale = {med[-1]:.2f} m)')
            ax.fill_between(dist_curviligne, q1, q3,
                             color=couleurs_N[N_val], alpha=0.2)
        ax.set_xlabel('Distance parcourue [m]')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{titre_suffixe} (mediane + Q1/Q3)')
        ax.legend()
        ax.grid(True, ls='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(filepath(nom_fichier), dpi=120)
        plt.close()

    def tracer_trajectoires(resultats, titre_suffixe, nom_fichier):
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(x_d, z_d, 'k', lw=3, label=r'Verite terrain')
        for N_val in CONFIGS_N:
            r = resultats[N_val]
            idx = selectionner_traj_representative(r['err_pos'])
            ax.plot(r['x_est'][idx, :], r['z_est'][idx, :],
                    '--', lw=2, color=couleurs_N[N_val],
                    label=fr'Trajectoire estimée (N = {N_val})')
        ax.set_xlabel(r'Coordonnee longitudinale $X_m$ [m]')
        ax.set_ylabel(r'Distance au mur $Z_m$ [m]')
        ax.set_title(f'Superposition des trajectoires - {titre_suffixe}')
        ax.legend()
        ax.grid(True, ls='--', alpha=0.7)
        ax.set_ylim(8.0, 12.0)
        plt.tight_layout()
        plt.savefig(filepath(nom_fichier), dpi=120)
        plt.close()

    # Fig 1-3 : Strategie B^2
    tracer_figure(resultats_B2, r'$\varepsilon_{pos}$ [m]',
                   'Erreur de position ($B^2$ seule)',
                   'Fig1_erreur_position_B2.png', metrique='err_pos')
    tracer_figure(resultats_B2, r'$\varepsilon_{pos}$ [m]',
                   'Erreur cumulee ($B^2$ seule)',
                   'Fig2_erreur_cumulee_B2.png', metrique='err_cum')
    tracer_trajectoires(resultats_B2, r'Ponderation $B^2$ seule',
                         'Fig3_trajectoires_B2.png')

    # Fig 4-6 : Strategie composite
    tracer_figure(resultats_comp, r'$\varepsilon_{pos}$ [m]',
                   r'Erreur de position ($B^2 \sin^2(2\alpha_p)$)',
                   'Fig4_erreur_position_composite.png', metrique='err_pos')
    tracer_figure(resultats_comp, r'$\varepsilon_{pos}$ [m]',
                   r'Erreur cumulee ($B^2 \sin^2(2\alpha_p)$)',
                   'Fig5_erreur_cumulee_composite.png', metrique='err_cum')
    tracer_trajectoires(resultats_comp,
                         r'Ponderation composite $B^2 \sin^2(2\alpha_p)$',
                         'Fig6_trajectoires_composite.png')

    print(f"\n=== SIMULATION TERMINEE ===")
    print(f"6 figures generees dans : {DOSSIER_SORTIE}")