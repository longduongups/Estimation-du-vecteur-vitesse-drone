"""
Simulation d'odometrie multi-cameras avec objets mobiles
=========================================================
Comparaison de 3 methodes face au dynamisme (tau_mob) :
  1. Sans filtre (Mediane brute de la strategie B^2)
  2. Seuillage adaptatif géométrique (MAD angulaire + Médiane sur inliers)
  3. RANSAC 2D optimisé (Consensus 2 points + Médiane sur inliers)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm

# ==================================================================
# PARAMETRES PHYSIQUES DU DISPOSITIF
# ==================================================================
DOSSIER_SORTIE = os.getcwd()

# Acquisition (identiques au code v4)
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
Z_DRONE_MIN      = 9.0
Z_DRONE_MAX      = 11.0

# Generation des K points
D_NOMINAL        = 10.0
SIGMA_FACADE     = 0.3
RATIO_FACADE     = 0.8
ANGLE_MAX_DEG    = 50.0
ANGLE_MIN_DEG    = 10.0
ZD_MAX_REJECT    = 20.0

# === PARAMETRES OBJETS MOBILES ===
V_MAX_MOB        = 15.0            # m/s (piétons, cyclistes, vehicules urbains)
TAU_MOB_VALEURS  = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]  # 6 valeurs

# Configuration
K_REF            = 1000
N_MC_REF         = 50
N_FIXE           = 4

# Optimisation
LOT_MC           = 100
N_JOBS           = 4

# Seeds
SEED_TRAJ        = 42
SEED_GEO         = 100
SEED_NOISE       = 200
SEED_MOB         = 300


# ==================================================================
# 1. FONCTIONS NUMBA - 3 METHODES D'ESTIMATION (TURBO-OPTIMISEES)
# ==================================================================
@njit(cache=True, fastmath=True)
def median_1d(arr):
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    if n % 2 == 1:
        return sorted_arr[n // 2]
    else:
        return 0.5 * (sorted_arr[n // 2 - 1] + sorted_arr[n // 2])

@njit(cache=True, fastmath=True)
def estimer_lot_3methodes_numba(i_pk_1, i_pk_2, n_mc, bruit_px, c, sum_B2, f, seed_base, tau_mob):
    N = i_pk_1.shape[0]
    K = i_pk_1.shape[1]

    Tx_brut = np.zeros(n_mc)
    Tz_brut = np.zeros(n_mc)
    Tx_seuil = np.zeros(n_mc)
    Tz_seuil = np.zeros(n_mc)
    Tx_ransac = np.zeros(n_mc)
    Tz_ransac = np.zeros(n_mc)

    for m in range(n_mc):
        # Generation du bruit directement en registre CPU (Zero allocation RAM)
        np.random.seed(seed_base + m)
        Tx_pts = np.zeros(K)
        Tz_pts = np.zeros(K)
        D_pts  = np.zeros(K)

        for p in range(K):
            i_ref_1 = 0.0
            i_ref_2 = 0.0
            num_1 = 0.0
            num_2 = 0.0

            for k in range(N):
                i_obs_1_kp = i_pk_1[k, p] + np.random.uniform(-bruit_px, bruit_px)
                i_obs_2_kp = i_pk_2[k, p] + np.random.uniform(-bruit_px, bruit_px)
                i_ref_1 += i_obs_1_kp
                i_ref_2 += i_obs_2_kp
                num_1 += c[k] * i_obs_1_kp
                num_2 += c[k] * i_obs_2_kp

            i_ref_1 /= N
            i_ref_2 /= N

            inv_D_1 = num_1 / (-f * sum_B2)
            inv_D_2 = num_2 / (-f * sum_B2)
            D_hat_1 = 1.0 / inv_D_1 if abs(inv_D_1) > 1e-15 else ZD_MAX_REJECT
            D_hat_2 = 1.0 / inv_D_2 if abs(inv_D_2) > 1e-15 else ZD_MAX_REJECT

            D_pts[p] = 0.5 * (D_hat_1 + D_hat_2)

            denom = (i_ref_1 * D_hat_1 + i_ref_2 * D_hat_2) ** 2 + (f ** 2) * (D_hat_2 - D_hat_1) ** 2
            if denom < 1e-18:
                Tx_pts[p] = 0.0
                Tz_pts[p] = 0.0
            else:
                Knum = (D_hat_2 ** 2 - D_hat_1 ** 2) * (f ** 2) + \
                       D_hat_2 ** 2 * i_ref_2 ** 2 - D_hat_1 ** 2 * i_ref_1 ** 2
                Tx_pts[p] = -(1.0 / f) * (D_hat_2 * i_ref_2 + D_hat_1 * i_ref_1) * Knum / denom
                Tz_pts[p] = -(D_hat_2 - D_hat_1) * Knum / denom

        # -------------------------------------------------------------
        # METHODE 1 : SANS FILTRE (Médiane brute)
        # -------------------------------------------------------------
        tx_med = median_1d(Tx_pts)
        tz_med = median_1d(Tz_pts)
        Tx_brut[m] = tx_med
        Tz_brut[m] = tz_med

        if tau_mob <= 0.0:
            Tx_seuil[m] = tx_med
            Tz_seuil[m] = tz_med
            Tx_ransac[m] = tx_med
            Tz_ransac[m] = tz_med
            continue

        # -------------------------------------------------------------
        # METHODE 2 : SEUILLAGE GEOMETRIQUE (MAD angulaire)
        # -------------------------------------------------------------
        residus_norm_sq = np.zeros(K)
        for p in range(K):
            dist_sq_D = D_pts[p] * D_pts[p]
            if dist_sq_D < 1.0:
                dist_sq_D = 1.0
            residus_norm_sq[p] = np.sqrt((Tx_pts[p] - tx_med)**2 + (Tz_pts[p] - tz_med)**2) / dist_sq_D
        
        med_res = median_1d(residus_norm_sq)
        seuil_norm = 3.0 * (1.4826 * med_res)
        if seuil_norm < 0.0005:
            seuil_norm = 0.0005
            
        inliers_tx = np.zeros(K)
        inliers_tz = np.zeros(K)
        count = 0
        for p in range(K):
            if residus_norm_sq[p] < seuil_norm:
                inliers_tx[count] = Tx_pts[p]
                inliers_tz[count] = Tz_pts[p]
                count += 1
                
        if count > 0:
            Tx_seuil[m] = median_1d(inliers_tx[:count])
            Tz_seuil[m] = median_1d(inliers_tz[:count])
        else:
            Tx_seuil[m] = tx_med
            Tz_seuil[m] = tz_med

        # -------------------------------------------------------------
        # METHODE 3 : RANSAC 2D TURBO-OPTIMISE
        # -------------------------------------------------------------
        best_inliers = 0  # CORRECTION : Initialiser à 0 et non -1 !
        best_tx_hyp = tx_med
        best_tz_hyp = tz_med
        seuil_ransac_sq = 0.05 * 0.05
        
        n_iterations = 35
        
        for _ in range(n_iterations):
            idx0 = np.random.randint(0, K)
            idx1 = np.random.randint(0, K - 1)
            if idx1 >= idx0:
                idx1 += 1
            
            tx_hyp = 0.5 * (Tx_pts[idx0] + Tx_pts[idx1])
            tz_hyp = 0.5 * (Tz_pts[idx0] + Tz_pts[idx1])
            
            inliers_count = 0
            for p in range(K):
                dist_sq = (Tx_pts[p] - tx_hyp)**2 + (Tz_pts[p] - tz_hyp)**2
                if dist_sq < seuil_ransac_sq:
                    inliers_count += 1
            
            if inliers_count > best_inliers:
                best_inliers = inliers_count
                best_tx_hyp = tx_hyp
                best_tz_hyp = tz_hyp
                
        # CORRECTION : On ne crée le tableau que si des inliers existent (évite le crash à la compilation)
        if best_inliers > 0:
            temp_tx = np.zeros(best_inliers)
            temp_tz = np.zeros(best_inliers)
            c_idx = 0
            for p in range(K):
                dist_sq = (Tx_pts[p] - best_tx_hyp)**2 + (Tz_pts[p] - best_tz_hyp)**2
                if dist_sq < seuil_ransac_sq:
                    temp_tx[c_idx] = Tx_pts[p]
                    temp_tz[c_idx] = Tz_pts[p]
                    c_idx += 1
                    
            Tx_ransac[m] = median_1d(temp_tx[:c_idx])
            Tz_ransac[m] = median_1d(temp_tz[:c_idx])
        else:
            Tx_ransac[m] = tx_med
            Tz_ransac[m] = tz_med

    return (Tx_brut, Tz_brut), (Tx_seuil, Tz_seuil), (Tx_ransac, Tz_ransac)


# ==================================================================
# 2. TRAJECTOIRE ET GENERATION DES POINTS (FLUX DIRECTIONNEL)
# ==================================================================
def generer_trajectoire_VT(seed=SEED_TRAJ):
    rng = np.random.RandomState(seed)
    n_pas_max = int(DIST_OBJECTIF / (V_MS * DT)) + 100

    x_d, z_d, theta_d = np.zeros(n_pas_max), np.zeros(n_pas_max), np.zeros(n_pas_max)
    z_d[0] = Z_DRONE_INIT

    n = 0
    while x_d[n] < DIST_OBJECTIF and n < n_pas_max - 1:
        valide = False
        while not valide:
            th_n = theta_d[n]
            dtheta = rng.uniform(-INTENSITE_DELTA, INTENSITE_DELTA)
            if z_d[n] > Z_DRONE_MAX and th_n > 0: dtheta = rng.uniform(-INTENSITE_DELTA, 0)
            if z_d[n] < Z_DRONE_MIN and th_n < 0: dtheta = rng.uniform(0, INTENSITE_DELTA)
            th_new = th_n + dtheta

            if abs(dtheta) < EPSILON_ANGLE:
                dx, dz = V_MS * DT * np.cos(th_n), V_MS * DT * np.sin(th_n)
            else:
                R = (V_MS * DT) / dtheta
                dx, dz = R * (np.sin(th_new) - np.sin(th_n)), R * (np.cos(th_n) - np.cos(th_new))
            if dx > 0:
                x_d[n+1], z_d[n+1], theta_d[n+1] = x_d[n] + dx, z_d[n] + dz, th_new
                valide = True
        n += 1
    return x_d[:n+1], z_d[:n+1], theta_d[:n+1]

def genere_profondeur(K, rng):
    if K == 1: return np.array([D_NOMINAL])
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

def generer_K_points_mobiles(x_drone, z_drone, theta, K, tau_mob, rng_geo, rng_mob):
    alpha_max, alpha_min = np.deg2rad(ANGLE_MAX_DEG), np.deg2rad(ANGLE_MIN_DEG)
    angles = rng_geo.uniform(-alpha_max, alpha_max, K)
    masque = np.abs(angles) < alpha_min
    while masque.any():
        angles[masque] = rng_geo.uniform(-alpha_max, alpha_max, masque.sum())
        masque = np.abs(angles) < alpha_min

    D_world = genere_profondeur(K, rng_geo)
    Xw_1 = x_drone + D_world * np.cos(theta + angles)
    Zw_1 = z_drone + D_world * np.sin(theta + angles)

    K_mob = int(np.floor(tau_mob * K))
    Xw_2, Zw_2 = Xw_1.copy(), Zw_1.copy()

    if K_mob > 0:
        indices_mob = rng_mob.choice(K, K_mob, replace=False)
        # Trafic directionnel (tous vers l'avant) pour simuler un flux urbain réaliste
        v_x = rng_mob.uniform(5.0, V_MAX_MOB, K_mob)
        v_z = rng_mob.uniform(-V_MAX_MOB, V_MAX_MOB, K_mob)
        Xw_2[indices_mob] += v_x * DT
        Zw_2[indices_mob] += v_z * DT

    return Xw_1, Zw_1, Xw_2, Zw_2

def projeter_N_cameras(Xw, Zw, x_drone, z_drone, theta, N, f=FOCALE):
    dx, dz = Xw - x_drone, Zw - z_drone
    Xc = dx * np.cos(theta) + dz * np.sin(theta)
    D = dx * np.sin(theta) - dz * np.cos(theta)
    X_ck = np.arange(N) * (BASELINE_TOT / (N - 1)) - BASELINE_TOT / 2 if N > 1 else np.array([0.0])
    i_pk = -f * (Xc[None, :] - X_ck[:, None]) / D[None, :]
    return i_pk, Xc, D

def calculer_poids_paires(N):
    if N < 2: return np.zeros(N), 0.0
    X_ck = np.arange(N) * (BASELINE_TOT / (N - 1)) - BASELINE_TOT / 2
    c, sum_B2 = np.zeros(N), 0.0
    for i in range(N):
        for j in range(i + 1, N):
            B_ij = X_ck[j] - X_ck[i]
            c[i] += B_ij; c[j] -= B_ij; sum_B2 += B_ij * B_ij
    return c, sum_B2


def maj_position(Tx, Tz, x_prev, z_prev, theta_prev):
    theta_inc = np.where(np.abs(Tx) > 1e-12, -2.0 * np.arctan2(Tz, Tx), 0.0)
    theta_new = theta_prev + theta_inc
    vx = -Tx * np.cos(theta_prev) - Tz * np.sin(theta_prev)
    vz = -Tx * np.sin(theta_prev) + Tz * np.cos(theta_prev)
    return x_prev + vx, z_prev + vz, theta_new


# ==================================================================
# 3. SIMULATION PARALLELE POUR CHAQUE TAU_MOB
# ==================================================================
def run_simulation_tau_mob(x_d, z_d, theta_d, K, N, n_mc, lot_size, tau_mob):
    N_PAS = len(x_d) - 1
    c, sum_B2 = calculer_poids_paires(N)

    rng_geo = np.random.RandomState(SEED_GEO + N)
    rng_mob = np.random.RandomState(SEED_MOB + int(tau_mob * 1000))

    n_lots = (n_mc + lot_size - 1) // lot_size
    n_mc_effectif = n_lots * lot_size

    err_brut = np.zeros((n_mc_effectif, N_PAS + 1))
    err_seuil = np.zeros((n_mc_effectif, N_PAS + 1))
    err_ransac = np.zeros((n_mc_effectif, N_PAS + 1))
    err_cum_brut = np.zeros((n_mc_effectif, N_PAS + 1))
    err_cum_seuil = np.zeros((n_mc_effectif, N_PAS + 1))
    err_cum_ransac = np.zeros((n_mc_effectif, N_PAS + 1))

    x_est_b = [np.full(lot_size, x_d[0]) for _ in range(n_lots)]
    z_est_b = [np.full(lot_size, z_d[0]) for _ in range(n_lots)]
    th_est_b = [np.full(lot_size, theta_d[0]) for _ in range(n_lots)]

    x_est_s = [np.full(lot_size, x_d[0]) for _ in range(n_lots)]
    z_est_s = [np.full(lot_size, z_d[0]) for _ in range(n_lots)]
    th_est_s = [np.full(lot_size, theta_d[0]) for _ in range(n_lots)]

    x_est_r = [np.full(lot_size, x_d[0]) for _ in range(n_lots)]
    z_est_r = [np.full(lot_size, z_d[0]) for _ in range(n_lots)]
    th_est_r = [np.full(lot_size, theta_d[0]) for _ in range(n_lots)]

    pbar = tqdm(total=N_PAS, desc=f"tau_mob={tau_mob*100:.0f}%", ncols=100)

    for n in range(N_PAS):
        x_n, z_n, th_n = x_d[n], z_d[n], theta_d[n]
        x_np1, z_np1, th_np1 = x_d[n+1], z_d[n+1], theta_d[n+1]

        Xw_1, Zw_1, Xw_2, Zw_2 = generer_K_points_mobiles(x_n, z_n, th_n, K, tau_mob, rng_geo, rng_mob)
        i_pk_1, _, D1 = projeter_N_cameras(Xw_1, Zw_1, x_n, z_n, th_n, N)
        i_pk_2, _, D2 = projeter_N_cameras(Xw_2, Zw_2, x_np1, z_np1, th_np1, N)

        valid = (np.abs(D1) > 0.5) & (np.abs(D2) > 0.5)
        if not valid.all():
            i_pk_1, i_pk_2 = i_pk_1[:, valid], i_pk_2[:, valid]
        K_valid = i_pk_1.shape[1]

        if K_valid < 1:
            for lot in range(n_lots):
                idx_deb, idx_fin = lot * lot_size, (lot + 1) * lot_size
                err_brut[idx_deb:idx_fin, n+1] = err_brut[idx_deb:idx_fin, n]
                err_seuil[idx_deb:idx_fin, n+1] = err_seuil[idx_deb:idx_fin, n]
                err_ransac[idx_deb:idx_fin, n+1] = err_ransac[idx_deb:idx_fin, n]
                err_cum_brut[idx_deb:idx_fin, n+1] = err_cum_brut[idx_deb:idx_fin, n]
                err_cum_seuil[idx_deb:idx_fin, n+1] = err_cum_seuil[idx_deb:idx_fin, n]
                err_cum_ransac[idx_deb:idx_fin, n+1] = err_cum_ransac[idx_deb:idx_fin, n]
            pbar.update(1); continue

        for lot in range(n_lots):
            idx_deb, idx_fin = lot * lot_size, (lot + 1) * lot_size
            seed_lot = SEED_NOISE + n * 10000 + lot * 100

            # Appel direct sans allocation de bruit NumPy en RAM
            res_b, res_s, res_r = estimer_lot_3methodes_numba(
                i_pk_1, i_pk_2, lot_size, BRUIT_MAX_PX * PIXEL_SIZE, c, sum_B2, FOCALE, seed_lot, tau_mob
            )

            # --- Mise à jour et erreur SANS FILTRE ---
            x_prev_b = x_est_b[lot].copy()
            z_prev_b = z_est_b[lot].copy()
            x_est_b[lot], z_est_b[lot], th_est_b[lot] = maj_position(
                res_b[0], res_b[1], x_est_b[lot], z_est_b[lot], th_est_b[lot]
            )
            dist_b = np.sqrt((x_est_b[lot] - x_np1)**2 + (z_est_b[lot] - z_np1)**2)
            err_brut[idx_deb:idx_fin, n+1] = dist_b
            delta_est_b = np.sqrt((x_est_b[lot] - x_prev_b - (x_np1 - x_n))**2 + (z_est_b[lot] - z_prev_b - (z_np1 - z_n))**2)
            err_cum_brut[idx_deb:idx_fin, n+1] = err_cum_brut[idx_deb:idx_fin, n] + delta_est_b

            # --- Mise à jour et erreur SEUILLAGE MAD GEOMETRIQUE ---
            x_prev_s = x_est_s[lot].copy()
            z_prev_s = z_est_s[lot].copy()
            x_est_s[lot], z_est_s[lot], th_est_s[lot] = maj_position(
                res_s[0], res_s[1], x_est_s[lot], z_est_s[lot], th_est_s[lot]
            )
            dist_s = np.sqrt((x_est_s[lot] - x_np1)**2 + (z_est_s[lot] - z_np1)**2)
            err_seuil[idx_deb:idx_fin, n+1] = dist_s
            delta_est_s = np.sqrt((x_est_s[lot] - x_prev_s - (x_np1 - x_n))**2 + (z_est_s[lot] - z_prev_s - (z_np1 - z_n))**2)
            err_cum_seuil[idx_deb:idx_fin, n+1] = err_cum_seuil[idx_deb:idx_fin, n] + delta_est_s

            # --- Mise à jour et erreur RANSAC OPTIMISE ---
            x_prev_r = x_est_r[lot].copy()
            z_prev_r = z_est_r[lot].copy()
            x_est_r[lot], z_est_r[lot], th_est_r[lot] = maj_position(
                res_r[0], res_r[1], x_est_r[lot], z_est_r[lot], th_est_r[lot]
            )
            dist_r = np.sqrt((x_est_r[lot] - x_np1)**2 + (z_est_r[lot] - z_np1)**2)
            err_ransac[idx_deb:idx_fin, n+1] = dist_r
            delta_est_r = np.sqrt((x_est_r[lot] - x_prev_r - (x_np1 - x_n))**2 + (z_est_r[lot] - z_prev_r - (z_np1 - z_n))**2)
            err_cum_ransac[idx_deb:idx_fin, n+1] = err_cum_ransac[idx_deb:idx_fin, n] + delta_est_r

        pbar.update(1)
    pbar.close()

    return {
        'tau_mob': tau_mob,
        'err_brut': err_brut[:n_mc, -1],
        'err_seuil': err_seuil[:n_mc, -1],
        'err_ransac': err_ransac[:n_mc, -1],
        'err_cum_brut': err_cum_brut[:n_mc, -1],
        'err_cum_seuil': err_cum_seuil[:n_mc, -1],
        'err_cum_ransac': err_cum_ransac[:n_mc, -1]
    }

def stats_MC(err):
    return np.median(err), np.percentile(err, 25), np.percentile(err, 75)

def filepath(name):
    return os.path.join(DOSSIER_SORTIE, name)

# ==================================================================
# 4. EXECUTION PRINCIPALE ET TRACE
# ==================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SIMULATION OBJETS MOBILES - COMPARAISON AVEC CORRECTIFS PHYSIQUES")
    print("=" * 70)
    
    print("\nPre-compilation Numba (peut prendre 30-40s)...")
    t0 = time.time()
    c_t, sum_B2_t = calculer_poids_paires(N_FIXE)
    # Correctif Numba : passage d'entiers et formats de tableaux conformes
    _ = estimer_lot_3methodes_numba(
        np.zeros((N_FIXE, 5)), np.zeros((N_FIXE, 5)),
        2, BRUIT_MAX_PX * PIXEL_SIZE, c_t, sum_B2_t, FOCALE, 1234, 0.0
    )
    print(f"  Compilation terminee en {time.time() - t0:.1f} s")

    print("\nGeneration trajectoire verite terrain...")
    x_d, z_d, theta_d = generer_trajectoire_VT()
    print(f"  Trajectoire générée : {len(x_d)-1} pas sur {DIST_OBJECTIF} mètres.")

    print("\n" + "=" * 70)
    print(f"LANCEMENT DES SIMULATIONS SUR {len(TAU_MOB_VALEURS)} VALEURS DE tau_mob")
    print("=" * 70)
    t_start = time.time()

    resultats_list = Parallel(n_jobs=N_JOBS, backend='threading')(
        delayed(run_simulation_tau_mob)(x_d, z_d, theta_d, K_REF, N_FIXE, N_MC_REF, LOT_MC, tau_mob)
        for tau_mob in TAU_MOB_VALEURS
    )
    print(f"\nTemps total de calcul : {(time.time() - t_start)/60:.1f} min")

    tau_vals = np.array([r['tau_mob'] * 100 for r in resultats_list])
    
    med_brut = np.array([stats_MC(r['err_brut'])[0] for r in resultats_list])
    q1_brut  = np.array([stats_MC(r['err_brut'])[1] for r in resultats_list])
    q3_brut  = np.array([stats_MC(r['err_brut'])[2] for r in resultats_list])

    med_seuil = np.array([stats_MC(r['err_seuil'])[0] for r in resultats_list])
    q1_seuil  = np.array([stats_MC(r['err_seuil'])[1] for r in resultats_list])
    q3_seuil  = np.array([stats_MC(r['err_seuil'])[2] for r in resultats_list])

    med_ransac = np.array([stats_MC(r['err_ransac'])[0] for r in resultats_list])
    q1_ransac  = np.array([stats_MC(r['err_ransac'])[1] for r in resultats_list])
    q3_ransac  = np.array([stats_MC(r['err_ransac'])[2] for r in resultats_list])

    med_cum_brut = np.array([stats_MC(r['err_cum_brut'])[0] for r in resultats_list])
    q1_cum_brut  = np.array([stats_MC(r['err_cum_brut'])[1] for r in resultats_list])
    q3_cum_brut  = np.array([stats_MC(r['err_cum_brut'])[2] for r in resultats_list])

    med_cum_seuil = np.array([stats_MC(r['err_cum_seuil'])[0] for r in resultats_list])
    q1_cum_seuil  = np.array([stats_MC(r['err_cum_seuil'])[1] for r in resultats_list])
    q3_cum_seuil  = np.array([stats_MC(r['err_cum_seuil'])[2] for r in resultats_list])

    med_cum_ransac = np.array([stats_MC(r['err_cum_ransac'])[0] for r in resultats_list])
    q1_cum_ransac  = np.array([stats_MC(r['err_cum_ransac'])[1] for r in resultats_list])
    q3_cum_ransac  = np.array([stats_MC(r['err_cum_ransac'])[2] for r in resultats_list])

    print("\n" + "=" * 70)
    print(f"RESULTATS FINAUX : ERREUR CUMULEE A {DIST_OBJECTIF} METRES [m]")
    print("=" * 70)
    print(f"{'tau_mob':>8} | {'Erreur finale':>14} | {'Erreur cum.':>14} | {'Erreur finale':>14} | {'Erreur cum.':>14} | {'Erreur finale':>14} | {'Erreur cum.':>14}")
    print("-" * 105)
    for i in range(len(tau_vals)):
        print(f" {tau_vals[i]:>6.0f}%  | {med_brut[i]:>14.3f} | {med_cum_brut[i]:>14.3f} | {med_seuil[i]:>14.3f} | {med_cum_seuil[i]:>14.3f} | {med_ransac[i]:>14.3f} | {med_cum_ransac[i]:>14.3f}")

    print("\nGénération de la figure comparative superposée...")
    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(tau_vals, med_cum_brut, 'o--', color='#d95f02', lw=2, markersize=8, label='Sans filtre (Médiane brute)')
    ax.fill_between(tau_vals, q1_cum_brut, q3_cum_brut, color='#d95f02', alpha=0.15)

    ax.plot(tau_vals, med_cum_seuil, 's-', color='#1b9e77', lw=2.5, markersize=8, label='Seuillage MAD géométrique (angulaire)')
    ax.fill_between(tau_vals, q1_cum_seuil, q3_cum_seuil, color='#1b9e77', alpha=0.2)

    ax.plot(tau_vals, med_cum_ransac, '^-', color='#7570b3', lw=2.5, markersize=8, label='RANSAC 2D optimisé (Consensus inliers)')
    ax.fill_between(tau_vals, q1_cum_ransac, q3_cum_ransac, color='#7570b3', alpha=0.2)

    ax.axhline(med_cum_brut[0], color='black', ls=':', lw=1.5, alpha=0.6, label=f'Référence idéale absolue ({med_cum_brut[0]:.2f} m)')

    ax.set_xlabel(r'Taux de dynamisme $\tau_{mob}$ [%]', fontsize=12)
    ax.set_ylabel(f'Erreur cumulée finale $\\varepsilon_{{pos}}({int(DIST_OBJECTIF)}\\,\\mathrm{{m}})$ [m]', fontsize=12)

    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, ls='--', alpha=0.6)
    ax.set_xticks(tau_vals)
    
    plt.tight_layout()
    plt.savefig(filepath('Fig_comparaison_filtres_objets_mobiles.png'), dpi=150)
    plt.close()

    print(f"\n=== SIMULATION TERMINÉE AVEC SUCCÈS ===")
    print(f"Graphique exporté sous : Fig_comparaison_filtres_objets_mobiles.png")