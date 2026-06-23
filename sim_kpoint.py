import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================
# PARAMETRES
# ============================================================

FPS = 50
DT = 1.0 / FPS

V_MS = 10.0 / 3.6

FOCALE = 0.01
BASELINE = 0.20

PIXEL_SIZE = 5e-6
BRUIT_MAX_PX = 0.5

DIST_OBJECTIF = 500.0

INTENSITE_DELTA = 0.002

EPSILON_ANGLE = 1e-12

# points générés localement
D_MIN = 5.0
D_MAX = 15.0
D_NOMINAL = 10.0      
SIGMA_FACADE = 0.3    
RATIO_FACADE = 0.8    
ANGLE_MAX_DEG = 60.0
ANGLE_MIN_DEG = 5.0

CONFIGS = {
    1: 1000,
    100: 300,
    500: 70,
    1500: 50
}

np.random.seed(42)

# ============================================================
# MODELE 2
# ============================================================

def modele_2_vec(i1, i2, D1, D2, f=FOCALE):
    denom = (
        (i1 * D1 + i2 * D2) ** 2
        + (f ** 2) * (D2 - D1) ** 2
    )

    Knum = (
        (D2 ** 2 - D1 ** 2) * (f ** 2)
        + D2 ** 2 * i2 ** 2
        - D1 ** 2 * i1 ** 2
    )

    Tx = np.where(
        denom > 1e-15,
        -(1.0 / f) * (D2 * i2 + D1 * i1) * Knum / denom,
        0.0
    )

    Tz = np.where(
        denom > 1e-15,
        -(D2 - D1) * Knum / denom,
        0.0
    )

    return Tx, Tz

# ============================================================
# TRAJECTOIRE VERITE TERRAIN
# ============================================================

print("Generation trajectoire vérité terrain")

x_d = [0.0]
z_d = [10.0]
phi_d = [0.0]

while x_d[-1] < DIST_OBJECTIF:
    phi = phi_d[-1]
    valide = False

    while not valide:
        dphi = np.random.uniform(-INTENSITE_DELTA, INTENSITE_DELTA)

        if z_d[-1] > 12 and phi > 0:
            dphi = np.random.uniform(-INTENSITE_DELTA, 0)

        if z_d[-1] < 8 and phi < 0:
            dphi = np.random.uniform(0, INTENSITE_DELTA)

        if abs(dphi) < EPSILON_ANGLE:
            dx = V_MS * DT * np.cos(phi)
            dz = V_MS * DT * np.sin(phi)
        else:
            R = (V_MS * DT) / dphi
            dx = R * (np.sin(phi + dphi) - np.sin(phi))
            dz = R * (np.cos(phi) - np.cos(phi + dphi))

        if dx > 0:
            valide = True
            x_d.append(x_d[-1] + dx)
            z_d.append(z_d[-1] + dz)
            phi_d.append(phi + dphi)

x_d = np.array(x_d)
z_d = np.array(z_d)
phi_d = np.array(phi_d)

N_PAS = len(x_d) - 1

print("Nombre de pas :", N_PAS)

# ============================================================
# GENERATION DE PROFONDEUR
# ============================================================

def genere_profondeur(K):
    """Génère un tableau de distances D selon le profil de la façade"""
    # Règle spéciale demandée : si K=1, la distance est exactement de 10m
    if K == 1:
        return np.array([D_NOMINAL])
        
    K_facade = int(K * RATIO_FACADE)
    K_saillie = K - K_facade
    
    # 1. Génération des points de la façade (Loi normale tronquée à ±5m)
    z_facade = np.random.normal(0, SIGMA_FACADE, K_facade)
    masque = (z_facade < -5) | (z_facade > 5)
    
    while masque.any():
        z_facade[masque] = np.random.normal(0, SIGMA_FACADE, masque.sum())
        masque = (z_facade < -5) | (z_facade > 5)
        
    # 2. Génération des points en saillie (Loi uniforme entre -5m et 5m)
    z_saillie = np.random.uniform(-5, 5, K_saillie)
    
    # 3. Fusion et mélange aléatoire
    z_wall = np.concatenate([z_facade, z_saillie])
    np.random.shuffle(z_wall)
    
    # La profondeur finale est la distance nominale + le relief
    return D_NOMINAL + z_wall


def genere_points(K):
    # Génération des angles avec filtrage de la zone centrale
    angles = np.random.uniform(-np.deg2rad(ANGLE_MAX_DEG), np.deg2rad(ANGLE_MAX_DEG), K)
    mask = (np.abs(angles) < np.deg2rad(ANGLE_MIN_DEG))

    while np.any(mask):
        angles[mask] = np.random.uniform(-np.deg2rad(ANGLE_MAX_DEG), np.deg2rad(ANGLE_MAX_DEG), mask.sum())
        mask = (np.abs(angles) < np.deg2rad(ANGLE_MIN_DEG))

    # Utilisation de la distribution pour D
    D = genere_profondeur(K)
    Xc = D * np.tan(angles)

    return Xc, D

# ============================================================
# MONTE CARLO
# ============================================================

def simu_mc(K, n_mc):
    np.random.seed(123)

    x_est = np.zeros((n_mc, N_PAS + 1))
    z_est = np.zeros((n_mc, N_PAS + 1))
    phi_est = np.zeros((n_mc, N_PAS + 1))

    x_est[:, 0] = 0.0
    z_est[:, 0] = 10.0

    err = np.zeros((n_mc, N_PAS + 1))

    for n in range(N_PAS):
        x1, z1, phi1 = x_d[n], z_d[n], phi_d[n]
        x2, z2, phi2 = x_d[n + 1], z_d[n + 1], phi_d[n + 1]

        # GENERATION DES K POINTS
        angles = np.random.uniform(-np.deg2rad(ANGLE_MAX_DEG), np.deg2rad(ANGLE_MAX_DEG), K)
        mask = np.abs(angles) < np.deg2rad(ANGLE_MIN_DEG)
        while np.any(mask):
            angles[mask] = np.random.uniform(-np.deg2rad(ANGLE_MAX_DEG), np.deg2rad(ANGLE_MAX_DEG), np.sum(mask))
            mask = np.abs(angles) < np.deg2rad(ANGLE_MIN_DEG)

        # <-- CORRECTION ICI : Utilisation de la fonction avec la géométrie de façade
        D_world = genere_profondeur(K)

        Xw = x1 + D_world * np.cos(phi1 + angles)
        Zw = z1 + D_world * np.sin(phi1 + angles)

        dx1, dz1 = Xw - x1, Zw - z1
        Xc1 = dx1 * np.cos(phi1) + dz1 * np.sin(phi1)
        D1_true = dx1 * np.sin(phi1) - dz1 * np.cos(phi1)

        dx2, dz2 = Xw - x2, Zw - z2
        Xc2 = dx2 * np.cos(phi2) + dz2 * np.sin(phi2)
        D2_true = dx2 * np.sin(phi2) - dz2 * np.cos(phi2)

        valid = (np.abs(D1_true) > 0.5) & (np.abs(D2_true) > 0.5)

        Xc1 = Xc1[valid]
        Xc2 = Xc2[valid]
        D1_true = D1_true[valid]
        D2_true = D2_true[valid]

        K_valid = len(D1_true)

        if K_valid < 1:
            x_est[:, n + 1] = x_est[:, n]
            z_est[:, n + 1] = z_est[:, n]
            phi_est[:, n + 1] = phi_est[:, n]
            err[:, n + 1] = err[:, n]
            continue

        i1_true = -FOCALE * Xc1 / D1_true
        i2_true = -FOCALE * Xc2 / D2_true
        d1_true = FOCALE * BASELINE / D1_true
        d2_true = FOCALE * BASELINE / D2_true

        bruit_i1 = np.random.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX, (n_mc, K_valid)) * PIXEL_SIZE
        bruit_i2 = np.random.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX, (n_mc, K_valid)) * PIXEL_SIZE
        
        bruit_d1 = (np.random.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX, (n_mc, K_valid)) - 
                    np.random.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX, (n_mc, K_valid))) * PIXEL_SIZE
        bruit_d2 = (np.random.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX, (n_mc, K_valid)) - 
                    np.random.uniform(-BRUIT_MAX_PX, BRUIT_MAX_PX, (n_mc, K_valid))) * PIXEL_SIZE

        i1_obs = i1_true[None, :] + bruit_i1
        i2_obs = i2_true[None, :] + bruit_i2
        d1_obs = d1_true[None, :] + bruit_d1
        d2_obs = d2_true[None, :] + bruit_d2

        D1_obs = FOCALE * BASELINE / d1_obs
        D2_obs = FOCALE * BASELINE / d2_obs

        Tx_pts, Tz_pts = modele_2_vec(i1_obs, i2_obs, D1_obs, D2_obs)

        Tx = np.mean(Tx_pts, axis=1)
        Tz = np.mean(Tz_pts, axis=1)

        theta = np.where(np.abs(Tx) > 1e-12, -2.0 * np.arctan2(Tz, Tx), 0.0)

        phi_est[:, n + 1] = phi_est[:, n] + theta
        vx = -Tx * np.cos(phi_est[:, n]) - Tz * np.sin(phi_est[:, n])
        vz = -Tx * np.sin(phi_est[:, n]) + Tz * np.cos(phi_est[:, n])

        x_est[:, n + 1] = x_est[:, n] + vx
        z_est[:, n + 1] = z_est[:, n] + vz

        err[:, n + 1] = np.sqrt((x_est[:, n + 1] - x_d[n + 1])**2 + (z_est[:, n + 1] - z_d[n + 1])**2)

    return x_est, z_est, err

# ============================================================
# EXECUTION
# ============================================================

resultats = {}

for K, NMC in CONFIGS.items():
    print(f"\nK={K}  MC={NMC}")
    t0 = time.time()
    x_e, z_e, err = simu_mc(K, NMC)
    print("Temps =", round(time.time() - t0, 1), "s")

    med = np.median(err, axis=0)
    q25 = np.percentile(err, 25, axis=0)
    q75 = np.percentile(err, 75, axis=0)
    idx = np.argsort(err[:, -1])[len(err)//2]

    resultats[K] = {
        "med": med,
        "q25": q25,
        "q75": q75,
        "x": x_e[idx],
        "z": z_e[idx]
    }

    print("Erreur finale =", round(med[-1],4), "m")

# ============================================================
# FIGURE ERREUR
# ============================================================

# <-- CORRECTION ICI POUR LES COULEURS (500 au lieu de 300)
colors = {
    1: "#e41a1c",     
    100: "#d95f02",
    500: "#1b9e77",
    1500:"#377eb8"
}

iterations = np.arange(N_PAS+1)

plt.figure(figsize=(12,7))

# <-- CORRECTION ICI POUR LA BOUCLE
for K in [1, 100, 500, 1500]:
    r = resultats[K]
    plt.fill_between(iterations, r["q25"], r["q75"], alpha=0.15, color=colors[K])
    plt.plot(iterations, r["med"], lw=2, color=colors[K], label=f"K={K}")

plt.grid(True)
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Erreur [m]")
plt.title("Influence du nombre de points")
plt.tight_layout()
plt.show()

# ============================================================
# FIGURE TRAJECTOIRES
# ============================================================

plt.figure(figsize=(12,7))

plt.plot(x_d, z_d, 'k', lw=3, label='Verite terrain')

for K in [1, 100, 500, 1500]:
    r = resultats[K]
    plt.plot(r["x"], r["z"], '--', lw=2, color=colors[K], label=f"K={K}")

plt.grid(True)
plt.legend()
plt.xlabel("Xm [m]")
plt.ylabel("Zm [m]")
plt.title("Trajectoires estimees")
plt.tight_layout()
plt.show()