import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

# Désactiver les avertissements mathématiques mineurs
warnings.filterwarnings("ignore")

# =====================================================
# 1. PARAMÈTRES TECHNIQUES GLOBAUX
# =====================================================
INJECTER_BRUIT = False     # Mettez True pour simuler la vraie vie !
BRUIT_PIXEL = 1.0          
TAILLE_PIXEL = 5e-6 

FPS = 50
DT = 1.0 / FPS           # Pas de temps (0.02 s)
V_KMH = 10.0
V_MS = V_KMH / 3.6       # Vitesse réelle (≈ 2.78 m/s)
FOCALE = 0.01            # Distance focale de la caméra (f = 10 mm)
DIST_OBJECTIF = 500.0    # Distance totale à parcourir sur l'axe X

INTENSITE_DELTA = 0.005
N_POINTS_MUR = 1500
EPSILON_ANGLE = 1e-9

# =====================================================
# 2. GÉNÉRATION DE L'ENVIRONNEMENT
# =====================================================
x_wall = np.linspace(-50, DIST_OBJECTIF + 50, N_POINTS_MUR)
z_wall = np.zeros(N_POINTS_MUR)
y_wall = np.random.uniform(-5, 5, N_POINTS_MUR) 

# =====================================================
# 3. BOUCLE DE SIMULATION CINÉMATIQUE ET ODOMÉTRIQUE
# =====================================================
x_d, z_d = [0.0], [10.0]  # Coordonnées réelles de départ
phi_reel_list = [0.0]     

# --- INTÉGRATION GLOBALE ESTIMÉE ---
x_est_list, z_est_list = [0.0], [10.0] 
phi_est_list = [0.0] 
err_pos_cumulee = [0.0]   # Erreur globale en mètres

print("Début de la simulation (Calcul de l'erreur cumulée)...")

while x_d[-1] < DIST_OBJECTIF:
    phi_n = phi_reel_list[-1]
    valide = False

    # --- A. VÉRITÉ TERRAIN ---
    while not valide:
        v_rot = np.random.uniform(-INTENSITE_DELTA, INTENSITE_DELTA)

        if z_d[-1] > 10.3 and phi_n > 0:
            v_rot = np.random.uniform(-INTENSITE_DELTA, 0)
        elif z_d[-1] < 9.7 and phi_n < 0:
            v_rot = np.random.uniform(0, INTENSITE_DELTA)

        if abs(v_rot) < EPSILON_ANGLE:
            dx = V_MS * DT * np.cos(phi_n)
            dz = V_MS * DT * np.sin(phi_n)
        else:
            R_reel = (V_MS * DT) / v_rot
            dx = R_reel * (np.sin(phi_n + v_rot) - np.sin(phi_n))
            dz = R_reel * (np.cos(phi_n) - np.cos(phi_n + v_rot))

        if dx > 0:
            valide = True
            current_phi = phi_n + v_rot
            
    c1 = np.array([x_d[-1], z_d[-1]])
    c2 = c1 + np.array([dx, dz])

    # --- B. MODÈLE STÉNOPÉ ---
    idx = np.searchsorted(x_wall, c1[0] + 5.0)
    idx = np.clip(idx, 0, N_POINTS_MUR - 1)
    M = np.array([x_wall[idx], z_wall[idx]])

    dx1_monde, dz1_monde = M[0] - c1[0], M[1] - c1[1]
    dx2_monde, dz2_monde = M[0] - c2[0], M[1] - c2[1]

    x1_c = dx1_monde * np.cos(phi_n) + dz1_monde * np.sin(phi_n)
    D1   = dx1_monde * np.sin(phi_n) - dz1_monde * np.cos(phi_n)
    
    x2_c = dx2_monde * np.cos(current_phi) + dz2_monde * np.sin(current_phi)
    D2   = dx2_monde * np.sin(current_phi) - dz2_monde * np.cos(current_phi)

    i1_exact = -FOCALE * x1_c / D1
    i2_exact = -FOCALE * x2_c / D2

    # INJECTION DU BRUIT
    if INJECTER_BRUIT:
        i1 = i1_exact + np.random.normal(0, BRUIT_PIXEL * TAILLE_PIXEL)
        i2 = i2_exact + np.random.normal(0, BRUIT_PIXEL * TAILLE_PIXEL)
    else:
        i1, i2 = i1_exact, i2_exact

    # --- C. ESTIMATION ET INTÉGRATION ---
    try:
        denom = (i1 * D1 + i2 * D2)**2 + (FOCALE**2) * (D2 - D1)**2
        
        if denom < 1e-12:
            Tx_est, Tz_est = -dx, -dz
        else:
            K = (D2**2 - D1**2) * (FOCALE**2) + (D2**2) * (i2**2) - (D1**2) * (i1**2)
            Tx_est = -(1.0 / FOCALE) * (D2 * i2 + D1 * i1) * (K / denom)
            Tz_est = -(D2 - D1) * (K / denom)

        # 1. Calcul de l'angle estimé
        if abs(Tx_est) > 1e-12:
            theta_est = -2.0 * np.arctan(Tz_est / Tx_est)
            if abs(theta_est) < 1e-13: 
                theta_est = 0.0 
        else:
            theta_est = 0.0
            
        nouveau_cap_estime = phi_est_list[-1] + theta_est
        phi_est_list.append(nouveau_cap_estime)

        # 2. Intégration globale (Dead Reckoning)
        V_x_monde = -Tx_est * np.cos(phi_est_list[-2]) - Tz_est * np.sin(phi_est_list[-2])
        V_z_monde = -Tx_est * np.sin(phi_est_list[-2]) + Tz_est * np.cos(phi_est_list[-2])

        x_nouv_est = x_est_list[-1] + V_x_monde
        z_nouv_est = z_est_list[-1] + V_z_monde

        x_est_list.append(x_nouv_est)
        z_est_list.append(z_nouv_est)

        # 3. Calcul de l'erreur CUMULÉE
        err_globale = np.sqrt((x_nouv_est - c2[0])**2 + (z_nouv_est - c2[1])**2)
        err_pos_cumulee.append(err_globale)

    except:
        phi_est_list.append(phi_est_list[-1])
        x_est_list.append(x_est_list[-1] + V_MS * DT * np.cos(phi_est_list[-1]))
        z_est_list.append(z_est_list[-1] + V_MS * DT * np.sin(phi_est_list[-1]))
        err_pos_cumulee.append(err_pos_cumulee[-1])

    x_d.append(c2[0])
    z_d.append(c2[1])
    phi_reel_list.append(current_phi)

print("Simulation terminée. Génération des graphiques...")

error_phi_deg = np.degrees(np.array(phi_reel_list)) - np.degrees(np.array(phi_est_list))

# =====================================================
# 4. AFFICHAGE DES RÉSULTATS (3 FENÊTRES SÉPARÉES)
# =====================================================
titre_etat = "Avec Bruit" if INJECTER_BRUIT else "Sans Bruit"

# --- Fenêtre 1 : Environnement 3D ---
fig1 = plt.figure(figsize=(10, 8))
ax_3d = fig1.add_subplot(111, projection='3d')
ax_3d.scatter(x_wall, z_wall, y_wall, c='gray', s=1, alpha=0.1, label="Mur")
# Lignes plus fines avec linewidth=1
ax_3d.plot(x_d, z_d, np.zeros_like(x_d), color='red', linewidth=2, alpha=0.5, label="Trajectoire Réelle")
ax_3d.plot(x_est_list, z_est_list, np.zeros_like(x_est_list), color='black', linestyle='--', linewidth=1.5, label="Trajectoire Estimée")
ax_3d.set_title(f"1. Trajectoire Aléatoire Globale - {titre_etat}", fontweight='bold')
ax_3d.set_xlabel("X [m]")
ax_3d.set_ylabel("Z [m]")
ax_3d.set_ylim(-2, 12)
ax_3d.legend()

# --- Fenêtre 2 : Erreur de Position CUMULÉE ---
fig2 = plt.figure(figsize=(10, 5))
ax_err_pos = fig2.add_subplot(111)
ax_err_pos.plot(err_pos_cumulee, color='purple', lw=1.5, label="Dérive globale accumulée")
ax_err_pos.set_title(f"2. Erreur de Positionnement CUMULÉE - {titre_etat}", fontweight='bold')
ax_err_pos.set_ylabel("Erreur [mètres]")
ax_err_pos.set_xlabel("Itérations")

# ECHELLE INTELLIGENTE
if INJECTER_BRUIT:
    ax_err_pos.autoscale(axis='y') 
else:
    ax_err_pos.set_ylim(-0.000000001, 0.000000001) 

ax_err_pos.legend()
ax_err_pos.grid(True, linestyle='--', alpha=0.7)

# --- Fenêtre 3 : Erreur d'Orientation (Cap) ---
fig3 = plt.figure(figsize=(10, 5))
ax_err_phi = fig3.add_subplot(111)
ax_err_phi.plot(error_phi_deg, color='teal', lw=1.5, label="Erreur cumulative de cap")
ax_err_phi.set_title(f"3. Erreur d'Orientation Cumulative (Cap) - {titre_etat}", fontweight='bold')
ax_err_phi.set_ylabel("Erreur [degrés]")
ax_err_phi.set_xlabel("Itérations")

# ECHELLE INTELLIGENTE
if INJECTER_BRUIT:
    ax_err_phi.autoscale(axis='y') 
else:
    ax_err_phi.set_ylim(-0.000000001, 0.000000001) 

ax_err_phi.legend()
ax_err_phi.grid(True, linestyle='--', alpha=0.7)

plt.show()