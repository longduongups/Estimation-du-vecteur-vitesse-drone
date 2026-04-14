import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# =====================================================
# 1. PARAMÈTRES TECHNIQUES
# =====================================================
FPS = 50
DT = 1.0 / FPS
V_KMH = 10.0
V_MS = V_KMH / 3.6     # ~2.78 m/s
FOCALE = 0.035         # f = 35mm
DIST_OBJECTIF = 500.0
# Paramètres de vol
INTENSITE_DELTA = 0.005
N_POINTS_MUR = 2500
EPSILON_ANGLE = 1e-9   # Seuil pour basculer Arc -> Droite
# =====================================================
# 2. GÉNÉRATION DE L'ENVIRONNEMENT (Mur à Z = 0)
# =====================================================
# Le drone évolue à Z = 10, le mur est fixe à Z = 0
x_wall = np.linspace(-50, DIST_OBJECTIF + 50, N_POINTS_MUR)
z_wall = np.zeros(N_POINTS_MUR)
y_wall = np.random.uniform(-5, 5, N_POINTS_MUR)
# =====================================================
# 3. SIMULATION ET ESTIMATION DU VECTEUR VITESSE
# =====================================================
x_d, z_d = [0.0], [10.0]  # Départ à (0, 10)
phi_reel_list = [0.0]
phi_est_list = [0.0]
v_estim_list = []
while x_d[-1] < DIST_OBJECTIF:
    phi_n = phi_reel_list[-1]
    valide = False
    # --- LOGIQUE DE REJET : GARANTIR L'AVANCEMENT ---
    while not valide:
        # 1. Génération d'une variation d'angle (Delta Phi)
        v_rot = np.random.uniform(-INTENSITE_DELTA, INTENSITE_DELTA)
# --- CORRECTION DOUCE (Une seule fois si nécessaire) ---
        # Si le drone est trop loin (Z > 11) ET qu'il pointe encore vers l'extérieur (phi_n > 0)
        # On force une petite rotation pour le ramener.
        if z_d[-1] > 10.3 and phi_n > 0:
            v_rot = np.random.uniform(-INTENSITE_DELTA, 0)
            
        # Si le drone est trop près (Z < 9) ET qu'il pointe encore vers le mur (phi_n < 0)
        elif z_d[-1] < 9.7 and phi_n < 0:
            v_rot = np.random.uniform(0, INTENSITE_DELTA)
            
        # Dès que l'angle (phi_n) s'est inversé pour revenir dans le bon sens, 
        # les conditions ci-dessus sont ignorées et v_rot reste purement aléatoire !

        # 2. MODÈLE HYBRIDE : SEGMENT DROIT OU ARC CIRCULAIRE
        if abs(v_rot) < EPSILON_ANGLE:
            dx = V_MS * DT * np.cos(phi_n)
            dz = V_MS * DT * np.sin(phi_n)
        else:
            R_reel = (V_MS * DT) / v_rot
            dx = R_reel * (np.sin(phi_n + v_rot) - np.sin(phi_n))
            dz = R_reel * (np.cos(phi_n) - np.cos(phi_n + v_rot))

        # 3. FILTRE DE PROGRESSION
        if dx > 0:
            valide = True
            current_phi = phi_n + v_rot
            
    c1 = np.array([x_d[-1], z_d[-1]])
    c2 = c1 + np.array([dx, dz])
    # --- ÉVALUATION DE L'ESTIMATION ---
    # Sélection d'un point M cible sur le mur
    idx = np.searchsorted(x_wall, c1[0] + 5.0)
    idx = np.clip(idx, 0, N_POINTS_MUR - 1)
    M = np.array([x_wall[idx], z_wall[idx]])
    # Distances caméra (Profondeur sur l'axe Z)
    D1, D2 = abs(c1[1] - M[1]), abs(c2[1] - M[1])
    x1_c, x2_c = M[0] - c1[0], M[0] - c2[0]
    # Projections sur le plan image
    i1 = -FOCALE * x1_c / D1
    i2 = -FOCALE * x2_c / D2
    try:
        # Calcul du rayon et de la vitesse selon le modèle géométrique
        num_R = i1**2 * D1**2 - i2**2 * D2**2 + FOCALE**2 * (D1**2 - D2**2)
        den_R = 2 * FOCALE**2 * (D2 - D1)
        if abs(D2 - D1) < 1e-9:
            v_calc, theta_est = V_MS, 0.0
        else:
            R_est = num_R / den_R
            arg_sin = (2 * FOCALE * (D2 - D1) * (i2 * D2 - i1 * D1)) / num_R
            theta_est = np.arcsin(np.clip(arg_sin, -1, 1))
            v_calc = abs((R_est / DT) * theta_est)
        v_estim_list.append(v_calc)
        phi_est_list.append(phi_est_list[-1] + theta_est)
    except:
        v_estim_list.append(V_MS)
        phi_est_list.append(phi_est_list[-1])
    x_d.append(c2[0])
    z_d.append(c2[1])
    phi_reel_list.append(current_phi)
# =====================================================
# 4. AFFICHAGE DES RÉSULTATS
# =====================================================
fig = plt.figure(figsize=(15, 10))
# --- A. Environnement 3D ---
ax_3d = fig.add_subplot(221, projection='3d')
ax_3d.scatter(x_wall, z_wall, y_wall, c='gray', s=1, alpha=0.1, label="Mur (Z=0)")
ax_3d.plot(x_d, z_d, np.zeros_like(x_d), color='red', linewidth=2, label="Trajectoire Drone")
ax_3d.set_title("Visualisation de la Trajectoire 3D")
ax_3d.set_xlabel("X (Avancement)")
ax_3d.set_ylabel("Z (Profondeur)")
ax_3d.set_ylim(-2, 12)
ax_3d.legend()
# --- B. Estimation de la Vitesse ---
#ax_v = fig.add_subplot(222)
#ax_v.plot(v_estim_list, color='orange', label="Vitesse Estimée")
#ax_v.axhline(y=V_MS, color='r', linestyle='--', label=f"Vitesse Réelle ({V_MS:.2f} m/s)")
#ax_v.set_title("Estimation de la Norme de Vitesse")
#ax_v.set_ylabel("m/s")
#ax_v.set_ylim(V_MS - 0.5, V_MS + 0.5)
#ax_v.legend()
#ax_v.grid(True, alpha=0.3)
# --- C. Erreur de Cap ---
#error_phi = np.degrees(phi_reel_list) - np.degrees(phi_est_list)
#ax_phi = fig.add_subplot(212)
#ax_phi.plot(error_phi, color='purple', label="Erreur Cumulative")
#ax_phi.set_title("Dérive de l'Orientation (Cap)")
#ax_phi.set_ylabel("Erreur (degrés)")
#ax_phi.set_xlabel("Itérations")
#ax_phi.legend()
#ax_phi.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()