import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. PARAMÈTRES TECHNIQUES GLOBAUX
# =====================================================
INJECTER_BRUIT = False     # <-- SANS BRUIT (Validation mathématique de l'erreur cumulée)
BRUIT_PIXEL = 1.0          
TAILLE_PIXEL = 5e-6        

FPS = 50
DT = 1.0 / FPS           # Pas de temps (0.02 s)
V_KMH = 10.0
V_MS = V_KMH / 3.6       # Vitesse (≈ 2.78 m/s)
FOCALE = 0.01            # Focale (f = 10 mm)
DISTANCE_TOTALE = 500.0  # Distance de vol STRICTE de 500 mètres

# Les 6 scénarios de la Note d'Étude
scenarios = [
    'rect_0', 'rect_15', 'rect_30', 
    'curve_10', 'curve_30', 'curve_100'
]

res_err_pos = {}
res_err_phi = {}
res_traj_x = {}
res_traj_z = {}
res_est_x = {} # NOUVEAU : Sauvegarde X estimé
res_est_z = {} # NOUVEAU : Sauvegarde Z estimé

print("Lancement de la simulation (Distance : 500m) avec Bruit =", INJECTER_BRUIT)

# =====================================================
# 2. BOUCLE PRINCIPALE (SUR LES 6 SCÉNARIOS)
# =====================================================
for SCENARIO in scenarios:
    print(f"Calcul du scénario : {SCENARIO}...")
    
    # Paramétrage initial de l'orientation (theta) pour chaque cas
    if SCENARIO == 'rect_0': initial_phi, v_rot = 0.0, 0.0
    elif SCENARIO == 'rect_15': initial_phi, v_rot = np.radians(15), 0.0
    elif SCENARIO == 'rect_30': initial_phi, v_rot = np.radians(30), 0.0
    elif SCENARIO == 'curve_10': initial_phi, v_rot = 0.0, (V_MS * DT) / 10.0
    elif SCENARIO == 'curve_30': initial_phi, v_rot = 0.0, (V_MS * DT) / 30.0
    elif SCENARIO == 'curve_100': initial_phi, v_rot = 0.0, (V_MS * DT) / 100.0

    x_d, z_d = [0.0], [10.0]
    phi_reel_list = [initial_phi]     
    x_est_list, z_est_list = [0.0], [10.0] 
    phi_est_list = [initial_phi] 
    err_pos_cumulee = [0.0]
    
    dist_parcourue = 0.0

    # Boucle de vol sur 500 mètres
    while dist_parcourue < DISTANCE_TOTALE:
        phi_n = phi_reel_list[-1]

        # --- A. VÉRITÉ TERRAIN STRICTE ---
        if v_rot == 0.0:
            dx = V_MS * DT * np.cos(initial_phi)
            dz = V_MS * DT * np.sin(initial_phi)
            current_phi = initial_phi
        else:
            R_reel = (V_MS * DT) / v_rot
            dx = R_reel * (np.sin(phi_n + v_rot) - np.sin(phi_n))
            dz = R_reel * (np.cos(phi_n) - np.cos(phi_n + v_rot))
            current_phi = phi_n + v_rot

        dist_parcourue += np.sqrt(dx**2 + dz**2)
            
        c1 = np.array([x_d[-1], z_d[-1]])
        c2 = c1 + np.array([dx, dz])

        # --- B. MODÈLE STÉNOPÉ ---
        # Le point M avance virtuellement avec le drone (toujours à 10m devant l'objectif)
        dir_optique_x = np.sin(phi_n)
        dir_optique_z = -np.cos(phi_n)
        
        M = np.array([c1[0] + 10.0 * dir_optique_x, c1[1] + 10.0 * dir_optique_z])

        dx1_monde, dz1_monde = M[0] - c1[0], M[1] - c1[1]
        dx2_monde, dz2_monde = M[0] - c2[0], M[1] - c2[1]

        # Caméra perpendiculaire au vecteur vitesse
        x1_c = dx1_monde * np.cos(phi_n) + dz1_monde * np.sin(phi_n)
        D1   = dx1_monde * np.sin(phi_n) - dz1_monde * np.cos(phi_n)
        x2_c = dx2_monde * np.cos(current_phi) + dz2_monde * np.sin(current_phi)
        D2   = dx2_monde * np.sin(current_phi) - dz2_monde * np.cos(current_phi)

        i1_exact = -FOCALE * x1_c / D1
        i2_exact = -FOCALE * x2_c / D2

        if INJECTER_BRUIT:
            i1 = i1_exact + np.random.normal(0, BRUIT_PIXEL * TAILLE_PIXEL)
            i2 = i2_exact + np.random.normal(0, BRUIT_PIXEL * TAILLE_PIXEL)
        else:
            i1, i2 = i1_exact, i2_exact

        # --- C. ESTIMATION ODOMÉTRIQUE (Modèle 2) ---
        try:
            denom = (i1 * D1 + i2 * D2)**2 + (FOCALE**2) * (D2 - D1)**2
            if denom < 1e-12:
                Tx_est, Tz_est = -dx, -dz
            else:
                K = (D2**2 - D1**2) * (FOCALE**2) + (D2**2) * (i2**2) - (D1**2) * (i1**2)
                Tx_est = -(1.0 / FOCALE) * (D2 * i2 + D1 * i1) * (K / denom)
                Tz_est = -(D2 - D1) * (K / denom)

            if abs(Tx_est) > 1e-12:
                theta_est = -2.0 * np.arctan(Tz_est / Tx_est)
                # Nettoyage des erreurs de précision machine (flottants)
                if abs(theta_est) < 1e-13: 
                    theta_est = 0.0 
            else:
                theta_est = 0.0
                
            # Intégration CUMULÉE du cap
            phi_est_list.append(phi_est_list[-1] + theta_est)

            # Intégration CUMULÉE de la position (Dead Reckoning)
            V_x_monde = -Tx_est * np.cos(phi_est_list[-2]) - Tz_est * np.sin(phi_est_list[-2])
            V_z_monde = -Tx_est * np.sin(phi_est_list[-2]) + Tz_est * np.cos(phi_est_list[-2])

            x_nouv_est = x_est_list[-1] + V_x_monde
            z_nouv_est = z_est_list[-1] + V_z_monde

            x_est_list.append(x_nouv_est)
            z_est_list.append(z_nouv_est)

            # CALCUL DE L'ERREUR GLOBALE CUMULÉE (Distance absolue au temps T)
            err_pos_cumulee.append(np.sqrt((x_nouv_est - c2[0])**2 + (z_nouv_est - c2[1])**2))

        except:
            phi_est_list.append(phi_est_list[-1])
            x_est_list.append(x_est_list[-1])
            z_est_list.append(z_est_list[-1])
            err_pos_cumulee.append(err_pos_cumulee[-1])

        x_d.append(c2[0])
        z_d.append(c2[1])
        phi_reel_list.append(current_phi)
        
    res_err_pos[SCENARIO] = err_pos_cumulee
    res_err_phi[SCENARIO] = np.degrees(np.array(phi_reel_list)) - np.degrees(np.array(phi_est_list))
    res_traj_x[SCENARIO] = x_d
    res_traj_z[SCENARIO] = z_d
    res_est_x[SCENARIO] = x_est_list  # SAUVEGARDE DU X ESTIMÉ
    res_est_z[SCENARIO] = z_est_list  # SAUVEGARDE DU Z ESTIMÉ

print("Génération des graphiques de validation...")

# =====================================================
# 3. AFFICHAGE DES RÉSULTATS COMPARATIFS
# =====================================================
couleurs = ['b', 'g', 'r', 'c', 'm', 'y']
labels = [r'Ligne droite ($\alpha=0^\circ$)', r'Ligne droite ($\alpha=15^\circ$)', r'Ligne droite ($\alpha=30^\circ$)',
          r'Virage ($R=10m$)', r'Virage ($R=30m$)', r'Virage ($R=100m$)']

# --- Fenêtre 1 : Les 6 trajectoires ---
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111)
ax1.axhline(0, color='gray', linewidth=4, label="Mur d'inspection")

for i, sc in enumerate(scenarios):
    # Tracé de la VRAIE trajectoire : Très épaisse et semi-transparente (comme un surligneur)
    ax1.plot(res_traj_x[sc], res_traj_z[sc], color=couleurs[i], linewidth=6, alpha=0.4, label=labels[i])
    
    # Tracé de la trajectoire ESTIMÉE : Fine et noire par-dessus
    ax1.plot(res_est_x[sc], res_est_z[sc], color='black', linewidth=1.5, linestyle='--')

# Astuce pour la légende
ax1.plot([], [], color='black', linestyle='--', linewidth=1.5, label='Trajectoire Estimée')

ax1.set_title("1. Forme des 6 Trajectoires Testées (500 mètres de vol)", fontweight='bold')
ax1.set_xlabel("X (Avancement) [m]")
ax1.set_ylabel("Z (Profondeur) [m]")
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_aspect('equal', adjustable='box')

# Échelle millimétrique pour bien voir que l'intégration cumulée est parfaite
limite_y_pos = 0.000001  
limite_y_phi = 0.000001  

# --- Fenêtre 2 : Dérive de Position Cumulée (Sans Bruit) ---
fig2 = plt.figure(figsize=(12, 6))
ax2 = fig2.add_subplot(111)
for i, sc in enumerate(scenarios):
    distance_axe = np.linspace(0, 500, len(res_err_pos[sc]))
    ax2.plot(distance_axe, res_err_pos[sc], color=couleurs[i], linewidth=1.5, label=labels[i])
ax2.set_title("2. Erreur Odométrique de Position CUMULÉE (Validation mathématique sans bruit)", fontweight='bold')
ax2.set_xlabel("Distance parcourue [mètres]")
ax2.set_ylabel("Erreur Absolue [mètres]")
ax2.set_ylim(-limite_y_pos, limite_y_pos)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

# --- Fenêtre 3 : Dérive de Cap Cumulée (Sans Bruit) ---
fig3 = plt.figure(figsize=(12, 6))
ax3 = fig3.add_subplot(111)
for i, sc in enumerate(scenarios):
    distance_axe = np.linspace(0, 500, len(res_err_phi[sc]))
    ax3.plot(distance_axe, res_err_phi[sc], color=couleurs[i], linewidth=1.5, label=labels[i])
ax3.set_title("3. Erreur d'Orientation / Cap CUMULÉE (Validation mathématique sans bruit)", fontweight='bold')
ax3.set_xlabel("Distance parcourue [mètres]")
ax3.set_ylabel("Erreur [degrés]")
ax3.set_ylim(-limite_y_phi, limite_y_phi)
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.7)

plt.show()