"""
Simulation de trajectoire 500 m — Corrigée pour le signe d'intégration.
Mode 1 : SANS bruit (vérification géométrique : estimée = vérité terrain)
Mode 2 : AVEC bruit (analyse de la dérive)

Correction clé :
  dXw = -Tx * cos(phi) + Tz * sin(phi)
  dZw = -Tx * sin(phi) - Tz * cos(phi)
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# Paramètres
BRUIT_PIXEL_AMPL = 0.5
TAILLE_PIXEL = 5e-6
FPS = 50
DT = 1.0 / FPS
V_MS = 10.0 / 3.6
FOCALE = 0.01
BASELINE = 0.20
DIST_OBJECTIF = 500.0
INTENSITE_DELTA = 0.005
K_POINTS = 1500

X_MIN, X_MAX = -10, DIST_OBJECTIF + 10
Y_MIN, Y_MAX = -5, 5
Z_FACADE_MIN, Z_FACADE_MAX = 8.0, 15.0

def simuler(seed, injecter_bruit, etape='C', retour_traj=False):
    rng = np.random.default_rng(seed)
    X_pts = rng.uniform(X_MIN, X_MAX, K_POINTS)
    Y_pts = rng.uniform(Y_MIN, Y_MAX, K_POINTS)
    Z_pts = rng.uniform(Z_FACADE_MIN, Z_FACADE_MAX, K_POINTS)
    
    x_d, z_d, phi = 0.0, 0.0, 0.0
    x_est, z_est = 0.0, 0.0
    traj_v = [(x_d, z_d)]; traj_e = [(x_est, z_est)]
    err_cum = [0.0]
    
    while x_d < DIST_OBJECTIF:
        v_rot = rng.uniform(-INTENSITE_DELTA, INTENSITE_DELTA)
        if z_d > 3 and v_rot > 0: v_rot = -abs(v_rot)
        elif z_d < -3 and v_rot < 0: v_rot = abs(v_rot)
        
        L = V_MS * DT
        if abs(v_rot) < 1e-9:
            dx, dz = L*np.cos(phi), L*np.sin(phi)
        else:
            R = L / v_rot
            dx = R * (np.sin(phi + v_rot) - np.sin(phi))
            dz = R * (np.cos(phi) - np.cos(phi + v_rot))
        
        x_d_n = x_d + dx; z_d_n = z_d + dz; phi_n = phi + v_rot
        
        dxw1 = X_pts - x_d;   dzw1 = Z_pts - z_d
        dxw2 = X_pts - x_d_n; dzw2 = Z_pts - z_d_n
        x1c =  dxw1*np.cos(phi)   + dzw1*np.sin(phi)
        z1c = -dxw1*np.sin(phi)   + dzw1*np.cos(phi)
        x2c =  dxw2*np.cos(phi_n) + dzw2*np.sin(phi_n)
        z2c = -dxw2*np.sin(phi_n) + dzw2*np.cos(phi_n)
        
        mask = (z1c > 1.0) & (z2c > 1.0)
        if mask.sum() < 50:
            x_d, z_d, phi = x_d_n, z_d_n, phi_n
            traj_v.append((x_d, z_d)); traj_e.append(traj_e[-1])
            err_cum.append(err_cum[-1])
            continue
        
        x1c, z1c, x2c, z2c = x1c[mask], z1c[mask], x2c[mask], z2c[mask]
        i1_v = -FOCALE * x1c / z1c
        i2_v = -FOCALE * x2c / z2c
        D1_v, D2_v = z1c, z2c
        Kact = mask.sum()
        
        if injecter_bruit:
            if etape in ('A', 'C'):
                bi1 = rng.uniform(-BRUIT_PIXEL_AMPL, BRUIT_PIXEL_AMPL, Kact)
                bi2 = rng.uniform(-BRUIT_PIXEL_AMPL, BRUIT_PIXEL_AMPL, Kact)
                i1 = i1_v + bi1 * TAILLE_PIXEL
                i2 = i2_v + bi2 * TAILLE_PIXEL
            else:
                i1, i2 = i1_v, i2_v
            if etape in ('B', 'C'):
                bd1 = rng.uniform(-BRUIT_PIXEL_AMPL, BRUIT_PIXEL_AMPL, Kact)
                bd2 = rng.uniform(-BRUIT_PIXEL_AMPL, BRUIT_PIXEL_AMPL, Kact)
                D1 = D1_v + (D1_v**2 / (FOCALE*BASELINE)) * bd1 * TAILLE_PIXEL
                D2 = D2_v + (D2_v**2 / (FOCALE*BASELINE)) * bd2 * TAILLE_PIXEL
            else:
                D1, D2 = D1_v, D2_v
        else:
            i1, i2, D1, D2 = i1_v, i2_v, D1_v, D2_v
        
        denom = (i1*D1 + i2*D2)**2 + FOCALE**2 * (D2-D1)**2
        valides = denom > 1e-12
        if valides.sum() < 20:
            x_d, z_d, phi = x_d_n, z_d_n, phi_n
            traj_v.append((x_d, z_d)); traj_e.append(traj_e[-1])
            err_cum.append(err_cum[-1])
            continue
        
        Kf = (D2**2 - D1**2)*FOCALE**2 + D2**2*i2**2 - D1**2*i1**2
        Tx_p = -(1.0/FOCALE) * (D2*i2 + D1*i1) * Kf / denom
        Tz_p = -(D2 - D1) * Kf / denom
        Tx_est = np.median(Tx_p[valides])
        Tz_est = np.median(Tz_p[valides])
        
        # *** CORRECTION DES SIGNES ***
        # Le Modèle 2 donne (Tx, Tz) = déplacement du point M dans R_C
        # Le déplacement du drone dans R_W est l'opposé après inverse-rotation
        dXw = -Tx_est * np.cos(phi) + Tz_est * np.sin(phi)
        dZw = -Tx_est * np.sin(phi) - Tz_est * np.cos(phi)
        
        x_est_n = x_est + dXw
        z_est_n = z_est + dZw
        err = np.sqrt((x_est_n - x_d_n)**2 + (z_est_n - z_d_n)**2)
        err_cum.append(err)
        x_d, z_d, phi = x_d_n, z_d_n, phi_n
        x_est, z_est = x_est_n, z_est_n
        traj_v.append((x_d, z_d)); traj_e.append((x_est, z_est))
    
    tv = np.array(traj_v); te = np.array(traj_e); ec = np.array(err_cum)
    if retour_traj:
        return tv, te, ec, (X_pts, Y_pts, Z_pts)
    return ec

# ===========================
# TEST 1 : SANS BRUIT
# ===========================
print("="*60)
print("TEST 1 : SANS BRUIT (vérification géométrique du Modèle 2)")
print("="*60)
tv0, te0, ec0, _ = simuler(seed=42, injecter_bruit=False, retour_traj=True)
print(f"Nb pas       : {len(tv0)}")
print(f"Erreur finale: {ec0[-1]:.3e} m")
print(f"Erreur max   : {ec0.max():.3e} m")
print(f"Erreur médiane: {np.median(ec0):.3e} m")

# ===========================
# TEST 2 : AVEC BRUIT (étape C)
# ===========================
print("\n"+"="*60)
print("TEST 2 : AVEC BRUIT (étape C — combiné i et D)")
print("="*60)
tv1, te1, ec1, (Xp, Yp, Zp) = simuler(seed=42, injecter_bruit=True, etape='C', retour_traj=True)
print(f"Nb pas       : {len(tv1)}")
print(f"Erreur finale: {ec1[-1]:.3f} m")
print(f"Erreur médiane: {np.median(ec1):.3f} m")

# ===========================
# FIGURES
# ===========================
# Figure A : Trajectoire sans bruit (2D)
fig_a, axA = plt.subplots(figsize=(12, 5))
axA.plot(tv0[:, 0], tv0[:, 1], color='red', lw=2.5, label='Vérité terrain')
axA.plot(te0[:, 0], te0[:, 1], color='black', lw=1.2, linestyle='--', label='Estimée (Modèle 2)')
axA.set_xlabel('X$_W$ [m]')
axA.set_ylabel('Z$_W$ [m]')
axA.set_title(f"SANS bruit — Vérification géométrique (erreur finale = {ec0[-1]:.2e} m)",
              fontweight='bold')
axA.legend()
axA.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('figure_sans_bruit_2D.png', dpi=120, bbox_inches='tight')
plt.close()

# Figure B : Erreur sans bruit
fig_b, axB = plt.subplots(figsize=(11, 4.5))
dist0 = np.arange(len(ec0)) * V_MS * DT
axB.plot(dist0, ec0, color='green', lw=1.5)
axB.set_xlabel('Distance parcourue [m]')
axB.set_ylabel(r'Erreur $\varepsilon_{pos}$ [m]')
axB.set_title(f"SANS bruit — Erreur résiduelle numérique (max = {ec0.max():.2e} m)",
              fontweight='bold')
axB.set_yscale('log')
axB.grid(True, linestyle='--', alpha=0.7, which='both')
plt.tight_layout()
plt.savefig('figure_sans_bruit_err.png', dpi=120, bbox_inches='tight')
plt.close()

# Figure C : Trajectoire avec bruit (2D)
fig_c, axC = plt.subplots(figsize=(12, 5))
axC.plot(tv1[:, 0], tv1[:, 1], color='red', lw=2, label='Vérité terrain')
axC.plot(te1[:, 0], te1[:, 1], color='black', lw=1.2, linestyle='--', label='Estimée (Modèle 2)')
axC.axhspan(Z_FACADE_MIN, Z_FACADE_MAX, color='gray', alpha=0.15, label='Zone des K points')
axC.set_xlabel('X$_W$ [m]')
axC.set_ylabel('Z$_W$ [m]')
axC.set_title(f"AVEC bruit (étape C, ±0,5 px) — Erreur finale = {ec1[-1]:.2f} m",
              fontweight='bold')
axC.legend()
axC.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('figure_avec_bruit_2D.png', dpi=120, bbox_inches='tight')
plt.close()

# Figure D : 3D protocole avec bruit
fig_d = plt.figure(figsize=(13, 7))
ax3d = fig_d.add_subplot(111, projection='3d')
idx_sub = np.random.choice(len(Xp), 400, replace=False)
ax3d.scatter(Xp[idx_sub], Zp[idx_sub], Yp[idx_sub],
             c='gray', s=8, alpha=0.4,
             label=f"Nuage de K = {K_POINTS} points (sous-échantillonné à 400)")
ax3d.plot(tv1[:, 0], tv1[:, 1], np.zeros_like(tv1[:, 0]),
          color='red', lw=2.5, label='Trajectoire vérité terrain')
ax3d.plot(te1[:, 0], te1[:, 1], np.zeros_like(te1[:, 0]),
          color='black', lw=1.2, linestyle='--', label='Trajectoire estimée par Modèle 2')
ax3d.set_xlabel('X$_W$ [m]')
ax3d.set_ylabel('Z$_W$ [m]')
ax3d.set_zlabel('Y$_W$ [m]')
ax3d.set_title(f"Représentation 3D du protocole — parcours 500 m, étape C (avec bruit)",
               fontweight='bold')
ax3d.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('figure_3D_complet.png', dpi=120, bbox_inches='tight')
plt.close()

print("\n✓ 4 figures sauvegardées")