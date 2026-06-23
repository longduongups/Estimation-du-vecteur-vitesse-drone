import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Configuration de la figure
fig, ax = plt.subplots(figsize=(14, 8))
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlim(-2, 14)
ax.set_ylim(-2, 12)
ax.set_xlabel("$X_m$ [m] (axe d'avancement)", fontsize=11)
ax.set_ylabel("$Z_m$ [m] (axe latéral, vers façade)", fontsize=11)
ax.set_aspect('equal', adjustable='box') # Garder les proportions réelles

# ----------------------------------------------------
# 1. REPÈRE MONDE (Om, Xm, Zm, Ym)
# ----------------------------------------------------
ax.arrow(0, 0, 1.5, 0, head_width=0.2, head_length=0.3, fc='red', ec='red', zorder=5)
ax.arrow(0, 0, 0, 1.5, head_width=0.2, head_length=0.3, fc='blue', ec='blue', zorder=5)
# Ym (point sortant)
ax.plot(0, 0, 'go', markersize=12, fillstyle='none', markeredgewidth=2, zorder=5)
ax.plot(0, 0, 'g.', markersize=6, zorder=5)

ax.text(1.7, 0, "$X_m$", color='red', fontsize=12, va='center', fontweight='bold')
ax.text(0, 1.7, "$Z_m$", color='blue', fontsize=12, ha='center', fontweight='bold')
ax.text(-0.3, 0.3, "$Y_m$", color='green', fontsize=12, ha='right')
ax.text(-0.3, -0.3, "$O_m$", color='black', fontsize=11, ha='right', va='top')
ax.text(-0.5, -1, "Repère monde", color='black', fontsize=9, fontstyle='italic', ha='center')

# ----------------------------------------------------
# 2. VÉRITÉ TERRAIN (Façade et points)
# ----------------------------------------------------
ax.axhline(0, color='gray', linewidth=3, alpha=0.5)
ax.text(6, -0.6, "Façade observée (plan vertical à $Z_m=0$)", color='gray', ha='center', fontstyle='italic', fontsize=10)

# Génération des points d'intérêt autour de la façade
np.random.seed(42)
x_pts = np.random.uniform(1.5, 11, 40)
z_pts = np.random.normal(0, 0.2, 40)
ax.scatter(x_pts, z_pts, color='steelblue', s=20, alpha=0.8, label="Points d'intérêt", zorder=4)

# Lignes de projection discrètes sur le mur
for x in np.arange(1.5, 11.5, 0.5):
    ax.plot([x, x], [-0.2, 0.2], color='gray', alpha=0.3, linewidth=1)

# ----------------------------------------------------
# 3. DRONE ET CAMÉRAS
# ----------------------------------------------------
drone_x = 4.5
drone_z = 10
drone_width = 3.5
drone_height = 0.8

# Corps du drone
rect = patches.Rectangle((drone_x - 0.78, drone_z - 0.4), drone_width, drone_height, 
                         linewidth=1, edgecolor='wheat', facecolor='wheat', alpha=0.7)
ax.add_patch(rect)
ax.text(drone_x + 2.0, drone_z + 1, "Drone", color='darkgoldenrod', fontweight='bold')

# Sens de vol
ax.arrow(drone_x + 4.5, drone_z + 0.5, 1, 0, head_width=0.2, head_length=0.3, fc='orange', ec='orange')
ax.text(drone_x + 5, drone_z + 0.8, "Sens de vol", color='orange', fontweight='bold', ha='center')

# Dessin des 4 caméras
num_cams = 4
cam_xs = np.linspace(drone_x, drone_x + 2, num_cams)
for i, cx in enumerate(cam_xs):
    # Corps de la caméra (rectangle)
    cam_body = patches.Rectangle((cx - 0.15, drone_z), 0.3, 0.25, facecolor='#1f618d', edgecolor='black', zorder=6)
    # Objectif de la caméra (trapèze pointant vers le bas)
    cam_lens = patches.Polygon([[cx - 0.1, drone_z], [cx + 0.1, drone_z], 
                                [cx + 0.2, drone_z - 0.3], [cx - 0.2, drone_z - 0.3]], 
                               facecolor='#1f618d', edgecolor='black', zorder=6)
    ax.add_patch(cam_body)
    ax.add_patch(cam_lens)
    ax.text(cx, drone_z + 0.4, f"$C_{i}$", ha='center', fontsize=10)

# ----------------------------------------------------
# 4. REPÈRE CAMÉRA ET COTATIONS
# ----------------------------------------------------


# Repère local déplacé sur la caméra C3 (cam_xs[-1])
ax.arrow(cam_xs[-1], drone_z, 1.8, 0, head_width=0.2, head_length=0.3, fc='darkred', ec='darkred', zorder=5)
ax.text(cam_xs[-1] + 2.0, drone_z, "$X_c$", color='darkred', fontsize=12, va='center', fontweight='bold')

ax.arrow(cam_xs[-1], drone_z, 0, -1.8, head_width=0.2, head_length=0.3, fc='darkblue', ec='darkblue', zorder=5)
ax.text(cam_xs[-1] + 0.2, drone_z - 2.0, "$Z_c$ (axe optique)", color='darkblue', fontsize=11, ha='left', fontweight='bold')

# Btot (Espacement des caméras) - maintenu sous les caméras
ax.annotate('', xy=(cam_xs[0], drone_z - 1.1), xytext=(cam_xs[-1], drone_z - 1.1), 
            arrowprops=dict(arrowstyle='<|-|>', color='black', lw=1.5))
ax.text((cam_xs[0]+cam_xs[-1])/2, drone_z - 1, "$b_{tot} = \mathbf{0,20\ m}$", ha='center', va='bottom', fontsize=10)

# Formule de l'espacement
ax.text((cam_xs[0]+cam_xs[1])/2, drone_z + 0.9, r"$b_k(N) = \frac{b_{tot}}{N-1}$", 
        color='purple', ha='center', fontsize=10)
ax.annotate('', xy=(cam_xs[0], drone_z + 0.8), xytext=(cam_xs[1], drone_z + 0.8), 
            arrowprops=dict(arrowstyle='<->', color='purple'))

# Distance au mur (Zm = 10m)
ax.annotate('', xy=(drone_x - 1.5, 0), xytext=(drone_x - 1.5, drone_z), 
            arrowprops=dict(arrowstyle='<|-|>', color='gray', linestyle='dashed'))
ax.text(drone_x - 1.7, drone_z / 2, "$Z_m \\approx 10$ m", color='gray', ha='center', va='center', rotation=90, fontsize=10)

# Légende
ax.legend(loc='lower right', framealpha=1)

# Sauvegarde et affichage
plt.tight_layout()
plt.savefig("schema_drone_cameras_corrige_v2.png", dpi=300)
