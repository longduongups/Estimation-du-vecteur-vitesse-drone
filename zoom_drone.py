import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create the corrected figure
fig, ax = plt.subplots(figsize=(14, 8))
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_aspect('equal', adjustable='box')

# 4 cameras, true physical distances in meters (b_tot = 0.20 m, b_0 = 0.0667 m)
num_cams = 4
cam_xs = np.linspace(-0.10, 0.10, num_cams)  # C0 at -0.10, C3 at +0.10, Od at 0.0
drone_z = 0.0

# Drone body (from -0.15 m to +0.15 m)
drone_x_min = -0.15
drone_x_max =  0.15
rect = patches.Rectangle((drone_x_min, drone_z - 0.04),
                         drone_x_max - drone_x_min, 0.08,
                         linewidth=1.5, edgecolor='#8C7B65',
                         facecolor='#E8DCC4', alpha=0.8, zorder=2)
ax.add_patch(rect)

# Flight direction arrow
ax.arrow(drone_x_max + 0.01, drone_z + 0.08, 0.05, 0,
         head_width=0.015, head_length=0.015, fc='#D95F02', ec='#D95F02', lw=2, zorder=5)
ax.text(drone_x_max + 0.035, drone_z + 0.11, "Sens de vol",
        color='#D95F02', fontweight='bold', ha='center', fontsize=12)

# Cameras drawing
cam_width = 0.018
cam_height = 0.025
lens_width = 0.012
lens_length = 0.020

for i, cx in enumerate(cam_xs):
    # Camera body
    cam_body = patches.Rectangle((cx - cam_width/2, drone_z - cam_height/2), cam_width, cam_height,
                                 facecolor='#1f618d', edgecolor='black', lw=1.2, zorder=6)
    # Camera lens pointing downward (+Zd)
    cam_lens = patches.Polygon([[cx - lens_width/2, drone_z - cam_height/2],
                                [cx + lens_width/2, drone_z - cam_height/2],
                                [cx + lens_width, drone_z - cam_height/2 - lens_length],
                                [cx - lens_width, drone_z - cam_height/2 - lens_length]],
                               facecolor='#2980b9', edgecolor='black', lw=1.2, zorder=6)
    ax.add_patch(cam_body)
    ax.add_patch(cam_lens)
    
    # Label C_i
    ax.text(cx, drone_z + 0.05, f"$C_{i}$", ha='center', fontsize=13, fontweight='bold', color='#1a1a1a')
    
    # Center optical point marker
    ax.plot(cx, drone_z, marker='.', color='white', markersize=6, zorder=7)

# ==========================================================
# REPÈRE R_d : position physique (isobarycentre, Od en 0.0)
# ==========================================================
Od_x = 0.0
Od_z = drone_z

ax.plot(Od_x, Od_z, marker='o', color='purple', markersize=10,
        markeredgecolor='black', markeredgewidth=1.5, zorder=8)
ax.text(Od_x , Od_z + 0.01, "$O_d$", color='purple',
        fontsize=14, ha='right', fontweight='bold', zorder=8)

# X_d horizontal arrow
ax.arrow(Od_x, Od_z, 0.07, 0, head_width=0.012, head_length=0.012,
         fc='purple', ec='purple', zorder=7, linewidth=2.5)
ax.text(Od_x + 0.075, Od_z + 0.012, "$X_d$", color='purple', fontsize=14,
        va='center', fontweight='bold')

# Z_d vertical arrow
ax.arrow(Od_x, Od_z, 0, -0.07, head_width=0.012, head_length=0.012,
         fc='purple', ec='purple', zorder=7, linewidth=2.5)
ax.text(Od_x + 0.015, Od_z - 0.075, "$Z_d$", color='purple',
        fontsize=14, ha='left', fontweight='bold')

# ==========================================================
# REPÈRE R_{c,3} : position physique (sur le centre optique de C_3)
# ==========================================================
Oc3_x = cam_xs[3]
Oc3_z = drone_z

ax.plot(Oc3_x, Oc3_z, marker='o', color='darkred', markersize=9,
        markeredgecolor='black', markeredgewidth=1.5, zorder=8)
ax.text(Oc3_x + 0.012, Oc3_z + 0.01, "$O_{c,3}$", color='darkred',
        fontsize=13, ha='left', fontweight='bold', zorder=8)

# X_{c,3} horizontal arrow
ax.arrow(Oc3_x, Oc3_z, 0.05, 0, head_width=0.012, head_length=0.012,
         fc='darkred', ec='darkred', zorder=7, linewidth=2)
ax.text(Oc3_x + 0.065, Oc3_z, "$X_{c,3}$", color='darkred',
        fontsize=13, va='center', fontweight='bold')

# Z_{c,3} vertical arrow (optical axis)
ax.arrow(Oc3_x, Oc3_z, 0, -0.07, head_width=0.012, head_length=0.012,
         fc='darkred', ec='darkred', zorder=7, linewidth=2)
ax.text(Oc3_x + 0.012, Oc3_z - 0.075, "$Z_{c,3}$ (axe optique)",
        color='darkred', fontsize=12, ha='left', fontweight='bold')

# ==========================================================
# COTATIONS (Avec lignes de rappel verticales claires)
# ==========================================================

# Lignes de rappel verticales pour b_tot (entre C0 et C3)
ax.plot([cam_xs[0], cam_xs[0]], [drone_z - cam_height, drone_z - 0.12], color='black', linestyle=':', lw=1.2)
ax.plot([cam_xs[3], cam_xs[3]], [drone_z - cam_height, drone_z - 0.12], color='black', linestyle=':', lw=1.2)

# Flèche b_tot
ax.annotate('', xy=(cam_xs[0], drone_z - 0.11), xytext=(cam_xs[3], drone_z - 0.11),
            arrowprops=dict(arrowstyle='<|-|>', color='black', lw=2.0, mutation_scale=15))
ax.text((cam_xs[0] + cam_xs[3])/2, drone_z - 0.125,
        r"$b_{tot} = 0{,}20\text{ m}$ (entre centres optiques $C_0$ et $C_3$)",
        ha='center', va='top', fontsize=12, fontweight='bold', color='black',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

# Lignes de rappel verticales pour b_0 (entre C0 et C1)
ax.plot([cam_xs[0], cam_xs[0]], [drone_z + cam_height, drone_z + 0.15], color='#556B2F', linestyle=':', lw=1.2)
ax.plot([cam_xs[1], cam_xs[1]], [drone_z + cam_height, drone_z + 0.15], color='#556B2F', linestyle=':', lw=1.2)

# Flèche b_0
ax.annotate('', xy=(cam_xs[0], drone_z + 0.14), xytext=(cam_xs[1], drone_z + 0.14),
            arrowprops=dict(arrowstyle='<|-|>', color='#556B2F', lw=1.8, mutation_scale=12))
ax.text((cam_xs[0] + cam_xs[1])/2, drone_z + 0.155,
        r"$b_0(N) = \dfrac{b_{tot}}{N-1} = 6{,}67\text{ cm}$" + "\n" + r"(entre $C_0$ et $C_1$)",
        color='#556B2F', ha='center', va='bottom', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

# ==========================================================
# LIMITES ET LABELS D'AXES AVEC UNITÉS EXPLICITES
# ==========================================================
ax.set_xlim(-0.22, 0.25)
ax.set_ylim(-0.18, 0.25)

# UNITES SUR LES AXES (Demande de Nima)
ax.set_xlabel("Coordonnée latérale $X_d$ [mètres]", fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel("Coordonnée longitudinale $Z_d$ [mètres]", fontsize=13, fontweight='bold', labelpad=10)
ax.set_title("Représentation physique à l'échelle du dispositif multi-caméras ($N=4, b_{tot}=0{,}20\\text{ m}$)",
             fontsize=14, fontweight='bold', pad=15)

# Formatage des graduations en mètres avec 2 décimales
ax.xaxis.set_major_formatter('{x:.2f} m')
ax.yaxis.set_major_formatter('{x:.2f} m')

plt.tight_layout()
plt.savefig("figure3_corrige.png", dpi=300)
print("Figure corrigée générée avec succès : figure3_corrige_nima.png")