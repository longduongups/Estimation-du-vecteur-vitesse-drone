import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np

def dessiner_cas(ax, theta_deg, titre):
    # Configuration de la figure
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(-2, 14)
    ax.set_ylim(-2, 14)
    ax.set_xlabel("$X_m$ [m] (axe d'avancement)", fontsize=11)
    ax.set_ylabel("$Z_m$ [m] (axe latéral, vers façade)", fontsize=11)
    ax.set_aspect('equal', adjustable='box') # Garder les proportions réelles
    ax.set_title(titre, fontsize=14, fontweight='bold', pad=15)

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
    ax.text(6, -0.8, "Façade observée (plan vertical à $Z_m=0$)", color='gray', ha='center', fontstyle='italic', fontsize=10)

    # Génération des points d'intérêt autour de la façade
    np.random.seed(42)
    x_pts = np.random.uniform(1.5, 11, 40)
    z_pts = np.random.normal(0, 0.2, 40)
    ax.scatter(x_pts, z_pts, color='steelblue', s=20, alpha=0.8, label="Points d'intérêt", zorder=4)

    # Lignes de projection discrètes sur le mur
    for x in np.arange(1.5, 11.5, 0.5):
        ax.plot([x, x], [-0.2, 0.2], color='gray', alpha=0.3, linewidth=1)

    # ----------------------------------------------------
    # 3. DRONE (Bloc Simple)
    # ----------------------------------------------------
    drone_x = 4.5
    drone_z = 10
    drone_width = 2.8
    drone_height = 0.5
    theta_rad = np.radians(theta_deg)

    # Matrice de rotation autour du centre du drone
    trans = transforms.Affine2D().rotate_deg_around(drone_x, drone_z, theta_deg) + ax.transData

    # Corps du drone
    rect = patches.Rectangle((drone_x - drone_width/2, drone_z - drone_height/2), 
                             drone_width, drone_height, 
                             linewidth=1.5, edgecolor='black', facecolor='wheat', alpha=0.9,
                             transform=trans, zorder=6)
    ax.add_patch(rect)
    
    # Triangle avant pour indiquer le nez du drone et le sens de vol
    front = patches.Polygon([[drone_x + drone_width/2, drone_z - 0.3],
                             [drone_x + drone_width/2, drone_z + 0.3],
                             [drone_x + drone_width/2 + 0.5, drone_z]],
                            color='orange', transform=trans, zorder=6)
    ax.add_patch(front)

    ax.text(drone_x - 0.5, drone_z + 1.2, "Drone", color='darkgoldenrod', fontweight='bold', transform=trans)

    # ----------------------------------------------------
    # 4. REPÈRE DRONE (Xd, Zd)
    # ----------------------------------------------------
    xd_len = 3.0
    zd_len = 2.5
    
    # Projection trigonométrique du repère local
    xd_x = xd_len * np.cos(theta_rad)
    xd_z = xd_len * np.sin(theta_rad)
    
    # Zd pointe vers la façade (à -90° de Xd dans ce contexte)
    zd_x = zd_len * np.cos(theta_rad - np.pi/2)
    zd_z = zd_len * np.sin(theta_rad - np.pi/2)
    
    # Flèches du repère Drone
    ax.arrow(drone_x, drone_z, xd_x, xd_z, head_width=0.2, head_length=0.3, fc='darkred', ec='darkred', zorder=7)
    ax.arrow(drone_x, drone_z, zd_x, zd_z, head_width=0.2, head_length=0.3, fc='darkblue', ec='darkblue', zorder=7)
    
    ax.text(drone_x + xd_x * 1.15 + 0.5, drone_z + xd_z * 1.15, "$X_d$", color='darkred', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(drone_x + zd_x * 1.15 + 0.5, drone_z + zd_z * 1.1 -0.5, "$Z_d$ (axe optique)", color='darkblue', fontsize=11, fontweight='bold', ha='center', va='center')

    # Origine du drone Od
    ax.plot(drone_x, drone_z, 'ko', zorder=8)
    ax.text(drone_x - 0.4, drone_z + 0.4, "$O_d$", fontsize=12, fontweight='bold')

    # ----------------------------------------------------
    # 5. ILLUSTRATION DE L'ANGLE DE CAP (Theta)
    # ----------------------------------------------------
    # Ligne de référence horizontale en pointillés
    ax.plot([drone_x, drone_x + 4], [drone_z, drone_z], color='black', linestyle=':', linewidth=1.5)
    
    arc_radius = 2.0
    arc_x = drone_x + arc_radius * np.cos(theta_rad)
    arc_y = drone_z + arc_radius * np.sin(theta_rad)
    
    rad_curve = 0.2 if theta_deg > 0 else -0.2
    arc_color = 'forestgreen' if theta_deg > 0 else 'crimson'
    label_theta = r'$\theta > 0$' if theta_deg > 0 else r'$\theta < 0$'

    # Arc de cercle illustrant l'angle
    arc = patches.FancyArrowPatch((drone_x + arc_radius, drone_z), (arc_x, arc_y),
                                  connectionstyle=f"arc3,rad={rad_curve}",
                                  color=arc_color, arrowstyle="-|>", mutation_scale=15, linewidth=2, zorder=8)
    ax.add_patch(arc)
    
    ax.text(drone_x + arc_radius * 1.3 * np.cos(theta_rad / 2) + 1, 
            drone_z + arc_radius * 1.3 * np.sin(theta_rad / 2), 
            label_theta, color=arc_color, fontsize=12, fontweight='bold', ha='center', va='center')

    # Cotation distance Zm
    ax.annotate('', xy=(drone_x - 3, 0), xytext=(drone_x - 3, 10), 
                arrowprops=dict(arrowstyle='<|-|>', color='gray', linestyle='dashed'))
    ax.text(drone_x - 3.2, 5, "$Z_m \\approx 10$ m", color='gray', ha='center', va='center', rotation=90, fontsize=10)

    # Légende
    ax.legend(loc='lower right', framealpha=1)

# ==========================================
# Création de la figure globale avec 2 sous-graphiques
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

# Cas 1 : Cap positif
dessiner_cas(ax1, 25, "Sens trigonométrique : Cap positif")

# Cas 2 : Cap négatif
dessiner_cas(ax2, -25, "Sens horaire : Cap négatif")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("convention_cap_drone.png", dpi=300)
#plt.show()