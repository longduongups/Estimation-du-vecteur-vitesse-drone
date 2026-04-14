import cv2
import numpy as np
import os

# Configuration des dossiers
folder = "img_test_sift"
paires = [('p00.png', 'p01.png'), ('p10.png', 'p11.png'), ('p20.png', 'p21.png')]

def process_and_visualize(p1_name, p2_name, idx):
    # Chemins complets
    path1 = os.path.join(folder, p1_name)
    path2 = os.path.join(folder, p2_name)

    # 1. Chargement
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    if img1 is None or img2 is None:
        print(f"Erreur de lecture : {p1_name} ou {p2_name}")
        return 0

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 2. SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 3. Matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 4. Filtre de Ratio (Lowe)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # 5. Création de l'image de visualisation
    # On dessine seulement les 50 meilleurs pour que ce soit lisible, ou 'good' pour tous
# Modifier le flag pour voir TOUS les points, même ceux sans liens
    img_matches = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        good[:100],
        None, 
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS # Affiche les cercles SIFT
    )

    # 6. Sauvegarde du résultat
    output_name = f"resultat_paire_{idx}.png"
    cv2.imwrite(os.path.join(folder, output_name), img_matches)
    
    return len(good)

# Lancement
for i, (p1, p2) in enumerate(paires):
    n = process_and_visualize(p1, p2, i)
    print(f"Paire {i} traitée : {n} points trouvés. Image sauvegardée.")