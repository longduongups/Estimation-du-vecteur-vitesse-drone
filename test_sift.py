import cv2
import numpy as np
import os

folder = "img_test_sift"
paires = [('p00.png', 'p01.png'), ('p10.png', 'p11.png'), ('p20.png', 'p21.png'), ('p30.png', 'p31.png'), ('p40.png', 'p41.png')]

# Dictionnaire des algorithmes à tester
algorithmes = {
    "SIFT": cv2.SIFT_create(),
    "AKAZE": cv2.AKAZE_create(),
    "ORB": cv2.ORB_create(nfeatures=5000) # On force ORB à chercher jusqu'à 5000 points
}

def process_pair(p1_name, p2_name, algo_name, algo):
    img1 = cv2.imread(os.path.join(folder, p1_name), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(folder, p2_name), cv2.IMREAD_GRAYSCALE)
    
    # Détection et calcul des descripteurs
    kp1, des1 = algo.detectAndCompute(img1, None)
    kp2, des2 = algo.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return 0

    # Matching (NORM_HAMMING pour ORB/AKAZE, NORM_L2 pour SIFT)
    norm = cv2.NORM_HAMMING if algo_name in ["ORB", "AKAZE"] else cv2.NORM_L2
    bf = cv2.BFMatcher(norm)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Filtre de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    return len(good_matches)

# Lancement des tests
print(f"{'Algo':<10} | {'Paire 0':<10} | {'Paire 1':<10} | {'Paire 2':<10} | {'Paire 3':<10} | {'Paire 4':<10} | {'Moyenne K'}")
print("-" * 80)

for algo_name, algo in algorithmes.items():
    resultats = []
    for p1, p2 in paires:
        n_matches = process_pair(p1, p2, algo_name, algo)
        resultats.append(n_matches)
    
    moyenne = int(np.mean(resultats))
    print(f"{algo_name:<10} | {resultats[0]:<10} | {resultats[1]:<10} | {resultats[2]:<10} | {resultats[3]:<10} | {resultats[4]:<10} | {moyenne}")