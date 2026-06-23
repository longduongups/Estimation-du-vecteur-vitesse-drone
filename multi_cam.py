import numpy as np
import matplotlib.pyplot as plt

F = 0.010; B_TOT = 0.20; PIXEL_SIZE = 5e-6
D1_GT = 10.0; DT = 0.02; V_DRONE = 2.78
D_DRONE = V_DRONE * DT
I_FIXE = 0.005
N_TRIALS = 2000
BRUIT_AMPL = 0.5

# --- Modèle Physique ---
def calc_modele_2(i1, i2, D1, D2, f=F):
    denom = (i1*D1 + i2*D2)**2 + f**2 * (D2-D1)**2
    K = (D2**2 - D1**2)*f**2 + D2**2*i2**2 - D1**2*i1**2
    Tx = -(1.0/f) * (D2*i2 + D1*i1) * K / denom
    Tz = -(D2 - D1) * K / denom
    return Tx, Tz

X1 = -I_FIXE * D1_GT / F
X2 = X1 - D_DRONE
I2_REF = -X2 * F / D1_GT
TX_REF, TZ_REF = -D_DRONE, 0.0

# --- Estimateurs pour H2 (Capteurs partagés) ---
def fusion_H2(N, b0, D_true, di_capteurs_px, estimator='MC', f=F):
    n_trials = di_capteurs_px.shape[1]
    sum_w = 0
    sum_wD = np.zeros(n_trials)
    D_moy = np.zeros(n_trials)
    M = N * (N - 1) // 2
    
    for i in range(N):
        for j in range(i+1, N):
            B_ij = (j - i) * b0
            d_true = f * B_ij / D_true
            dd_ij = di_capteurs_px[i, :] - di_capteurs_px[j, :]
            d_obs = d_true + dd_ij * PIXEL_SIZE
            D_ij = f * B_ij / d_obs
            
            if estimator == 'MC':
                w = B_ij**2
                sum_w += w
                sum_wD += w * D_ij
            elif estimator == 'moy':
                D_moy += D_ij / M
                
    return sum_wD / sum_w if estimator == 'MC' else D_moy

def simulate_H2(N, estimator='MC', n=N_TRIALS):
    b0 = B_TOT / (N - 1)
    di_t1 = np.random.uniform(-BRUIT_AMPL, BRUIT_AMPL, (N, n))
    di_t2 = np.random.uniform(-BRUIT_AMPL, BRUIT_AMPL, (N, n))
    
    D1f = fusion_H2(N, b0, D1_GT, di_t1, estimator)
    D2f = fusion_H2(N, b0, D1_GT, di_t2, estimator)
    
    i1 = I_FIXE + di_t1[0, :] * PIXEL_SIZE
    i2 = I2_REF + di_t2[0, :] * PIXEL_SIZE
    tx, tz = calc_modele_2(i1, i2, D1f, D2f)
    return np.sqrt((tx-TX_REF)**2 + (tz-TZ_REF)**2)

# --- Estimateurs pour H1 (Paires indépendantes) ---
def fusion_H1(N, b0, D_true, dd_paires_px, estimator='MC', f=F):
    n_trials = dd_paires_px.shape[1]
    sum_w = 0
    sum_wD = np.zeros(n_trials)
    D_moy = np.zeros(n_trials)
    M = N * (N - 1) // 2
    idx = 0
    
    for i in range(N):
        for j in range(i+1, N):
            B_ij = (j - i) * b0
            d_true = f * B_ij / D_true
            d_obs = d_true + dd_paires_px[idx, :] * PIXEL_SIZE
            D_ij = f * B_ij / d_obs
            
            if estimator == 'MC':
                w = B_ij**2
                sum_w += w
                sum_wD += w * D_ij
            elif estimator == 'moy':
                D_moy += D_ij / M
            idx += 1
            
    return sum_wD / sum_w if estimator == 'MC' else D_moy

def simulate_H1(N, estimator='MC', n=N_TRIALS):
    b0 = B_TOT / (N - 1)
    M = N * (N - 1) // 2
    a_t1 = np.random.uniform(-BRUIT_AMPL, BRUIT_AMPL, (M, n))
    b_t1 = np.random.uniform(-BRUIT_AMPL, BRUIT_AMPL, (M, n))
    a_t2 = np.random.uniform(-BRUIT_AMPL, BRUIT_AMPL, (M, n))
    b_t2 = np.random.uniform(-BRUIT_AMPL, BRUIT_AMPL, (M, n))
    
    dd_t1 = a_t1 - b_t1
    dd_t2 = a_t2 - b_t2
    
    D1f = fusion_H1(N, b0, D1_GT, dd_t1, estimator)
    D2f = fusion_H1(N, b0, D1_GT, dd_t2, estimator)
    
    if N == 2:
        i1 = I_FIXE + a_t1[0, :] * PIXEL_SIZE
        i2 = I2_REF + a_t2[0, :] * PIXEL_SIZE
    else:
        i1 = I_FIXE
        i2 = I2_REF
        
    tx, tz = calc_modele_2(i1, i2, D1f, D2f)
    return np.sqrt((tx-TX_REF)**2 + (tz-TZ_REF)**2)

# --- Calcul des gains théoriques ---
def gain_H1_MC(N):
    return np.sqrt(N**2 * (N+1) / (12 * (N-1)))

def gain_H2_MC(N):
    return np.sqrt(N * (N+1) / (6 * (N-1)))

def gain_Moy(N):
    H2_n = sum(1/(x**2) for x in range(1, N))
    H_n = sum(1/x for x in range(1, N))
    return np.sqrt(N**2 / (4 * (N * H2_n - H_n)))

# --- Boucle Principale ---
N_vals = list(range(2, 17))

# Dictionnaires pour stocker les quantiles (Q25, Med, Q75)
res = { 
    'H1_Moy': {'q25': [], 'med': [], 'q75': []}, 
    'H2_Moy': {'q25': [], 'med': [], 'q75': []}, 
    'H1_MC': {'q25': [], 'med': [], 'q75': []}, 
    'H2_MC': {'q25': [], 'med': [], 'q75': []} 
}

print("Simulation en cours (2000 tirages)...")
for N in N_vals:
    # Simuler les 4 cas
    sim_H1_Moy = simulate_H1(N, 'moy')
    sim_H2_Moy = simulate_H2(N, 'moy')
    sim_H1_MC = simulate_H1(N, 'MC')
    sim_H2_MC = simulate_H2(N, 'MC')
    
    # Extraire et stocker Q25, Médiane, Q75
    res['H1_Moy']['q25'].append(np.percentile(sim_H1_Moy, 25))
    res['H1_Moy']['med'].append(np.median(sim_H1_Moy))
    res['H1_Moy']['q75'].append(np.percentile(sim_H1_Moy, 75))
    
    res['H2_Moy']['q25'].append(np.percentile(sim_H2_Moy, 25))
    res['H2_Moy']['med'].append(np.median(sim_H2_Moy))
    res['H2_Moy']['q75'].append(np.percentile(sim_H2_Moy, 75))
    
    res['H1_MC']['q25'].append(np.percentile(sim_H1_MC, 25))
    res['H1_MC']['med'].append(np.median(sim_H1_MC))
    res['H1_MC']['q75'].append(np.percentile(sim_H1_MC, 75))
    
    res['H2_MC']['q25'].append(np.percentile(sim_H2_MC, 25))
    res['H2_MC']['med'].append(np.median(sim_H2_MC))
    res['H2_MC']['q75'].append(np.percentile(sim_H2_MC, 75))

# Reference pour la théorie (N=2)
eps_ref = res['H2_MC']['med'][0] 

th_H1_MC  = [eps_ref / gain_H1_MC(N) for N in N_vals]
th_H2_MC  = [eps_ref / gain_H2_MC(N) for N in N_vals]
th_Moy    = [eps_ref / gain_Moy(N) for N in N_vals]

# --- Génération de la figure (4 panneaux) ---
fig, axs = plt.subplots(2, 2, figsize=(15, 11))

# Panneau 1 : H1 Moyenne
axs[0,0].fill_between(N_vals, res['H1_Moy']['q25'], res['H1_Moy']['q75'], color="#a501e1", alpha=0.2, label='Intervalle interquartile (Q1-Q3)')
axs[0,0].plot(N_vals, res['H1_Moy']['med'], 's-', color="#a501e1", lw=2.5, label='Simulé (Médiane)')
axs[0,0].plot(N_vals, th_Moy, 'k--', lw=2, label='Théorique')
axs[0,0].set_title(r"1. $H_1$ : Moyenne Arithmétique (Indépendant)")
axs[0,0].set_ylabel(r"Erreur de position $\varepsilon_{pos}$ [m]")

# Panneau 2 : H2 Moyenne
axs[0,1].fill_between(N_vals, res['H2_Moy']['q25'], res['H2_Moy']['q75'], color="#02ab05", alpha=0.2, label='Intervalle interquartile (Q1-Q3)')
axs[0,1].plot(N_vals, res['H2_Moy']['med'], 'o-', color="#02ab05", lw=2.5, label='Simulé (Médiane)')
axs[0,1].plot(N_vals, th_Moy, 'k--', lw=2, label='Théorique')
axs[0,1].set_title(r"2. $H_2$ : Moyenne Arithmétique (Partagé)")

# Panneau 3 : H1 Moindres Carrés
axs[1,0].fill_between(N_vals, res['H1_MC']['q25'], res['H1_MC']['q75'], color="#d62728", alpha=0.2, label='Intervalle interquartile (Q1-Q3)')
axs[1,0].plot(N_vals, res['H1_MC']['med'], 's-', color='#d62728', lw=2.5, label='Simulé (Médiane)')
axs[1,0].plot(N_vals, th_H1_MC, 'k--', lw=2, label='Théorique')
axs[1,0].set_title(r"3. $H_1$ : Inverse-Variance MC (Indépendant)")
axs[1,0].set_xlabel("Nombre de caméras (N)")
axs[1,0].set_ylabel(r"Erreur de position $\varepsilon_{pos}$ [m]")

# Panneau 4 : H2 Moindres Carrés
axs[1,1].fill_between(N_vals, res['H2_MC']['q25'], res['H2_MC']['q75'], color="#1f77b4", alpha=0.2, label='Intervalle interquartile (Q1-Q3)')
axs[1,1].plot(N_vals, res['H2_MC']['med'], 'o-', color='#1f77b4', lw=2.5, label='Simulé (Médiane)')
axs[1,1].plot(N_vals, th_H2_MC, 'k--', lw=2, label='Théorique')
axs[1,1].set_title(r"4. $H_2$ : Inverse-Variance MC (Partagé)")
axs[1,1].set_xlabel("Nombre de caméras (N)")

# Formatage commun
for ax in axs.flat:
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xticks(N_vals)
    ax.set_xlim(2, 16)
    ax.set_ylim(0, max(res['H1_Moy']['q75']) * 1.05)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('comparaison_4_cas_complets_marge.png', dpi=150)
print("Graphique sauvegardé sous 'comparaison_4_cas_complets_marge.png'")