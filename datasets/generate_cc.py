import pandas as pd
import numpy as np
from scipy.integrate import RK45

def generate_hubble_lcdm(H_0, Om_m_0):
    cc_df = pd.read_csv('datasets/cosmic_chronometers.csv')
    cc_z = np.array(cc_df["z"].values).ravel()
    cc_std = np.array(cc_df["std"].values).ravel()

    c = (Om_m_0 * 3) ** (1 / 3)
    x = ((1 + cc_z) * c) ** 3 / 3

    H = H_0 * ((x + 1 - Om_m_0) ** (1/2))

    with open('datasets/cc_lcdm_syn.csv', 'w') as f:
        f.write("z,h,std\n")
        for i in range(len(cc_z)):
            f.write(f"{cc_z[i]},{H[i]},{cc_std[i]}\n")
    return np.stack(sorted(np.stack([cc_z, H, cc_std]).T, key=lambda x: x[0]))
    

def generate_hubble_cpl(H_0, Om_m_0, w_0, w_1):
    cc_df = pd.read_csv('datasets/cosmic_chronometers.csv')
    cc_z = np.array(cc_df["z"].values).ravel()
    cc_std = np.array(cc_df["std"].values).ravel()

    x = np.exp(3*(np.log(cc_z+1) + w_1*(np.log(cc_z+1) + 1/(1+cc_z) - 1) + w_0*np.log(cc_z+1)))
    H = H_0*((Om_m_0*((1+cc_z)**3) + (1-Om_m_0)*x) ** (1/2))

    with open('datasets/cc_cpl_syn.csv', 'w') as f:
        f.write("z,h,std\n")
        for i in range(len(cc_z)):
            f.write(f"{cc_z[i]},{H[i]},{cc_std[i]}\n")
    return np.stack(sorted(np.stack([cc_z, H, cc_std]).T, key=lambda x: x[0]))

def generate_hubble_quint(H_0, Om_m_0, lam_prime):
    cc_df = pd.read_csv('datasets/cosmic_chronometers.csv')
    cc_z = np.array(cc_df["z"].values).ravel()
    cc_std = np.array(cc_df["std"].values).ravel()

    z_0 = 10.0
    N_prime_0 = 0
    lam_max = 3.0
    N_0_abs = np.abs(np.log(1/(1 + z_0)))

    def func(N_prime, Y):
        x, y = Y
        return [
            N_0_abs*(-3*x + lam_max*(np.sqrt(6)/2)*lam_prime*(y ** 2) + (3/2)*x*(1 + (x**2) - (y**2))),
            N_0_abs*(-lam_max*(np.sqrt(6)/2)*lam_prime*(y * x) + (3/2)*y*(1 + (x**2) - (y**2)))]
    
    initial_conditions = np.array([0, ((1 - Om_m_0)/(Om_m_0*(np.e**(-3*(N_prime_0 - 1)*N_0_abs)) + 1 - Om_m_0)) ** (1/2)])
    rk4_sol = RK45(func, t0=N_prime_0, y0=initial_conditions, t_bound=1, max_step=0.001)
    t_values = [N_prime_0]
    x_values = [initial_conditions[0]]
    y_values = [initial_conditions[1]]

    while rk4_sol.status != "finished":
        rk4_sol.step()
        t_values.append(rk4_sol.t)
        x_values.append(rk4_sol.y[0])
        y_values.append(rk4_sol.y[1])
    rk4_t = np.array(t_values)
    rk4_x_points = np.array(x_values)
    rk4_y_points = np.array(y_values)

    Ns = np.log(1/(1 + cc_z))
    N_primes = (Ns/N_0_abs) + 1

    x = np.interp(N_primes, rk4_t, rk4_x_points)
    y = np.interp(N_primes, rk4_t, rk4_y_points)
    H = H_0*((Om_m_0 * ((1 + cc_z) ** 3))/(1 - (x ** 2) - (y ** 2))) ** (1/2)

    with open('datasets/cc_quint_syn.csv', 'w') as f:
        f.write("z,h,std\n")
        for i in range(len(cc_z)):
            f.write(f"{cc_z[i]},{H[i]},{cc_std[i]}\n")
    return np.stack(sorted(np.stack([cc_z, H, cc_std]).T, key=lambda x: x[0]))

def generate_hubble_hs(H_0, Om_m_0, b_prime):
    cc_df = pd.read_csv('datasets/cosmic_chronometers.csv')
    cc_z = np.array(cc_df["z"].values).ravel()
    cc_std = np.array(cc_df["std"].values).ravel()

    z_prime_0 = 0.0
    z_0 = 10.0
    b_max = 5.0

    def func(z_prime, Y):
        x, y, v, Om, r_prime = Y
        b = b_max * b_prime
        z = z_0 * (1 - z_prime)
        r = np.exp(r_prime)
        Gamma = (r + b)*(((r + b)**2) - 2*b)/(4*r*b)

        return [
            -1*(-Om - 2*v + x + 4*y + x*v + x**2)/(z + 1)*z_0,
            (v*x*Gamma - x*y + 4*y - 2*y*v)/(z + 1)*z_0,
            v*(x*Gamma + 4 - 2*v)/(z + 1)*z_0,
            -Om*(-1 + 2*v + x)/(z + 1)*z_0,
            (Gamma*x)/(z + 1)*z_0,
        ]
        
    initial_conditions = np.array(
        [
            0,
            (Om_m_0*((1 + z_0)**3) + 2*(1 - Om_m_0))/(2*(Om_m_0*((1 + z_0)**3) + 1 - Om_m_0)),
            (Om_m_0*((1 + z_0)**3) + 4*(1 - Om_m_0))/(2*(Om_m_0*((1 + z_0)**3) + 1 - Om_m_0)),
            Om_m_0*((1 + z_0)**3)/((Om_m_0*((1 + z_0)**3) + 1 - Om_m_0)),
            np.log((Om_m_0*((1 + z_0)**3) + 4*(1 - Om_m_0))/(1 - Om_m_0))
        ]
    )

    rk4_sol = RK45(func, t0=z_prime_0, y0=initial_conditions, t_bound=1, max_step=0.001)
    t_values = [z_prime_0]
    x_values = [initial_conditions[0]]
    y_values = [initial_conditions[1]]
    v_values = [initial_conditions[2]]
    Om_values = [initial_conditions[3]]
    r_values = [initial_conditions[4]]

    while rk4_sol.status != "finished":
        rk4_sol.step()
        
        t_values.append(rk4_sol.t)
        x_values.append(rk4_sol.y[0])
        y_values.append(rk4_sol.y[1])
        v_values.append(rk4_sol.y[2])
        Om_values.append(rk4_sol.y[3])
        r_values.append(rk4_sol.y[4])

    rk4_t = np.array(t_values)
    rk4_x_points = np.array(x_values)
    rk4_y_points = np.array(y_values)
    rk4_v_points = np.array(v_values)
    rk4_Om_points = np.array(Om_values)
    rk4_r_points = np.array(r_values)

    zs_prime = 1 - (cc_z/z_0)
    r_prime = np.interp(zs_prime, rk4_t, rk4_r_points)
    r = np.exp(r_prime)
    v = np.interp(zs_prime, rk4_t, rk4_v_points)
    H = H_0*np.sqrt(r*(1 - Om_m_0)/(2*v))

    with open('datasets/cc_hs_syn.csv', 'w') as f:
        f.write("z,h,std\n")
        for i in range(len(cc_z)):
            f.write(f"{cc_z[i]},{H[i]},{cc_std[i]}\n")
    return np.stack(sorted(np.stack([cc_z, H, cc_std]).T, key=lambda x: x[0]))


if __name__ == "__main__":
    H_0 = 65
    Om_m_0 = 0.25

    lcdm = generate_hubble_lcdm(H_0, Om_m_0).T
    cpl = generate_hubble_cpl(H_0, Om_m_0, -1, -2.5).T
    quint = generate_hubble_quint(H_0, Om_m_0, 0.5).T
    hs = generate_hubble_hs(H_0, Om_m_0, 0.25).T


