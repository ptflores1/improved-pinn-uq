import pandas as pd
import numpy as np
from scipy.integrate import RK45

#alpha = 8.96e-2
beta = 3.12704e-4
gamma = 2.1899e-3
tau = 6.49e-4
#delta = 1.76e-2
theta = 4.4e-3
rho = 2.47e-4
#sigma = 6.7e-2
phi = 6.55e-3
omega = 1.5e-4

def generate_lab(alpha, delta, sigma, std=0.2):
    def func(t, Y):
        x, y, z = Y
        
        return [
            alpha*x - beta*x**2 - gamma*x*y**2 - tau*x*z**2,
            delta*y - theta*y**2 - rho*y*z,
            sigma*z - phi*z**2 - omega*x*z,
        ]
        
    initial_conditions = np.array([7.5797, 6.44, 1.9])

    rk4_sol = RK45(func, t0=0, y0=initial_conditions, t_bound=48, max_step=0.001)

    t_values = [0]
    x_values = [initial_conditions[0]]
    y_values = [initial_conditions[1]]
    z_values = [initial_conditions[2]]

    while rk4_sol.status != "finished":
        rk4_sol.step()
        
        t_values.append(rk4_sol.t)
        x_values.append(rk4_sol.y[0])
        y_values.append(rk4_sol.y[1])
        z_values.append(rk4_sol.y[2])

    rk4_t = np.array(t_values)
    rk4_x_points = np.array(x_values)
    rk4_y_points = np.array(y_values)
    rk4_z_points = np.array(z_values)
    
    t = np.linspace(0, 48, 9)
    x = np.interp(t, rk4_t, rk4_x_points) + np.random.normal(0, std, 9)
    y = np.interp(t, rk4_t, rk4_y_points) + np.random.normal(0, std, 9)
    z = np.interp(t, rk4_t, rk4_z_points) + np.random.normal(0, std, 9)

    with open("datasets/lab_syn.csv", "w") as f:
        f.write("t,x,y,z\n")
        for i in range(9):
            f.write(f"{t[i]},{x[i]},{y[i]},{z[i]}\n")
    return t, x, y, z

if __name__ == "__main__":
    t, x, y, z = generate_lab(0.0896, 0.0176, 0.067)

    

