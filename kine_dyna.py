import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time
from scipy.integrate import solve_ivp



#Initializing variables

i_goal = 0
time_start = None
time_prev = None
goal_threshold = 0.5




class Kinematics:
    def __init__(self):

        self.n    = 3               #Number of segments for the SBA
        self.d    = 4.7               #Geometric parameter
        self.gain = 1              #Gain for the U_ell
        self.r    = 5.6
    
    def volume_to_length(self, v1, v2, v3):
        l1, l2, l3 = 45.67*v1,  45.67*v2,  45.67*v3

        return l1, l2, l3

    def lengths_to_volumes(self, ells, base_height):
        volumes = 100/4667 * (ells - base_height)
        return volumes*1000

    def Rz(self, phi):
        T = np.array([[np.cos(phi), -np.sin(phi), 0],
                      [np.sin(phi),  np.cos(phi), 0],
                      [          0,            0, 1]])
        return T

    def Ry(self, theta):
        T = np.array([[ np.cos(theta), 0, np.sin(theta)],
                      [             0, 1,             0],
                      [-np.sin(theta), 0, np.cos(theta)]])
        return T

    def Rx(self, theta):
        T = np.array([[1,             0,              0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta),  np.cos(theta)]])
        return T


    def Link_function(self, qi): #Method to calculate the length of each prismatic links of each segment
        theta_i = qi[1,0]        #based on current configuration variables
        S_i     = qi[2,0]        #also evaluating the variation of these link lengths w.r.t change in theta, and arc length
                                 #The link length does not depend upon the out of the plane angle (phi)
        if theta_i < 1e-3:
            li = S_i
            dl_theta = 0
            dl_S = 1
        else:
            li = 2*S_i*np.sin(theta_i/2)/theta_i
            dl_theta = 2*S_i*(np.cos(theta_i/2)/(2*theta_i) - np.sin(theta_i/2)/(theta_i**2))
            dl_S = 2*np.sin(theta_i/2)/theta_i

        return li, dl_theta, dl_S

    def single_map(self, qi):
        #Method helps to evaluate the Xi vector along with the Jacobian_xi_w.r.t q for each segment
        phi   = qi[0,0]
        theta = qi[1,0]

        li, dl_theta, dl_S = self.Link_function(qi)

        xi = np.array([[phi], [theta/2], [li], [theta/2], [-phi]])

        Jmi = np.array([[1,0,0], [0, 0.5, 0], [0,dl_theta,dl_S], [0,0.5,0],[-1,0,0]])

        return xi, Jmi

    def Build_XI(self, q):
        #Method helps to build the final Xi which is a (3*n,1) vector and J_xi_q which is a (5*n, 3*n)
        xi = np.zeros((5*self.n,1))
        Jm = np.zeros((5*self.n, 3*self.n))

        for i in range(self.n):
            rows = slice(i*5, i*5 + 5)
            qi = q[3*i : 3*i + 3].reshape(3,1)
            xi_block, Jmi = self.single_map(qi)

            xi[rows] = xi_block
            Jm[rows, 3*i : 3*i + 3] = Jmi
        return xi, Jm

    def Augmented_DH(self, q):
        #Function to derive the DH Table based on our revolute and prismatic joints
        DH_aug = np.zeros((5*self.n, 4))
        Xi, _ = self.Build_XI(q)

        for i in range(self.n):
            # if i == (self.n - 1):
            #     base = i*5
            #     xi = Xi[base: base + 5, 0]
            #     DH_aug[base + 0, :] = [xi[0],    0,    0, -np.pi/2]
            #     DH_aug[base + 1, :] = [xi[1],    0,    0,  np.pi/2]
            #     DH_aug[base + 2, :] = [0,       xi[2], 0, -np.pi/2]
            #     DH_aug[base + 3, :] = [xi[3],    0,    0,  np.pi/2]
            #     DH_aug[base + 4, :] = [xi[4],    r,    0,        0]
            # else:
            base = i*5
            xi = Xi[base: base + 5, 0]
            DH_aug[base + 0, :] = [xi[0],    0,    0, -np.pi/2]
            DH_aug[base + 1, :] = [xi[1],    0,    0,  np.pi/2]
            DH_aug[base + 2, :] = [0,       xi[2], 0, -np.pi/2]
            DH_aug[base + 3, :] = [xi[3],    0,    0,  np.pi/2]
            DH_aug[base + 4, :] = [xi[4],    0,    0,        0]

        return DH_aug

    def DH_transform(self, theta, d, a, alpha):
        #Basic 4x4 Tranformation matrix
        T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                      [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                      [0,                            np.sin(alpha),                np.cos(alpha),               d],
                      [0,                                     0,                               0,               1]])

        return T

    def Compute_Augmented_Jacobian(self, dh_aug):
        # This function allows us to evaluate the Jacobian for the RRPRR model. The jacobian size is
        # 3 x 5*n, (here n is number of segments and 5 basically is the number of joints per segment
        # in our case we have 5 joints per segment)

        N = 5*self.n
        T = np.eye(4)
        p = np.zeros((3, N+1))
        z = np.tile(np.array([[0.],[0.],[1.]]), (1 , N+1))

        new_dh_aug = dh_aug
        new_dh_aug[-1,1] = self.r

        for k in range(N):
            theta, d, a, alpha = new_dh_aug[k]
            T     = T@self.DH_transform(theta,d,a,alpha)
            p[:, k+1] = T[0:3, 3]
            z[:, k+1] = T[0:3, 2]

        p_tip = p[:, -1]
        Jxi   = np.zeros((3, N))

        for k in range(N):
            idx = k%5
            if idx == 2:
                Jxi[:,k] = z[0:3, k]
            else:
                Jxi[:, k] = np.cross(z[0:3,k], (p_tip - p[:,k]))

        return Jxi

    def ComputeJacobian(self, q):
        global r
        # Evaluating the final jacobian for the soft robot J(q). Its size is 3 x 3*n
        dh_aug = self.Augmented_DH(q)
        Jxi    = self.Compute_Augmented_Jacobian(dh_aug)
        _, Jm  = self.Build_XI(q)

        J_q    = Jxi @ Jm

        return J_q

    def Model(self, t, q, f_ext_fun, K,D, q_ref):
        # Our basic dynamic model Kq + Dq_dot = J^T.F_ext
        J  = self.ComputeJacobian(q, )
        f  = f_ext_fun(t)
        dq = np.linalg.solve(D, J.T.dot(f) - K.dot(q - q_ref))
        return dq
    
    def new_Model(self, t, q, f_ext_fun, K,D, q_ref, q_ref_dot):
        # Our basic dynamic model Kq + Dq_dot = J^T.F_ext
        J  = self.ComputeJacobian(q, )
        f  = f_ext_fun(t)
        dq = q_ref_dot + np.linalg.solve(D, J.T.dot(f) - K.dot(q - q_ref))
        return dq

    def q_dynamics(self, q0, f_ext_fun, t0, dt, K, D):
        # Function to evaluate the configuration variable (q) at t = t + dt under the action of
        # external force acting at the tip
        q0 = np.asarray(q0).reshape(-1)
        t_span = (t0, t0 + dt)

        sol = solve_ivp(fun = lambda t, q: self.Model(t, q, f_ext_fun, K,D, q0),
                        t_span = t_span,
                        y0 = q0,
                        method= 'RK45',
                        t_eval = [t0 + dt])

        q_next = sol.y[:,-1]

        return q_next.reshape(-1,1)
    
    def q_dynamics_new(self, q_init, q_ref, q_ref_dot, f_ext_fun, t0, dt, K, D):
        # Function to evaluate the configuration variable (q) at t = t + dt under the action of
        # external force acting at the tip
        q_init = np.asarray(q_init).reshape(-1)
        t_span = (t0, t0 + dt)

        sol = solve_ivp(fun = lambda t, q: self.new_Model(t, q, f_ext_fun, K,D, q_ref, q_ref_dot),
                        t_span = t_span,
                        y0 = q_init,
                        method= 'RK45',
                        t_eval = [t0 + dt])

        q_next = sol.y[:,-1]

        return q_next.reshape(-1,1)

    def q_no_load(self, ell):
        # Function to evaluate the q under no load from the given Ells

        q_0 = np.zeros(3*self.n)


        ell_1, ell_2, ell_3 = ell

        S = (ell_1 + ell_2 + ell_3)/3


        if np.isclose(ell_1,ell_2, atol = 0.00095) and np.isclose(ell_2, ell_3, atol  = 0.00095) and np.isclose(ell_1, ell_3, atol = 0.00095):
            theta = 0.0
            kappa = 0.0
            phi = 0
        else:
            kappa = 2*math.sqrt(ell_1**2 + ell_2**2 + ell_3**2 - ell_1 * ell_2 - ell_1*ell_3 - ell_2*ell_3)/(self.d*(ell_1 + ell_2 + ell_3))
            theta = kappa * S
            phi = math.atan2(math.sqrt(3) * (ell_2 + ell_3 - 2*ell_1), (3*(ell_2 - ell_3)))



        # if np.isclose(ell_1, ell_2, atol = 0.05) and np.isclose(ell_2, ell_3, tol = 0.05) and np.isclose(ell_1, ell_3, atol = 0.05):
        #     theta = 0.1
        #     phi      = phi % (2*math.pi)
        #     prev_phi = prev_phi % (2*math.pi)

        #     eps = 0.01  # small tolerance for “near 0” or “near π”
        #     if abs(prev_phi) < eps and abs(phi - math.pi) < eps:
        #         hi = prev_phi
        #     elif abs(prev_phi - math.pi) < eps and abs(phi) < eps:
        #         phi = prev_phi
        #     else:
        #         # 3) compute minimal signed difference in (−π, π]
        #         diff = ((phi - prev_phi + math.pi) % (2*math.pi)) - math.pi

        #         # 4) if jump > 45° (0.785 rad), nudge by ±0.08 rad
        #         if abs(diff) > 0.785:
        #             phi = prev_phi + (0.08 if diff > 0 else -0.08)
        #             # re‐normalize if needed
        #             phi = phi % (2*math.pi)



        for i in range(self.n):
            idx = i*3
            q_0[idx] = phi
            q_0[idx+1] = theta/self.n
            q_0[idx+2] = S/self.n

        return q_0

    def Compute_Jacobian_Ell_Q(self, q):
        # Function to evaluate the jacobian relating change in Ell w.r.t change in q
        q = np.asarray(q).reshape(-1)
        J = np.zeros((3,3*self.n))
        def Compute_d_Ell(q,d,i,j):
            if i ==0:
                dphi   = -d*q[j+1]*np.cos(q[j])
                dtheta = -d*np.sin(q[j-1])
            elif i ==1:
                dphi   = d*q[j+1]*np.sin(math.pi/6 - q[j])
                dtheta = d*np.cos(math.pi/6 - q[j-1])
            else:
                dphi   = d*q[j+1]*np.sin(math.pi/6 + q[j])
                dtheta = -d*np.cos(math.pi/6 + q[j-1])
            return dphi, dtheta

        for i in range(3):
            for j in range(0,3*self.n,3):
                J[i,j],_ = Compute_d_Ell(q,self.d,i,j)
            for j in range(1,3*self.n,3):
                _,J[i,j] = Compute_d_Ell(q,self.d,i,j)
            for j in range(2,3*self.n,3):
                J[i,j] = 1
        return J

    def Actuated_Jacobian(self, q):
        # Final actuated Jacobian equating change in tip_coordinates based on changes in Ell
        J_rq = self.ComputeJacobian(q)
        J_Eq = self.Compute_Jacobian_Ell_Q(q)

        J_Eq_inv = np.linalg.pinv(J_Eq)

        J_a = J_rq @ J_Eq_inv

        return J_a

    # Next three functions basically help to evaluate the plotting points based on the q, in other words
    # is the forward_kinematics of the SBA

    def Transform(self, theta, phi,l):
        T  = np.eye(4)

        H1 = np.eye(4)
        H1[:3, :3] = self.Rz(phi)

        H3 = np.eye(4)
        H3[:3, :3] = self.Rz(-phi)

        if abs(theta) < 1e-4:
            H2 = np.eye(4)
            H2[:3, :3] = self.Ry(theta)
            H2[:3,  3] = [0.0, 0.0, l]

            T = H1 @ H2 @ H3
        else:
            r = l/theta

            H2 = np.eye(4)
            H2[:3, :3] = self.Ry(theta)
            H2[:3,  3] = [r*(1 - np.cos(theta)),
                                            0.0,
                                r*np.sin(theta)]
            T = H1 @ H2 @ H3

        return T

    def Position_Vector(self, theta, samples, phi, l, T):
        t = np.linspace(0, theta, samples)
        X = np.zeros((3, samples))
        L = np.linspace(0,l,samples)

        for i in range(samples):
            Mat = self.Transform(t[i], phi, L[i])
            Transform_matrix = T @ Mat
            X[:, i] = Transform_matrix[:3, 3]

        return X

    def Compute_Soft_Curve (self, q):
        q = np.asarray(q).reshape(-1)
        samples = 100
        T = np.eye(4)
        softPts = np.zeros((3, samples*self.n))
        skelPts = np.zeros((3, self.n + 1))
        v = np.zeros((3,1))
        phi_s   = q[0::3]
        theta_s = q[1::3]
        L       = q[2::3]

        for i in range(self.n):
            Xi = self.Position_Vector(theta_s[i], samples, phi_s[i], L[i], T)
            start = i*samples
            end   = start + samples
            softPts[:, start:end] = Xi
            T = T @ self.Transform(theta_s[i], phi_s[i], L[i])
            skelPts[:, i+1] = softPts[:, end - 1]

        v = softPts[:, -1]

        return softPts, skelPts, v

    def Compute_actual_tip(self,q):
        dh_aug = self.Augmented_DH(q)
        new_dh_aug = dh_aug
        new_dh_aug[-1,1] = self.r

        T = np.eye(4)

        for k in range(len(new_dh_aug[:,0])):
            theta, d, a, alpha = new_dh_aug[k]
            T = T @ self.DH_transform(theta, d, a, alpha)

        actual_tip = T[:3,3]

        return actual_tip

    def ell_q(self,q):
        ell_1 = q.copy()
        ell_2 = q.copy()
        ell_3 = q.copy()
        for i in range(0,3*self.n,3):
            ell_1[2+i] = q[2+i] - self.d*q[i+1]*np.sin(q[i])
            ell_2[2+i] = q[2+i] + self.d*q[i+1]*np.cos((math.pi/6) - q[i])
            ell_3[2+i] = q[2+i] - self.d*q[i+1]*np.cos((math.pi/6) + q[i])

        return ell_1, ell_2, ell_3




def InvKinematics(frame, path_line, goal_mark, hSoft, Ell_1, Ell_2, Ell_3, hSkel, top, rigid_portion, bottom, hForce,
                  time_text, len1_mark, len2_mark, len3_mark, kin, PathCoords, K, D):

    global i_goal
    global q_0, q_next, u_ells
    global time_start, time_prev, tip_prev
    global goal_threshold, r
    global q_0_prev

    print('q_0_prev: {}'.format(q_0_prev))


    # print('u_ells : {}' .format(u_ells))



    time_now = time.time()
    t_elapsed = time_now - time_start
    dt = time_now - time_prev
    time_prev = time_now


    _, numGoals = PathCoords.shape

    f_ext_fun = lambda t: np.array([0, 0, 0])

    q_0_current = kin.q_no_load(u_ells)
    print('q_0_current: {}'.format(q_0_current))

    q_0_dot = (q_0_current - q_0_prev)/dt

    q_next = kin.q_dynamics(q_0_current, f_ext_fun, t_elapsed, dt, K,D, q_0_dot)
    

    print('q; {}'.format(q_next))

    softPts, skelPts, _ = kin.Compute_Soft_Curve(q_next)

    ell_1, ell_2, ell_3 = kin.ell_q(q_next)
    ell_1_pts,_,_ = kin.Compute_Soft_Curve(ell_1)
    ell_2_pts,_,_ = kin.Compute_Soft_Curve(ell_2)
    ell_3_pts,_,_ = kin.Compute_Soft_Curve(ell_3)

    ell_1_origin = np.array([0,kin.d,0]).reshape(-1,1)
    ell_2_origin = np.array([-kin.d*np.cos(math.pi/6), -kin.d*np.sin(math.pi/6), 0]).reshape(-1,1)
    ell_3_origin = np.array([kin.d*np.cos(math.pi/6), -kin.d*np.sin(math.pi/6),0]).reshape(-1,1)


    ell_1_pts = ell_1_pts + ell_1_origin
    ell_2_pts = ell_2_pts + ell_2_origin
    ell_3_pts = ell_3_pts + ell_3_origin

    tip_coord = kin.Compute_actual_tip(q_next)

    # skelPts = np.append(skelPts, tip_coord.reshape(-1,1), axis = 1)





    f_vec      = f_ext_fun(t_elapsed)
    goal_coord = PathCoords[:,i_goal]
    error      = goal_coord - tip_coord
    tip_vel    = (tip_coord - tip_prev)/dt

    # Main inverse_kinematics closed loop (CLIK)
    try:
        J_ac = kin.Actuated_Jacobian(q_next)
        U, S, Vh = np.linalg.svd(J_ac, full_matrices = False)
        V = Vh.T
        sigma_0 = 0.01*max(S)
        nu = 50
        h = (S**3 +nu*S**2 + 2*S + 2*sigma_0)/(S**2 + nu*S + 2)

        H_inv = np.diag(1.0/h)
        J_ac_inv = V @ H_inv @ U.T

        error_norm = np.linalg.norm(error)

        # rate_fb = 0.19 * (1.0  + max(0.0, 2.0 - error_norm))
        rate_fb = 0.15
        rate_ff = 5e-3

        delta_u = J_ac_inv @ (rate_fb *error + rate_ff*(-tip_vel))

    except np.linalg.LinAlgError:
        print("Failed to converge")

        delta_u = np.zeros(3)

    u_ells = u_ells + kin.gain*delta_u
    u_ells = np.round(np.clip(u_ells, 2, 50), 3)

    # print('delta_U: {}' .format(delta_u))
    # print('U_ells: {}'. format(u_ells))

    if np.linalg.norm(error) < goal_threshold:
        i_goal = ((i_goal+1) % numGoals)

    f_vec = f_ext_fun(t_elapsed)

    path_line.set_data(PathCoords[0,:], PathCoords[1,:])
    path_line.set_3d_properties(PathCoords[2,:])

    goal_mark.set_data([goal_coord[0]], [goal_coord[1]])
    goal_mark.set_3d_properties([goal_coord[2]])

    hSoft.set_data(softPts[0,:], softPts[1,:])
    hSoft.set_3d_properties(softPts[2,:])

    Ell_1.set_data(ell_1_pts[0,:], ell_1_pts[1,:])
    Ell_1.set_3d_properties(ell_1_pts[2,:])
    Ell_2.set_data(ell_2_pts[0,:], ell_2_pts[1,:])
    Ell_2.set_3d_properties(ell_2_pts[2,:])
    Ell_3.set_data(ell_3_pts[0,:], ell_3_pts[1,:])
    Ell_3.set_3d_properties(ell_3_pts[2,:])

    hSkel.set_data(skelPts[0,:], skelPts[1,:])
    hSkel.set_3d_properties(skelPts[2,:])

    top.set_data([ell_1_pts[0,-1], ell_2_pts[0,-1], ell_3_pts[0,-1], ell_1_pts[0,-1]], [ell_1_pts[1,-1], ell_2_pts[1,-1], ell_3_pts[1,-1], ell_1_pts[1,-1]])
    top.set_3d_properties([ell_1_pts[2,-1], ell_2_pts[2,-1], ell_3_pts[2,-1], ell_1_pts[2,-1]])

    rigid_portion.set_data([skelPts[0,-1], tip_coord[0]], [skelPts[1,-1], tip_coord[1]])
    rigid_portion.set_3d_properties([skelPts[2,-1], tip_coord[2]])

    bottom.set_data([ell_1_pts[0,0], ell_2_pts[0,0], ell_3_pts[0,0], ell_1_pts[0,0]], [ell_1_pts[1,0], ell_2_pts[1,0], ell_3_pts[1,0], ell_1_pts[1,0]])
    bottom.set_3d_properties([ell_1_pts[2,0], ell_2_pts[2,0], ell_3_pts[2,0], ell_2_pts[2,0]])

    hForce.set_data([tip_coord[0], tip_coord[0] + 5*f_vec[0]], [tip_coord[1], tip_coord[1] + 5*f_vec[1]])
    hForce.set_3d_properties([tip_coord[2], tip_coord[2] + 5*f_vec[2]])

    len1_mark.set_data([1,1], [0, u_ells[0]])
    len2_mark.set_data([2,2], [0, u_ells[1]])
    len3_mark.set_data([3,3], [0, u_ells[2]])

    time_text.set_text(f'Time: {t_elapsed:.2f}s')

    tip_prev = tip_coord
    q_0_prev = q_0_current

    print('U_Ells: {}'. format(u_ells))

    return path_line, goal_mark, hSoft, Ell_1, Ell_2, Ell_3, hSkel,top,rigid_portion, bottom, hForce, time_text, len1_mark, len2_mark, len3_mark

def main():
    global q_0, q_next, u_ells, time_prev, tip_prev, goal_threshold, time_start
    global q_0_prev

    kin = Kinematics()

    # Path
    # PathCoords = np.array([[   8.00,    7.3068,    5.3644,    2.5576,   -0.5576,   -3.3644,   -5.3068,   -6.0000,   -5.3068,   -3.3644,   -0.5576,    2.5576,    5.3644,    7.3068,    8.0000],
    #                        [   0.00,    3.0372,    5.4728,    6.8245,    6.8245,    5.4728,    3.0372,    0.0000,   -3.0372,   -5.4728,   -6.8245,   -6.8245,   -5.4728,   -3.0372,   -0.0000],
    #                        [40.0000,    39.8019,   39.2470,   38.4450,   37.5550,   36.7530,   36.1981,   36.0000,   36.1981,   36.7530,   37.5550,   38.4450,   39.2470,    39.8019,  40.0000]])


    # PathCoords = np.array([[0], [0], [32.73]])

    PathCoords = np.array([[   0.01, -5, -15,   -13,   -11,    -9,    -7,    -5,    -3,    -1,     1,     3,     5,     7,     9,    11,    13,    15, 8],
                           [      0,  -3, 0,         0,  0,         0,    0,    0,    0,     0,     0,      0,     0,     0,    0,     0,     0,    0,  0],
                           [    25, 22, 20.4772,    24.2736,   26.5758,   28.1909,   29.3527,   30.1658,   30.6844,   30.9374,   30.9374,   30.6844,   30.1658,    29.3527,   28.1909,   26.5758,    24.2736,    20.4772, 20]])

    
    # PathCoords = np.array([[     0,   0,         0,  0,         0,    0,    0,    0,     0,     0,      0,     0,     0,    0,     0,     0,    0,  0],
    #                        [   0.05, -15,   -13,   -11,    -9,    -7,    -5,    -3,    -1,     2,     4,     5,     7,     9,    11,    13,    15, 8],
    #                        [    25, 20.4772,    24.2736,   26.5758,   28.1909,   29.3527,   30.1658,   30.6844,   30.9374,   30.9374,   30.6844,   30.1658,    29.3527,   28.1909,   26.5758,    24.2736,    20.4772, 20]])

    # PathCoords = np.array([[     1,   2.4,   3.1,  3.9,   4.4,    5,    5.9,    7.2,   7.9,     7.9,      7.2,     5.9,     5,    5.5,   3.9,     3.1,    2.4,  1],
    #                        [   0.01, -15,   -13,   -11,    -9,    -7,    -5,    -3,    -1,     1,     3,     5,     7,     9,    11,    13,    15, 8],
    #                        [    25, 20.4772, 24.2736,   26.5758,   28.1909,   29.3527,   30.1658,   30.6844,   30.9374,   30.9374,   30.6844,   30.1658,    29.3527,   28.1909,   26.5758,    24.2736,    20.4772, 20]])
    
    
    u_ells = np.array([8.9134, 8.9134, 8.9134])

    
    q_0_prev = kin.q_no_load(u_ells)

    tip_prev = kin.Compute_actual_tip(q_0_prev)

    # Physical Impedance of the system, the stiffness and the D matrix
    # K = np.diag([15,15,.5,15,15,.5,15,15,.5])
    # D = np.diag([.5,.5,.5,.5,.5,.5,.5,.5,.5])

    # K = np.diag([15,15,.5,15,15,.5,15,15,.5])
    # D = np.diag([1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5])

    # K = np.diag([15,15,.5,15,15,.5,15,15,.5])
    # D = np.diag([2,2,2,2,2,2,2,2,2])


    K = np.diag([15,15,.5,15,15,.5,15,15,.5])
    D = np.diag([5,5,5,5,5,5,5,5,5])

    # Plotting and figure handles

    fig = plt.figure(figsize = (12,5))

    ax1 = fig.add_subplot(1,2,1, projection ='3d')
    ax2 = fig.add_subplot(1,2,2)

    path_line, = ax1.plot([], [], [], '-o', color = 'tab:blue', label = 'path_coords', markersize = 5)
    goal_mark, = ax1.plot(PathCoords[0,:],PathCoords[1,:],PathCoords[2,:], '.', color = 'tab:red', markersize = 4, label = 'goal_point')
    hSoft,     = ax1.plot([], [], [], '-', color = 'tab:orange', label = 'PCC', linewidth = 3, )
    Ell_1,     = ax1.plot([], [], [], '-', color = 'tab:blue', label = 'Ell_1', linewidth = 3, alpha = 0.5)
    Ell_2,     = ax1.plot([], [], [], '-', color = 'tab:green', label = 'Ell_2', linewidth = 3, alpha = 0.5)
    Ell_3,     = ax1.plot([], [], [], '-', color = 'tab:brown', label = 'Ell_3', linewidth = 3, alpha = 0.5)
    hSkel,     = ax1.plot([], [], [], '--o', color = 'black', label = 'Rigid_structure', linewidth = 1)

    top,       = ax1.plot([], [], [], '--', color = 'black', linewidth = 1, alpha = 0.5)

    rigid_portion, = ax1.plot([], [], [], '-', color = 'black', label = 'Offset', linewidth = 3)

    bottom,    = ax1.plot([], [], [], '--', color = 'black', linewidth = 1, alpha = 0.5)

    hForce, = ax1.plot([], [], [], '-', color='cyan', linewidth=3, label='Force Vector')

    time_text = ax1.text2D(0.02, 0.95, '', transform = ax1.transAxes,fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    len1_mark, = ax2.plot([], [], '-o', color = 'tab:blue', label = 'chambel_1', linewidth = 3)
    len2_mark, = ax2.plot([], [], '-o', color = 'tab:green', label = 'chamber_2', linewidth = 3)
    len3_mark, = ax2.plot([], [], '-o', color = 'tab:brown', label = 'chamber_3', linewidth = 3)

    ax1.set_xlim([-35, 35])
    ax1.set_ylim([-35, 35])
    ax1.set_zlim([0,45])
    ax1.legend()
    ax1.set_xlabel('X-axis (mm)')
    ax1.set_ylabel('Y-axis (mm)')
    ax1.set_zlabel('Z-axis (mm)')
    ax1.grid(True)

    ax2.set_xlim([0, 4])
    ax2.set_ylim([0,50])
    ax2.legend()
    ax2.set_xlabel('Chamber  #')
    ax2.set_ylabel('chamber lengths (mm)')

    time_start = time.time()

    time_prev= time_start - 0.05



    anim = animation.FuncAnimation(
        fig, InvKinematics, fargs = (path_line, goal_mark, hSoft,Ell_1, Ell_2, Ell_3, hSkel,top, rigid_portion, bottom, hForce,
                                     time_text, len1_mark, len2_mark, len3_mark, kin, PathCoords, K,D),
                                     cache_frame_data= False, interval = 30, blit = True
    )

    # Function to stop/play the simulation using spacebar
    paused = {'flag': False}
    def on_press(event):
        if event.key == ' ' or event.key == 'space':
            if paused['flag']:
                anim.event_source.start()
                paused['flag'] = False
            else:
                anim.event_source.stop()
                paused['flag'] = True


    fig.canvas.mpl_connect('key_press_event', on_press)






    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()


















































