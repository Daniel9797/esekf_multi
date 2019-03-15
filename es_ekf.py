
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rotations import Quaternion, skew_symmetric, quaternion_matrix, wraptopi
sys.path.append('./data')

## Loading the data

with open('data/p1_data.pkl', 'rb') as file:
    data = pickle.load(file)

gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']

## Plot ground truth trajectory

gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth trajectory')
ax.set_zlim(-1, 5)
plt.show()


## Transformation of lidar frame to imu frame using rotation matrix and translation vector

# This is the correct calibration rotation matrix, corresponding to an euler rotation of 0.05, 0.05, .1.
C_li = np.array([
    [ 0.99376, -0.09722,  0.05466],
    [ 0.09971,  0.99401, -0.04475],
    [-0.04998,  0.04992,  0.9975 ]
])

# This is an incorrect calibration rotation matrix, corresponding to a rotation of 0.05, 0.05, 0.05
# C_li = np.array([
#     [ 0.9975 , -0.04742,  0.05235],
#     [ 0.04992,  0.99763, -0.04742],
#     [-0.04998,  0.04992,  0.9975 ]
# ])

t_li_i = np.array([0.5, 0.1, 0.5])

lidar.data = (C_li@lidar.data.T).T + t_li_i

## Sensor Variance

var_imu_f = 0.1
var_imu_w = 1.0
var_gnss = 0.01
var_lidar = 0.25


g = np.array([0, 0, -9.81])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian


## Setup initial values for the filter

p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep

p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
p_cov[0] = np.eye(9)  # covariance of estimate
gnss_i = 0
lidar_i = 0


## Measurement update

def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    # Compute Kalman Gain

    K_k = p_cov_check@h_jac.T@np.linalg.inv(h_jac@p_cov_check@h_jac.T + sensor_var*np.eye(3))

    # Compute error state

    delta_x = K_k@(y_k - p_check)

    # Correct predicted state

    p_check = p_check + delta_x[:3]

    v_check = v_check + delta_x[3:6]
    
    q_check = q_check

    # Compute corrected covariance

    p_cov_check = (np.eye(9) - K_k@h_jac)@p_cov_check

    return p_check, v_check, q_check, p_cov_check


## Error state extended kalman filter loop

P_k = p_cov[0]
p_k = p_est[0]
v_k = v_est[0]
q_k = q_est[0]

for k in range(1, imu_f.data.shape[0]):  
    delta_t = imu_f.t[k] - imu_f.t[k - 1]

    # Update state with IMU inputs
    
    C_ns = quaternion_matrix(q_k)
    
    p_k = p_k + delta_t*v_k + ((0.5*delta_t*delta_t)*(C_ns@imu_f.data[k-1]-g ))
 
    v_k = v_k + delta_t*(C_ns@imu_f.data[k-1]-g)
    
    q_k = Quaternion(axis_angle = imu_w.data[k-1]*delta_t).quat_mult(q_k)

    # Linearize Motion Model

    F_k = np.zeros([9,9])
    F_k = np.diag(np.ones(9))
    F_k[3:6,6:] = -skew_symmetric(C_ns@imu_f.data[k-1])*delta_t
    F_k[:3,3:6] = delta_t*np.eye(3)
    
    # Propagate uncertainty
    
    Q_k = np.zeros([6,6])
    Q_k = (delta_t**2)*np.diag([var_imu_f,var_imu_f,var_imu_f,var_imu_w,var_imu_w,var_imu_w])

    P_k = F_k@P_k@F_k.T + l_jac@Q_k@l_jac.T

    # Check availability of GNSS and LIDAR measurements
    
    if (k in np.in1d(imu_f.t,gnss.t).nonzero()[0]):
        y_k = gnss.data[np.where(gnss.t == imu_f.t[k])[0].item()]
        p_check, v_check, q_check, p_cov_check = measurement_update(var_gnss, P_k, y_k, p_k, v_k, q_k)
        
        p_k = p_check
        v_k = v_check
        q_k = q_check
        P_k = p_cov_check

    elif (k in np.in1d(imu_f.t,lidar.t).nonzero()[0]):
        y_k = lidar.data[np.where(lidar.t == imu_f.t[k])[0].item()]
        p_check, v_check, q_check, p_cov_check = measurement_update(var_lidar, P_k, y_k, p_k, v_k, q_k)
        
        p_k = p_check
        v_k = v_check
        q_k = q_check
        P_k = p_cov_check
        

    p_est[k] = p_k
    v_est[k] = v_k
    q_est[k] = q_k
    p_cov[k] = P_k

## Plot the estimated trajectory

est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Final Estimated Trajectory')
ax.legend()
ax.set_zlim(-1, 5)
plt.show()

## PLot the error for each of the 6 DOF with uncertainty bounds

error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error plots')
num_gt = gt.p.shape[0]
p_est_euler = []

# Convert estimated quaternions to euler angles
for q in q_est:
    p_est_euler.append(Quaternion(*q).to_euler())
p_est_euler = np.array(p_est_euler)

# Get uncertainty estimates from P matrix
p_cov_diag_std = np.sqrt(np.diagonal(p_cov, axis1=1, axis2=2))

titles = ['x', 'y', 'z', 'x rot', 'y rot', 'z rot']
for i in range(3):

    ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])
    ax[0, i].plot(range(num_gt), 3 * p_cov_diag_std[:num_gt, i], 'r--')
    ax[0, i].plot(range(num_gt), -3 * p_cov_diag_std[:num_gt, i], 'r--')
    ax[0, i].set_title(titles[i])

for i in range(3):
    ax[1, i].plot(range(num_gt), gt.r[:, i] - p_est_euler[:num_gt, i])
    ax[1, i].plot(range(num_gt), 3 * p_cov_diag_std[:num_gt, i+6], 'r--')
    ax[1, i].plot(range(num_gt), -3 * p_cov_diag_std[:num_gt, i+6], 'r--')
    ax[1, i].set_title(titles[i+3])
plt.show()