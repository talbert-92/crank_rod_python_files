import numpy as np
import sympy as sp
from sympy.physics.mechanics import *
import math
import pandas as pd 
import crankRod_function as crf
from scipy.integrate import odeint
import matplotlib.pyplot as plt

fact_rad_deg = 180/np.pi
fact_deg_rad = np.pi/180
#***********
#   PATH OF CSV FILES
#***********
path = ""

#***********
#   CREATE SYMBOLS - STATIC 
#***********
a_1, a_2 = sp.symbols('a_1 a_2')
X = sp.symbols('X')
length_list = [a_1,a_2,X]
theta1 = sp.Symbol('theta1',real = True)
theta2 = sp.Symbol('theta2',real = True)
tau = sp.Symbol('tau',positive=True,real = True)
stat_list = [theta1,theta2,X]
theta1_dot, theta2_dot, X_dot = sp.symbols("theta1_dot theta2_dot X_dot")
theta1_ddot, theta2_ddot, X_ddot = sp.symbols("theta1_ddot theta2_ddot X_ddot")
static_list = [theta1,theta2,X]
static_list_dot = [theta1_dot,theta2_dot,X_dot]
static_list_ddot = [theta1_ddot,theta2_ddot,X_ddot]

#***********
#   CREATE SYMBOLS - DYNAMIC 
#***********
t = sp.Symbol("t")
X_t, theta1_t, theta2_t = dynamicsymbols('X_t theta1_t theta2_t')
theta1_dot_t = theta1_t.diff(t)
theta2_dot_t = theta2_t.diff(t)
X_dot_t = X_t.diff(t)
theta1_ddot_t = theta1_dot_t.diff(t)
theta2_ddot_t = theta2_dot_t.diff(t)
X_ddot_t = X_dot_t.diff(t)
print("ddot terms: ",theta1_ddot_t,theta2_ddot_t,X_ddot_t)
dyn_list = [theta1_t,theta2_t,X_t]
dyn_list_dot = [theta1_dot_t,theta2_dot_t,X_dot_t]
dyn_list_ddot = [theta1_ddot_t,theta2_ddot_t,X_ddot_t]

#***********
#   CREATE SYMBOLS - MASS PROPERTIES 
#***********
Icm_a1, Icm_a2, m_a1, m_a2, m_x, g_s = sp.symbols("Icm_a1 Icm_a2 ma1 ma2 mx g")
param_list = [Icm_a1,Icm_a2,m_a1,m_a2,m_x,g_s] 
F_a = sp.Symbol("F_a")
k_p, k_i, theta1_dot_goal = sp.symbols("k_p k_i theta1_dot_goal")
#***********
#   CREATE SYMBOLS - MASS PROPERTIES 
#***********
a1 = 1
a2 = 2
ma1 = 1
ma2 = 1
Icma1 = 1/12
Icma2 = 1/3
mx = 1
g = 9.81
tau_float = 0
Fa = 0
param_list_length = [a1,a2]

#***********
#   CONTROL VARIABLES
#***********
kp = 1.12
ki=0.055
theta1Dot_goal = 1
integral_term = 0
#**************************
#	CLOSING EQUATIONS
#**************************
[eq_X,eq_Y] = crf.closure_eq(length_list,static_list)

#**************************
#	CLOSING EQUATION FOR COMPUTING X POS - X = f(theta1)
#**************************
X_expr = crf.compute_X_piston(length_list,static_list)
print("X expression positive: ",X_expr)					

#**************************
#	CLOSING EQUATION FOR COMPUTING THETA 2 - theta2 = f(theta1) - ATAN2
#**************************
expr_theta2 = crf.compute_theta2(length_list,static_list)
print("theta2 expression: ",expr_theta2)

#**************************
#	DERIVATION PROCESS - VELOCITY SECTION
#**************************
[theta2_dot_f_theta1,X_dot_f_theta1] = crf.compute_velocities(length_list,static_list,dyn_list,static_list_dot,dyn_list_dot)
print("theta2 dot f theta1: ",theta2_dot_f_theta1)
print("xDot dot f theta1: ",X_dot_f_theta1)
#*********************************
#   COMPUTE VELOCITIES OF COG BODIES
#*********************************

[v_a1_COG_squared,v_a2_COG_squared,v_x_COG_squared] = crf.compute_COG_vel_squared(length_list,static_list,dyn_list,static_list_dot,dyn_list_dot)

print("v a_1 COG: ", v_a1_COG_squared)
print("v a_2 COG: ", v_a2_COG_squared)
print("v x COG: ", v_x_COG_squared)

#*********************************
#   COMPUTE ACCELERATIONS OF COG BODIES
#*********************************

[theta2_Ddot_expr,X_Ddot_expr] = crf.compute_accelerations(length_list,static_list,dyn_list,static_list_dot,dyn_list_dot,static_list_ddot,dyn_list_ddot)

print("theta2 acceleration: ",theta2_Ddot_expr)
print("X acceleration: ",X_Ddot_expr)


#*********************************
#   COMPUTE ENERGY OF THE SYSTEM (KINETIC + POTENTIAL)
#*********************************
[T,V] = crf.compute_energy_mechanism(length_list,static_list,dyn_list,static_list_dot,dyn_list_dot,param_list)
print("Kinetic energy: ",T)
print("Potential energy: ",V)

#*********************************
#   LAGRANGIAN EXPRESION
#*********************************
L = T-V
print("")
print("Lagrangian: ",L)
print("")

#*********************************
#   LAGRANGIAN F(THETA1)
#*********************************
L_sub_theta2_dot = L.subs({theta2_dot:theta2_dot_f_theta1,X_dot:X_dot_f_theta1})
print("Lagrangian theta2_dot f(theta1): ", L_sub_theta2_dot)
L_f_theta1 = L_sub_theta2_dot.subs({theta2:expr_theta2})
print("Lagrangian f(theta1): ",L_f_theta1)

#*********************************
#   LAGRANGIAN F(THETA1)
#*********************************
dL_dtheta1 = L_f_theta1.diff(theta1)
dL_dtheta1_dot = L_f_theta1.diff(theta1_dot)
dL_dtheta1_dot_f_t = dL_dtheta1_dot.subs({theta1:theta1_t,theta1_dot:theta1_dot_t})
print("dL dtheta1_dot f(t): ",dL_dtheta1_dot_f_t)
dL_dtheta1_dot_dt = dL_dtheta1_dot_f_t.diff(t)
print("dL dtheta1_dot dt: ",dL_dtheta1_dot_dt)

#*********************************
#   EOM SECTION
#*********************************
EOM_LHS_f_t = dL_dtheta1_dot_dt - dL_dtheta1
EOM_LHS_sub1 = EOM_LHS_f_t.subs({theta1_ddot_t:theta1_ddot})
EOM_LHS_sub2 = EOM_LHS_sub1.subs({theta1_dot_t:theta1_dot})
EOM_LHS = EOM_LHS_sub2.subs({theta1_t:theta1})
print("")
print("EOM LHS: ",EOM_LHS)
print("")

pistonForce = -F_a*sp.cos(theta1)
dX_dot_dtheta1_dot = X_dot_f_theta1.diff(theta1_dot)
#EOM_RHS = tau*sp.cos(theta1) + pistonForce*dX_dot_dtheta1_dot
EOM_RHS = tau 
print("")
print("EOM RHS: ",EOM_RHS)
print("")
#****************************
#   ODE SYSTEM TO BE SOLVED
#****************************
def secondOrder(y,t,x_ddot_numerical):
    x_num = y[0]
    xdot_num = y[1]
    x_ddot = x_ddot_numerical

    return [xdot_num, x_ddot]


#********************************
#   INITIAL CONDITION
#********************************
y0 = [0,0]
EOM_LHS_sub = EOM_LHS.subs({theta1:y0[0],theta1_dot:y0[-1],
                            Icm_a1:Icma1,Icm_a2:Icma2,m_a1:ma1,m_a2:ma2,
                            m_x:mx,g_s:g,tau:tau_float,a_1:a1,a_2:a2})
EOM_RHS_sub = EOM_RHS.subs({theta1:y0[0],theta1_dot:y0[-1],tau:tau_float,F_a:Fa,a_1:a1,a_2:a2,theta1_dot_goal:theta1Dot_goal,k_p:kp,k_i:ki})
EOM_eq = sp.Eq(EOM_LHS_sub,EOM_RHS_sub)

theta1_ddot_expr = sp.solve(EOM_eq,theta1_ddot)
theta1_ddot_expr = np.float32(theta1_ddot_expr)


#********************************
#   INIT VECTOR
#********************************
t = np.linspace(0,1,1000)
theta1_res = np.zeros_like(t)
theta1_res[0] = y0[0]
theta1_dot_res = np.zeros_like(t)
theta1_dot_res[0] = y0[1]
theta1_ddot_res = np.zeros_like(t)
temp,theta1_ddot_res[0] = secondOrder(y0,[t[0],t[1]],theta1_ddot_expr)
x_piston_position = np.zeros_like(t)
x_piston_velocity = np.zeros_like(t)
x_piston_acceleration = np.zeros_like(t)


#********************************
#   SOLVE ODE
#********************************
for i in range(len(t)-1):
    ts = [t[i],t[i+1]]
    EOM_LHS_sub = EOM_LHS.subs({theta1:y0[0],theta1_dot:y0[-1],
                                Icm_a1:Icma1,Icm_a2:Icma2,m_a1:ma1,m_a2:ma2,
                                m_x:mx,g_s:g,tau:tau_float,a_1:a1,a_2:a2,F_a:Fa})
    EOM_RHS_sub = EOM_RHS.subs({theta1:y0[0],theta1_dot:y0[-1],tau:tau_float,F_a:Fa,a_1:a1,a_2:a2,theta1_dot_goal:theta1Dot_goal,k_p:kp,k_i:ki})
    EOM_eq = sp.Eq(EOM_LHS_sub,EOM_RHS_sub)
    theta1_ddot_expr = sp.solve(EOM_eq,theta1_ddot)
    #print(theta1_ddot_expr)
    theta1_ddot_expr = np.float32(theta1_ddot_expr)
    #print(theta1_ddot_expr)

    y = odeint(secondOrder,y0,ts,args=(theta1_ddot_expr,))

    # if y[-1][0] >= 2*np.pi:
    #     k = math.floor(y[-1][0]/(2*np.pi))
    #     y[-1][0] = y[-1][0] - k*2*np.pi
    # if y[-1][0] <= -2*np.pi:
    #     k = math.floor(y[-1][0]/(2*np.pi))
    #     print("k: ",k)
    #     y[-1][0] = y[-1][0] + k*2*np.pi

    theta1_res[i+1] = y[-1][0]
    theta1_dot_res[i+1] = y[-1][1]
    y0 = y[-1]
    temp, theta1_ddot_res[i+1] = secondOrder(y0,ts,theta1_ddot_expr)


#********************************
#   POST PROCESSING
#********************************
print("theta 1 dynamic: ",theta1_res*fact_rad_deg)
print("theta 1 dot dynamic: ",theta1_dot_res)
print("theta 1 ddot dynamic: ",theta1_ddot_res)
for i in range(len(theta1_res)):
    x_piston_position_temp = X_expr.subs({theta1:theta1_res[i],theta1_dot:theta1_dot_res[i],a_1:a1,a_2:a2})
    x_piston_position_temp = np.float(x_piston_position_temp)
    x_piston_position[i] = x_piston_position_temp
    x_piston_velocity_temp = X_dot_f_theta1.subs({theta1:theta1_res[i],theta1_dot:theta1_dot_res[i],a_1:a1,a_2:a2})
    x_piston_velocity_temp = np.float(x_piston_velocity_temp)
    x_piston_velocity[i] = x_piston_velocity_temp
    x_piston_acceleration_temp = X_Ddot_expr.subs({theta1:theta1_res[i],theta1_dot:theta1_dot_res[i],theta1_ddot:theta1_ddot_res[i],a_1:a1,a_2:a2})
    x_piston_acceleration_temp  = np.float(x_piston_acceleration_temp)
    x_piston_acceleration[i] = x_piston_acceleration_temp
print("piston position: ",x_piston_position)    
print("piston velocity: ",x_piston_velocity)
print("piston acceleration: ",x_piston_acceleration)
[theta2_float,X_float,X_A,Y_A,X_B,Y_B] = crf.kinematic(theta1_res,param_list_length,static_list,length_list)

crf.plot_mechanism(theta1_res,X_A,Y_A,X_B,Y_B)

plt.figure(2)
plt.subplot(3,1,1)
plt.xlabel('time [s]')
plt.ylabel('crank angle [deg]')
plt.title("Crank angle vs time")
plt.plot(t,theta1_res*fact_rad_deg)
plt.tight_layout()
plt.subplot(3,1,2)
plt.xlabel('time [s]')
plt.ylabel('crank velocity [rad/s]')
plt.title("Crank speed vs time")
plt.plot(t,theta1_dot_res)
plt.tight_layout()
plt.subplot(3,1,3)
plt.title("Piston position vs time")
plt.xlabel('time [s]')
plt.ylabel('piston position [m]')
plt.plot(t,x_piston_position)
plt.tight_layout()


colnames = ['time', 'theta1_mat', 'theta1_dot_mat', 'theta1_ddot_mat']
result_matlab = pd.read_csv(path,delimiter=',',skiprows=0,names=colnames)
time_mat = result_matlab.time.tolist()
theta1_mat = result_matlab.theta1_mat.tolist()
theta1_dot_mat = result_matlab.theta1_dot_mat.tolist()
theta1_ddot_mat = result_matlab.theta1_ddot_mat.tolist()


plt.figure(4)
plt.subplot(3,1,1)
plt.xlabel('time [s]',fontsize=14)
plt.ylabel('crank angle [deg]',fontsize=10)
plt.title(r"Crank angle vs time - ($ \theta_1$ vs time) - Matlab Benchmark",fontsize=18)
plt.plot(t,theta1_res*fact_rad_deg,label="python",linewidth=2)
plt.plot(time_mat,theta1_mat,label="matlab",linewidth=2)
plt.legend(loc="best",fontsize=18)
plt.tight_layout()
plt.subplot(3,1,2)
plt.xlabel('time [s]',fontsize=10)
plt.ylabel('crank velocity [rad/s]',fontsize=14)
plt.title(r"Crank speed vs time - ($ \dot \theta_1$ vs time) - Matlab Benchmark",fontsize=18)
plt.plot(t,theta1_dot_res,label="python",linewidth=2)
plt.plot(time_mat,theta1_dot_mat,label="matlab",linewidth=2)
plt.legend(loc="best",fontsize=18)
plt.tight_layout()
plt.subplot(3,1,3)
plt.xlabel('time [s]',fontsize=14)
plt.ylabel('crank acceleration [rad/s^2]',fontsize=10)
plt.title(r"Crank acceleration vs time - ($ \ddot \theta_1$ vs time) - Matlab Benchmark",fontsize=18)
plt.plot(t,theta1_ddot_res,label="python",linewidth=2)
plt.plot(time_mat,theta1_ddot_mat,label="matlab",linewidth=2)
plt.legend(loc="best",fontsize=18)
plt.tight_layout()


plt.show()

#*********************************
#   CHECK KINEMATIC
#*********************************
# theta1_float = np.linspace(start = 0, stop = 360, num = 12)*np.pi/180
# X_float = np.zeros_like(theta1_float)
# theta2_float = np.zeros_like(theta1_float)

# i = 0
# for x in theta1_float:
#     X_float_temp = X_expr.subs({theta1:x,a_1:a_1,a_2:a_2})
#     theta2_temp = expr_theta2.subs({theta1:x,a_1:a_1,a_2:a_2})
#     X_float_temp = float(X_float_temp)
#     theta2_temp = float(theta2_temp)
#     X_float[i] = X_float_temp
#     theta2_float[i] = theta2_temp
#     i = i+1
# print("theta 1 array deg:",theta1_float*fact_rad_deg)
# print("theta 2 array deg:",theta2_float*fact_rad_deg)
# print("X pos array:",X_float)