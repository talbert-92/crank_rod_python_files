import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy.physics.mechanics import *
t = sp.Symbol("t")

#**************************
#	CLOSING EQUATIONS
#**************************
def closure_eq(length_list,static_list):
	a_1,a_2,X = length_list
	theta1,theta2,X = static_list
	eq_X = a_1*sp.cos(theta1) + a_2*sp.cos(theta2) - X
	eq_Y = a_1*sp.sin(theta1) + a_2*sp.sin(theta2) 
	return [eq_X,eq_Y]

#**************************
#	CLOSING EQUATION FOR COMPUTING X POS
#**************************
def compute_X_piston(length_list,static_list):
	a_1,a_2,X = length_list
	theta1,theta2,X = static_list
	X_LHS = a_2*sp.cos(theta2)
	X_RHS = X - a_1*sp.cos(theta1)

	Y_LHS = a_2*sp.sin(theta2)
	Y_RHS = - a_1*sp.sin(theta1)

	eq_LHS = sp.simplify(X_LHS**2 + Y_LHS**2)
	eq_RHS = sp.simplify(X_RHS**2 + Y_RHS**2)

	# print("eq LHS: ",eq_LHS)
	# print("eq RHS: ",eq_RHS)

	eq_to_solve = sp.Eq(eq_LHS,eq_RHS)

	expr_x = sp.solve(eq_to_solve,X)

	# print("X expression: ", expr_x)

	X_expr = expr_x[1]
	# print("X expression positive: ",X_expr)
	return X_expr

#**************************
#	COMPUTE THETA 2 IN FUNCTION OF THETA1
#**************************
def compute_theta2(length_list,static_list):
	a_1,a_2,X = length_list
	theta1,theta2,X = static_list	
	X_expr = compute_X_piston(length_list,static_list)
	expr_theta2 = sp.atan2(-a_1*sp.sin(theta1), X_expr-a_1*sp.cos(theta1))	
	return expr_theta2

#**************************
#	COMPUTE KINEMATIC F(THETA1)
#**************************
def compute_kinematic(length_list,static_list):
	a_1,a_2,X = length_list
	theta1,theta2,X = static_list
	X_LHS = a_2*sp.cos(theta2)
	X_RHS = X - a_1*sp.cos(theta1)

	Y_LHS = a_2*sp.sin(theta2)
	Y_RHS = - a_1*sp.sin(theta1)

	eq_LHS = sp.simplify(X_LHS**2 + Y_LHS**2)
	eq_RHS = sp.simplify(X_RHS**2 + Y_RHS**2)

	# print("eq LHS: ",eq_LHS)
	# print("eq RHS: ",eq_RHS)

	eq_to_solve = sp.Eq(eq_LHS,eq_RHS)

	expr_x = sp.solve(eq_to_solve,X)

	# print("X expression: ", expr_x)

	X_expr = expr_x[1]
	expr_theta2 = sp.atan2(-a_1*sp.sin(theta1), X_expr-a_1*sp.cos(theta1))	

	return [X_expr,expr_theta2]

#**************************
#	DERIVATION PROCESS - VELOCITY SECTION
#**************************
def compute_velocities(length_list,static_list,dyn_list,static_list_dot,dyn_list_dot):
	a_1,a_2,X = length_list
	theta1,theta2,X = static_list
	theta1_t, theta2_t, X_t = dyn_list
	theta1_dot, theta2_dot, X_dot = static_list_dot
	theta1_dot_t, theta2_dot_t, X_dot_t = dyn_list_dot
	[eq_X,eq_Y] = closure_eq(length_list,static_list)

	[X_expr,expr_theta2] = compute_kinematic(length_list,static_list)
	eq_X_time = eq_X.subs({theta1:theta1_t, theta2:theta2_t, X:X_t})
	eq_Y_time = eq_Y.subs({theta1:theta1_t, theta2:theta2_t, X:X_t})

	d_eqX_dt = eq_X_time.diff(t)
	d_eqY_dt = eq_Y_time.diff(t)

	# print("velocity eq X: ",d_eqX_dt)
	# print("velocity eq Y: ",d_eqY_dt)

	term_theta1Dot_X = sp.collect(d_eqX_dt,theta1_dot_t,evaluate=False)		#B11 - change sign
	#term_to_subtract_X_dot_t = term_theta1Dot_X[theta1_dot_t]*theta1_dot_t
	B11 = term_theta1Dot_X[theta1_dot_t]
	term_theta1Dot_Y = sp.collect(d_eqY_dt,theta1_dot_t,evaluate=False)		#B21 - change sign
	#term_to_subtract_Y_dot = term_theta1Dot_Y[theta1_dot_t]*theta1_dot_t	
	B21 = term_theta1Dot_Y[theta1_dot_t]

	term_theta2Dot_X = sp.collect(d_eqX_dt,theta2_dot_t,evaluate=False)		#A11 
#	term_to_subtract_X_dot_t = term_theta2Dot_X[theta2_dot_t]*theta2_dot_t
	A11 = term_theta2Dot_X[theta2_dot_t]
	term_XDot_X = sp.collect(d_eqX_dt,X_dot_t,evaluate=False)					#A12 
#	term_to_subtract_X_dot_t = term_XDot_X[X_dot_t]*X_dot_t
	A12 = term_XDot_X[X_dot_t]

	term_theta2Dot_Y = sp.collect(d_eqY_dt,theta2_dot_t,evaluate=False)		#A21
#	term_to_subtract_X_dot_t = term_theta2Dot_Y[theta2_dot_t]*theta2_dot_t
	A21 = term_theta2Dot_Y[theta2_dot_t]
	term_XDot_Y = sp.collect(d_eqY_dt,X_dot_t,evaluate=False)					#A22
	try:
#		term_to_subtract_X_dot_t = term_XDot_Y[X_dot_t]*X_dot_t
		A22 = term_XDot_Y[X_dot_t]
	except:
		A22 = 0

	# d_eqX_dt = d_eqX_dt - term_to_subtract_X_dot_t
	# d_eqY_dt = d_eqY_dt - term_to_subtract_Y_dot

	A_matrix = sp.Matrix(([A11, A12],[A21,A22] ))
	B_matrix = sp.Matrix(([-B11],[-B21]) )

	# print("A matrix: ",A_matrix)
	# print("B matrix: ",B_matrix)

	A_inv = A_matrix.inv()
	#print("A matrix inverse: ",A_inv)

	S_matrix = A_inv*B_matrix

	S_matrix = S_matrix.subs({theta1_t:theta1, theta2_t:theta2, X_t:X, 
											theta1_dot_t:theta1_dot, theta2_dot_t:theta2_dot, X_dot_t:X_dot})

	#print("S matrix: ", S_matrix)   #first line theta2_dot = f(theta1_dot) second line X_dot = d(theta1_dot)

	theta2_dot_f_theta1 = S_matrix[0,0].subs({theta2:expr_theta2})
	theta2_dot_f_theta1 = theta2_dot_f_theta1*theta1_dot
	X_dot_f_theta1 = S_matrix[1,0].subs({theta2:expr_theta2})
	X_dot_f_theta1 = X_dot_f_theta1*theta1_dot
	# print("theta2 dot f theta1: ",theta2_dot_f_theta1)
	# print("xDot dot f theta1: ",X_dot_f_theta1)
	return [theta2_dot_f_theta1,X_dot_f_theta1]

#*********************************
#   COMPUTE  COG BODIES
#*********************************
def compute_COG(length_list,dyn_list):
	a_1,a_2,X = length_list
	theta1_t, theta2_t, X_t = dyn_list

	a1_COG_x = a_1/2*sp.cos(theta1_t)
	a1_COG_y = a_1/2*sp.sin(theta1_t)

	a2_COG_x = a_1*sp.cos(theta1_t) + a_2/2*sp.cos(theta2_t)
	a2_COG_y = a_1*sp.sin(theta1_t) + a_2/2*sp.sin(theta2_t)

	x_COG_x  = a_1*sp.cos(theta1_t) + a_2*sp.cos(theta2_t)

	return [a1_COG_x,a1_COG_y,a2_COG_x,a2_COG_y,x_COG_x]

def compute_hinge(length_list,dyn_list):
	a_1,a_2,X = length_list
	theta1_t, theta2_t, X_t = dyn_list

	a1_x = a_1*sp.cos(theta1_t)
	a1_y = a_1*sp.sin(theta1_t)

	a2_x = a_1*sp.cos(theta1_t) + a_2*sp.cos(theta2_t)
	a2_y = a_1*sp.sin(theta1_t) + a_2*sp.sin(theta2_t)


	return [a1_x,a1_y,a2_x,a2_y]

#*********************************
#   COMPUTE VELOCITIES OF COG BODIES
#*********************************
def compute_COG_vel_squared(length_list,static_list,dyn_list,static_list_dot,dyn_list_dot):
	a_1,a_2,X = length_list
	theta1,theta2,X = static_list
	theta1_t, theta2_t, X_t = dyn_list
	theta1_dot, theta2_dot, X_dot = static_list_dot
	theta1_dot_t, theta2_dot_t, X_dot_t = dyn_list_dot

	[a1_COG_x,a1_COG_y,a2_COG_x,a2_COG_y,x_COG_x] = compute_COG(length_list,dyn_list)

	v_a1_COG_squared = sp.simplify((a1_COG_x.diff(t))**2 + (a1_COG_y.diff(t))**2) 
	v_a2_COG_squared = sp.simplify((a2_COG_x.diff(t))**2 + (a2_COG_y.diff(t))**2) 
	v_x_COG_squared = sp.simplify((x_COG_x.diff(t))**2 )

	v_a1_COG_squared = v_a1_COG_squared.subs({theta1_dot_t:theta1_dot, theta2_dot_t:theta2_dot, X_dot_t:X_dot})
	v_a1_COG_squared = v_a1_COG_squared.subs({theta1_t:theta1, theta2_t:theta2, X_t:X})       

	v_a2_COG_squared = v_a2_COG_squared.subs({theta1_dot_t:theta1_dot, theta2_dot_t:theta2_dot, X_dot_t:X_dot})
	v_a2_COG_squared = v_a2_COG_squared.subs({theta1_t:theta1, theta2_t:theta2, X_t:X})

	v_x_COG_squared = v_x_COG_squared.subs({theta1_dot_t:theta1_dot, theta2_dot_t:theta2_dot, X_dot_t:X_dot})
	v_x_COG_squared = v_x_COG_squared.subs({theta1_t:theta1, theta2_t:theta2, X_t:X})

	return [v_a1_COG_squared,v_a2_COG_squared,v_x_COG_squared]

#*********************************
#   COMPUTE ACCELERATIONS OF COG BODIES
#*********************************
def compute_accelerations(length_list,static_list,dyn_list,static_list_dot,dyn_list_dot,static_list_ddot,dyn_list_ddot):
    a_1,a_2,X = length_list
    theta1,theta2,X = static_list
    theta1_t, theta2_t, X_t = dyn_list
    theta1_dot, theta2_dot, X_dot = static_list_dot
    theta1_dot_t, theta2_dot_t, X_dot_t = dyn_list_dot
    theta1_ddot,theta2_ddot,X_ddot = static_list_ddot
    theta1_ddot_t,theta2_ddot_t,X_ddot_t = dyn_list_ddot 

    [eq_X,eq_Y] = closure_eq(length_list,static_list)
    [X_expr,expr_theta2] = compute_kinematic(length_list,static_list)   
    [theta2_dot_f_theta1,X_dot_f_theta1] = compute_velocities(length_list,static_list,dyn_list,static_list_dot,dyn_list_dot)
    eq_X_time = eq_X.subs({theta1:theta1_t, theta2:theta2_t, X:X_t})
    eq_Y_time = eq_Y.subs({theta1:theta1_t, theta2:theta2_t, X:X_t})
       
    d_eqX_dt = eq_X_time.diff(t)
    d_eqY_dt = eq_Y_time.diff(t)
        
    d_eqX_dt2 = d_eqX_dt.diff(t)
    d_eqY_dt2 = d_eqY_dt.diff(t)
    
    #print("acc X eq: ",d_eqX_dt2)
    #print("acc Y eq: ",d_eqY_dt2)
    
    
    term_theta2Ddot_X = sp.collect(d_eqX_dt2,theta2_ddot_t,evaluate=False)		#A11 
    A11 = term_theta2Ddot_X[theta2_ddot_t]
    #print("A11: ",A11)
    term_XDdot_X = sp.collect(d_eqX_dt2,X_ddot_t,evaluate=False)
    A12 = term_XDdot_X[X_ddot_t]
    #print("A12: ",A12)
    term_theta2Ddot_Y = sp.collect(d_eqY_dt2,theta2_ddot_t,evaluate=False)		#A11 
    A21 = term_theta2Ddot_Y[theta2_ddot_t]
    #print("A21: ",A21)
    term_XDdot_Y = sp.collect(d_eqY_dt2,X_ddot_t,evaluate=False)
    try:
        A22 = term_XDdot_Y[X_ddot_t]
    except:
        A22 = 0
    #print("A22: ",A22)
    
    term_theta2Dot_X = sp.collect(d_eqX_dt2,theta2_dot_t**2,evaluate=False)		#A11 
    B11 = term_theta2Dot_X[theta2_dot_t**2]
    #print("B11: ",B11)
    #term_XDot_X = sp.collect(d_eqX_dt2,X_dot_t,evaluate=False)		#A11 
    B12 = 0
    #print("B12: ",B12)
    term_theta2Dot_Y = sp.collect(d_eqY_dt2,theta2_dot_t**2,evaluate=False)		#A11 
    B21 = term_theta2Dot_Y[theta2_dot_t**2]
    #print("B21: ",B21)
    #term_XDot_X = sp.collect(d_eqX_dt2,X_dot_t,evaluate=False)		#A11 
    B22 = 0
    #print("B12: ",B22)
    
    term_theta1Ddot_X = sp.collect(d_eqX_dt2,theta1_ddot_t,evaluate=False)		#A11 
    C11 = term_theta1Ddot_X[theta1_ddot_t]
    #print("C11: ",C11)
    term_theta1Ddot_Y = sp.collect(d_eqY_dt2,theta1_ddot_t,evaluate=False)
    C21 = term_theta1Ddot_Y[theta1_ddot_t]
    #print("C21: ",C21)
    
    term_theta1Dot_X = sp.collect(d_eqX_dt2,theta1_dot_t**2,evaluate=False)		#A11 
    D11 = term_theta1Dot_X[theta1_dot_t**2]
    #print("D11: ",D11)
    term_theta1Dot_Y = sp.collect(d_eqY_dt2,theta1_dot_t**2,evaluate=False)
    D21 = term_theta1Dot_Y[theta1_dot_t**2]
    #print("D21: ",D21)
    
    A_matrix = sp.Matrix(([A11, A12],[A21,A22] ))
    B_matrix = sp.Matrix(([-B11, -B12],[-B21,-B22] ))
    C_matrix = sp.Matrix(([-C11],[-C21]) )
    D_matrix = sp.Matrix(([-D11],[-D21]) )
    A_inv = A_matrix.inv()
    AB_matrix = A_inv*B_matrix
    AC_matrix = A_inv*C_matrix
    AD_matrix = A_inv*D_matrix
    
    theta2_Ddot_expr_t = AB_matrix[0,0]*theta2_dot_t**2 + AB_matrix[0,1]*X_dot_t**2+AC_matrix[0,0]*theta1_ddot_t  + AD_matrix[0,0]*theta1_dot_t**2
    X_Ddot_expr_t = AB_matrix[1,0]*theta2_dot_t**2 + AB_matrix[1,1]*X_dot_t**2+AC_matrix[1,0]*theta1_ddot_t  + AD_matrix[1,0]*theta1_dot_t**2
    
    theta2_Ddot_expr = theta2_Ddot_expr_t.subs({theta1_ddot_t:theta1_ddot})
    theta2_Ddot_expr = theta2_Ddot_expr.subs({theta1_dot_t:theta1_dot,theta2_dot_t:theta2_dot,
                                                X_dot_t:X_dot,
                                                theta1_t:theta1,theta2_t:theta2,X_t:X})
    X_Ddot_expr = X_Ddot_expr_t.subs({theta1_ddot_t:theta1_ddot})
    X_Ddot_expr = X_Ddot_expr.subs({theta1_dot_t:theta1_dot,theta2_dot_t:theta2_dot,
                                                X_dot_t:X_dot,
                                                theta1_t:theta1,theta2_t:theta2,X_t:X})
    
    theta2_Ddot_expr = theta2_Ddot_expr.subs({theta2_dot:theta2_dot_f_theta1})
    X_Ddot_expr = X_Ddot_expr.subs({theta2_dot:theta2_dot_f_theta1})
    theta2_Ddot_expr = theta2_Ddot_expr.subs({theta2:expr_theta2})
    X_Ddot_expr = X_Ddot_expr.subs({theta2:expr_theta2})
    
    #print("theta2 acceleration: ",theta2_Ddot_expr)
    #print("X acceleration: ",X_Ddot_expr)
    return [theta2_Ddot_expr,X_Ddot_expr]
    
#*********************************
#   COMPUTE KINETIC ENERGY OF THE SYSTEM
#*********************************
def compute_energy_mechanism(length_list,static_list,dyn_list,static_list_dot,dyn_list_dot,param):
	a_1,a_2,X = length_list
	theta1,theta2,X = static_list
	theta1_t, theta2_t, X_t = dyn_list
	theta1_dot, theta2_dot, X_dot = static_list_dot
	theta1_dot_t, theta2_dot_t, X_dot_t = dyn_list_dot
	Icm_a1,Icm_a2,m_a1,m_a2,m_x,g_s = param
	[a1_COG_x,a1_COG_y,a2_COG_x,a2_COG_y,x_COG_x] = compute_COG(length_list,dyn_list)

	v_a1_COG_squared = sp.simplify((a1_COG_x.diff(t))**2 + (a1_COG_y.diff(t))**2) 
	v_a2_COG_squared = sp.simplify((a2_COG_x.diff(t))**2 + (a2_COG_y.diff(t))**2) 
	v_x_COG_squared = sp.simplify((x_COG_x.diff(t))**2 )

	v_a1_COG_squared = v_a1_COG_squared.subs({theta1_dot_t:theta1_dot, theta2_dot_t:theta2_dot, X_dot_t:X_dot})
	v_a1_COG_squared = v_a1_COG_squared.subs({theta1_t:theta1, theta2_t:theta2, X_t:X})       

	v_a2_COG_squared = v_a2_COG_squared.subs({theta1_dot_t:theta1_dot, theta2_dot_t:theta2_dot, X_dot_t:X_dot})
	v_a2_COG_squared = v_a2_COG_squared.subs({theta1_t:theta1, theta2_t:theta2, X_t:X})

	v_x_COG_squared = v_x_COG_squared.subs({theta1_dot_t:theta1_dot, theta2_dot_t:theta2_dot, X_dot_t:X_dot})
	v_x_COG_squared = v_x_COG_squared.subs({theta1_t:theta1, theta2_t:theta2, X_t:X})
	
	#*********************************
	#   COMPUTE KINETIC ENERGY OF THE SYSTEM
	#*********************************

	T_a1 = 1/2*(m_a1*v_a1_COG_squared + Icm_a1*theta1_dot**2)
	T_a2 = 1/2*(m_a2*v_a2_COG_squared + Icm_a2*theta2_dot**2)
	T_x = 1/2*m_x*v_x_COG_squared

	T = T_a1 + T_a2 + T_x

	#*********************************
	#   COMPUTE POTENTIAL ENERGY OF THE SYSTEM
	#*********************************
	V_a1 = m_a1*g_s*a1_COG_y
	V_a2 = m_a1*g_s*a2_COG_y

	V_a1 = V_a1.subs({theta1_dot_t:theta1_dot, theta2_dot_t:theta2_dot, X_dot_t:X_dot})
	V_a1 = V_a1.subs({theta1_t:theta1, theta2_t:theta2, X_t:X})

	V_a2 = V_a2.subs({theta1_dot_t:theta1_dot, theta2_dot_t:theta2_dot, X_dot_t:X_dot})
	V_a2 = V_a2.subs({theta1_t:theta1, theta2_t:theta2, X_t:X})
	V = V_a1 + V_a2

	return [T,V]

#********************************
#   KINEMATIC
#********************************

def kinematic(theta1_res,length_list_num,static_list,length_list):
	a1,a2 = length_list_num
	a_1,a_2,X = length_list
	theta1,theta2,X = static_list

	X_float = np.zeros_like(theta1_res)
	theta2_float = np.zeros_like(theta1_res)
	[X_expr,expr_theta2] = compute_kinematic(length_list,static_list)
	i = 0
	for ii in theta1_res:
		X_float_temp = X_expr.subs({theta1:ii,a_1:a1,a_2:a2})
		theta2_temp = expr_theta2.subs({theta1:ii,a_1:a1,a_2:a2})
		X_float_temp = float(X_float_temp)
		theta2_temp = float(theta2_temp)
		X_float[i] = X_float_temp
		theta2_float[i] = theta2_temp
		i = i+1
	# print("theta 1 array deg:",theta1_res*fact_rad_deg)
	# print("theta 2 array deg:",theta2_float*fact_rad_deg)
	# print("X pos array:",X_float)

	X_A = a1*np.cos(theta1_res)
	Y_A = a1*np.sin(theta1_res)

	X_B = X_float
	Y_B = np.zeros_like(theta1_res)

	return [theta2_float,X_float,X_A,Y_A,X_B,Y_B]


#********************************
#   PLOT
#********************************

def plot_mechanism(theta1_res,X_A,Y_A,X_B,Y_B):
	plt.figure(1,figsize=(12, 7))
	plt.title("Crank rod mechanism")
	plt.xlabel("X coordinate [m]")
	plt.ylabel("Y coordinate [m]")

	for i in range(len(theta1_res)):
		crank_x = [0,X_A[i]]
		crank_y = [0,Y_A[i]]
		
		rod_X = [X_A[i],X_B[i]]
		rod_Y = [Y_A[i],Y_B[i]]

		plt.plot(0,0,marker="^",markersize=20,color="k")
		plt.plot(X_A[i],Y_A[i],marker="o",markersize=15,color="k")
		plt.plot(crank_x,crank_y,'b',label="crank",linewidth=3)
		plt.plot(rod_X,rod_Y,'r',label="rod",linewidth=3)
		plt.plot(X_B[i],Y_B[i],marker="s",markersize=20,color="k",label="piston")
		plt.legend(loc="upper right")
		plt.ylim((-2.5,2.5))
		plt.xlim((-1.5,3.5))
		plt.legend(loc="upper right",fontsize=20)
		plt.title("Crank rod mechanism")
		plt.xlabel("X coordinate [m]")
		plt.ylabel("Y coordinate [m]")
		plt.tight_layout()


		if i<=1:
			plt.pause(5)
		else:
			plt.pause(1e-4)
		if i!=len(theta1_res)-1:
			plt.clf()
		else:
			plt.show()
	