#!/usr/bin/python3
# Remilia Grimm
# grimm.remilia@gmail.com
##########################
#   Suggested Reading
#   (included in the repo):
#
#   J Gutzelius'
#   The Three Body Problem
#
#   J Worthington's
#   A Study of the Planar Circular
#   Restricted Three Body Problem
#   And the Vanishing Twist
#
#######################################################
# Maths and original pythonic modeling by Gauruv Deshmukh
# Link:
# https://towardsdatascience.com/modelling-the-three-body-problem-in-classical-mechanics-using-python-9dc270ad7767
#
########################################################

import scipy as sp
import numpy as n
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

#Universal constant for gravitation
g=6.6740e-11 #N-m2/kg2

#Modeling from blog:
#Reference quantities
m_nd=1.989e+30 #kg Solar Mass
r_nd=5.326e+12 #m Distance between bodies
v_nd=30000 #m/s Velocity of Terra wrt Sol, relative
t_nd=79.91*365*24*3600*0.51 #s Alpha's Centauri's orbit
# Constants
K1=g*t_nd*m_nd/(r_nd**2*v_nd)
K2=v_nd*t_nd/r_nd

#Define masses
m1=1.1 #AC A
m2=0.907
m3=1.0

#AC B #Define initial position vectors

r1=[-0.5,0,0]
r2=[0.5,0,0]
r3=[0,1,0]

#Convert pvectors to arrays
r1=n.array(r1,dtype="float64")
r2=n.array(r2,dtype="float64")
r3=n.array(r3,dtype="float64")

#Find centroid
r_com=(m1*r1+m2*r2)/(m1+m2)#Define initial velocities
v1=[0.01,0.01,0] #m/s
v2=[-0.05,0,-0.1] #m/s#Velocity -> to arrays
v3=[0,-0.01,0]

#Find velocity of centroids
v1=n.array(v1,dtype="float64")
v2=n.array(v2,dtype="float64")
v3=n.array(v3,dtype="float64")
v_com=(m1*v1+m2*v2)/(m1+m2)

#Gauruv's odeint modeling and equations

#Definition: Equations of motion
def TwoBodyMotion(w,t,g,m1,m2):
    r1=w[:3]
    r2=w[3:6]
    v1=w[6:9]
    v2=w[9:12]

    r=sp.linalg.norm(r2-r1) # Find vector's magnitude
    #Good excuse for me to do some Calculus review, too <3
    r3=(r2-r1)
    r4=(r1-r2)

    dv1bydt=K1*m2*r3/r**3
    dv2bydt=K1*m1*r4/r**3
    dr1bydt=K2*v1
    dr2bydt=K2*v2

    r_dx=n.concatenate((dr1bydt,dr2bydt))
    dx=n.concatenate((r_dx,dv1bydt,dv2bydt))
    return dx

#Package initial parameters
init_parameters=n.array([r1,r2,v1,v2]) #parameter array
init_parameters=init_parameters.flatten() #Make the array 1-dimensional
time_span=n.linspace(0,8,500) #8 Orbital routes and their periods
import scipy.integrate
twobodyresult=sp.integrate.odeint(TwoBodyMotion,init_parameters,time_span,args=(g,m1,m2))

r1_sol=twobodyresult[:,:3]
r2_sol=twobodyresult[:,3:6]

# Plot from blog
# Create figure
fig=plot.figure(figsize=(15,15))#Create 3D axes
ax=fig.add_subplot(111,projection="3d")#Plot the orbits
ax.plot(r1_sol[:,0],r1_sol[:,1],r1_sol[:,2],color="darkblue")
ax.plot(r2_sol[:,0],r2_sol[:,1],r2_sol[:,2],color="tab:red")#Plot the final positions of the stars
ax.scatter(r1_sol[-1,0],r1_sol[-1,1],r1_sol[-1,2],color="darkblue",marker="o",s=100,label="Alpha Centauri A")
ax.scatter(r2_sol[-1,0],r2_sol[-1,1],r2_sol[-1,2],color="tab:red",marker="o",s=100,label="Alpha Centauri B")#Add a few more bells and whistles
ax.set_xlabel("x-coordinate",fontsize=14)
ax.set_ylabel("y-coordinate",fontsize=14)
ax.set_zlabel("z-coordinate",fontsize=14)
ax.set_title("Visualization of orbits of stars in a two-body system\n",fontsize=14)
ax.legend(loc="upper left",fontsize=14)

#Find location of Centroid
#rcom_solution=(m1*r1_solution+m2*r2_solution)/(m1+m2)
#Find location of AC A w.r.t Centroid
#r1com_solution=r1_sol-rcom_solution
#Location of AC B w.r.t Centroid
#r2com_solution=r2_solution-rcom_solution

# Following the modeling:
# Quantities for reference
m_nd=1.989e+30 #kg #mass of the sun
r_nd=5.326e+12 #m #distance between stars in Alpha Centauri
v_nd=30000 #m/s #relative velocity of Eart wrt Sol
t_nd=79.91*365*24*3600*0.51 #s #orbital period of AC

#Constants
K1=g*t_nd*m_nd/(r_nd**2*v_nd)
K2=v_nd*t_nd/r_nd

# Three body Centroid data
r_com=(m1*r1+m2*r2+m3*r3)/(m1+m2+m3)
# velocity of three centroids
v_com=(m1*v1+m2*v2+m3*v3)/(m1+m2+m3)


def ThreeBodyEquations(w, t, G, m1, m2, m3):
    r1 = w[:3]
    r2 = w[3:6]
    r3 = w[6:9]
    v1 = w[9:12]
    v2 = w[12:15]
    v3 = w[15:18]
    r12 = sp.linalg.norm(r2 - r1)
    r13 = sp.linalg.norm(r3 - r1)
    r23 = sp.linalg.norm(r3 - r2)

    dv1bydt = K1 * m2 * (r2 - r1) / r12 ** 3 + K1 * m3 * (r3 - r1) / r13 ** 3
    dv2bydt = K1 * m1 * (r1 - r2) / r12 ** 3 + K1 * m3 * (r3 - r2) / r23 ** 3
    dv3bydt = K1 * m1 * (r1 - r3) / r13 ** 3 + K1 * m2 * (r2 - r3) / r23 ** 3
    dr1bydt = K2 * v1
    dr2bydt = K2 * v2
    dr3bydt = K2 * v3
    r12_derivs = n.concatenate((dr1bydt, dr2bydt))
    r_derivs = n.concatenate((r12_derivs, dr3bydt))
    v12_derivs = n.concatenate((dv1bydt, dv2bydt))
    v_derivs = n.concatenate((v12_derivs, dv3bydt))
    derivation = n.concatenate((r_derivs, v_derivs))
    return derivation

#Parameters
init_params=n.array([r1,r2,r3,v1,v2,v3]) #Initial parameters
init_params=init_params.flatten() #Create a 1-dimensional vector
time_span=n.linspace(0,20,500) #20 orbital periods and 500 points
import scipy.integrate
threeBodyEquations=sp.integrate.odeint(ThreeBodyEquations,init_params,time_span,args=(g,m1,m2,m3))

r1_sol=threeBodyEquations[:,:3]
r2_sol=threeBodyEquations[:,3:6]
r3_sol=threeBodyEquations[:,6:9]

fig=plot.figure(figsize=(15,15))#Create 3D axes
ax=fig.add_subplot(111,projection="3d")#Plot the orbits
ax.plot(r1_sol[:,0],r1_sol[:,1],r1_sol[:,2],color="darkblue")
ax.plot(r2_sol[:,0],r2_sol[:,1],r2_sol[:,2],color="tab:red")#Plot the final positions of the stars
ax.plot(r3_sol[:,0],r3_sol[:,1],r3_sol[:,2],color="purple")
ax.scatter(r1_sol[-1,0],r1_sol[-1,1],r1_sol[-1,2],color="darkblue",marker="o",s=100,label="Alpha Centauri A")
ax.scatter(r2_sol[-1,0],r2_sol[-1,1],r2_sol[-1,2],color="tab:red",marker="o",s=100,label="Alpha Centauri B")#Add a few more bells and whistles
ax.scatter(r3_sol[-1,0],r3_sol[-1,1],r3_sol[-1,2],color="purple",marker="o",s=100,label="Sol")
ax.set_xlabel("x-coordinate",fontsize=14)
ax.set_ylabel("y-coordinate",fontsize=14)
ax.set_zlabel("z-coordinate",fontsize=14)
ax.set_title("Visualization of orbits of stars in a three-body system\n",fontsize=14)
ax.legend(loc="upper left",fontsize=14)