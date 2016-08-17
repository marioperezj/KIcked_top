import argparse
import pickle 
from math import *
from numpy import linalg as LA
import numpy as np
import cmath as cmat
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg as LA
parser = argparse.ArgumentParser(description='Produces eigenvalues for a kicked top')
parser.add_argument("-file", help="Put file to print the eigenvalues into a file", action='store_true')
parser.add_argument("-j","--j", help="Set the value for the maximum angular momentum on the z component",
                    type=float)
parser.add_argument("-k","--k", help="Set the constant k",
                    type=float)
parser.add_argument("-p","--p", help="Set the constant p",
                    type=float)                                      
args = parser.parse_args()  
l=args.j #Valor maximo del momento angular en la direccion z
p=args.p #parametro del hamiltoniano
k=args.k #parametro del Hamiltoniano
def unitary(x):
    return np.dot(x,hermitian(x))
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)+pi
    return(rho, phi)
def chabas(x,y):       #funcion que aplica un cambio de base de la matriz x 
    z=np.zeros_like(x) #con el cambio y
    z=np.dot(LA.inv(y),np.dot(x,y))
    return z
def ket(j_var,m_var): #generando los kets en la base |jm>
    x=np.zeros(int(round((2*j_var+1))),dtype=complex)
    x[int(round(j_var+m_var))]=1
    return x
def square_J_z(j_var,m_var): #operador J_z^2
    return ((m_var)**2)*ket(j_var,m_var)
def J_z(j_var,m_var): 
    return (m_var)*ket(j_var,m_var)    
def J_plus(j_var,m_var):
    if (j_var==m_var):
        return 0*np.zeros(int(round(2*j_var+1)))
    else:
        return (sqrt((j_var-m_var)*(j_var+m_var+1)))*ket(j_var,m_var+1)    
def J_minus(j_var,m_var):
    if (j_var==-m_var):
        return 0*np.zeros(int(round(2*j_var+1)))
    else:
        return (sqrt((j_var+m_var)*(j_var-m_var+1)))*ket(j_var,m_var-1)   
def J_x(j_var,m_var):
    if (j_var==m_var):
        return (0.5)*(sqrt((j_var+m_var)*(j_var-m_var+1)))*ket(j_var,m_var-1)
    elif (j_var==-m_var):
        return (0.5)*(sqrt((j_var-m_var)*(j_var+m_var+1)))*ket(j_var,m_var+1)
    else:
        return (0.5)*(sqrt((j_var-m_var)*(j_var+m_var+1)))*ket(j_var,m_var+1)+(0.5)*(sqrt((j_var+m_var)*(j_var-m_var+1)))*ket(j_var,m_var-1)
def J_y(j_var,m_var):
    if (j_var==m_var):
        return (complex(0,0.5))*(sqrt((j_var+m_var)*(j_var-m_var+1)))*ket(j_var,m_var-1)
    elif (j_var==-m_var):
        return (complex(0,-0.5))*(sqrt((j_var-m_var)*(j_var+m_var+1)))*ket(j_var,m_var+1)
    else:
        return (complex(0,0.5))*(sqrt((j_var+m_var)*(j_var-m_var+1)))*ket(j_var,m_var-1)+(complex(0,-0.5))*(sqrt((j_var-m_var)*(j_var+m_var+1)))*ket(j_var,m_var+1)
def mat(j,ope):
    matrix=np.zeros((int(round(2*j+1)),int(round(2*j+1))),dtype=complex)
    for i in range(int(round(2*j+1))):
        for k in range(int(round(2*j+1))):
            matrix[i,k]=np.dot(ket(j,i-j),ope(j,k-j))
    return matrix     
def flo(j,k,p): # construye el operador de floquet para el hamiltoniano dado, ya devuleve la matriz
    return np.dot(LA.expm(complex(0,-k/(2*j))*mat(j,square_J_z)),LA.expm(complex(0,-p)*mat(j,J_y)))
def R(j,ope): #operador que genera la transformacion R(y),la cual es la matriz dentro de la funcion
    return LA.expm(complex(0,-pi)*mat(j,ope))
def hermitian(matri): #calculando el hermitiano de un operador o estado
    return np.transpose(np.conj(matri))    
def pos_block(m,n): #construccion de los bloques postivo y negativo del operador de floquet
    if m%2==0 and (m).is_integer():
       return n[int(l):int(2*l+1),int(l):int(2*l+1)]
    elif (m).is_integer():
       return n[0:int(l+1),0:int(l+1)]
    else:
       return n[int(l+0.5):int(2*l+1),int(l+0.5):int(2*l+1)]
def neg_block(m,n):
    if m%2==0 and (m).is_integer():
       return n[0:int(l),0:int(l)]
    elif (m).is_integer():
       return n[int(l+1):int(2*l+1),int(l+1):int(2*l+1)]
    else:
       return n[0:int(l+0.5),0:int(l+0.5)]
 #esta funcion crea las quasienergias del operador de floquet, las quasienergias estan separadas en 
x1,y=LA.eig(R(l,J_y)) # un bloque de paridad positiva y otro de paridad negativa, al final regresa un vector que 
if (l).is_integer():
    x=np.real(x1)
else:
    x=np.imag(x1)    
y_sort=y[:,x.argsort()] # las quasienergias hacer return dist_eig
new_flo=chabas(flo(l,k,p),y_sort)
pos_new_flo=pos_block(l,new_flo)
neg_new_flo=neg_block(l,new_flo)
eig_vals_pos,eig_vecs_pos=LA.eig(pos_new_flo)
eig_vals_neg,eig_vecs_neg=LA.eig(neg_new_flo)
r_pos,eig_ene_pos1=cart2pol(np.real(eig_vals_pos),np.imag(eig_vals_pos))
r_neg,eig_ene_neg1=cart2pol(np.real(eig_vals_neg),np.imag(eig_vals_neg))
eig_ene_pos=np.sort(eig_ene_pos1)
eig_ene_neg=np.sort(eig_ene_neg1)
eig_ene=np.append(eig_ene_pos,eig_ene_neg)
print(eig_ene)
#plt.matshow(np.real(new_flo),vmin=-1.5,vmax=1.5)
#plt.colorbar()
#plt.matshow(np.real(pos_new_flo),vmin=-1.5,vmax=1.5)
#plt.colorbar()
#plt.matshow(np.real(neg_new_flo),vmin=-1.5,vmax=1.5)
#plt.colorbar()
#plt.show()
