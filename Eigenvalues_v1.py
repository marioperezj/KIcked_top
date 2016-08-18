import argparse
import pickle 
from math import *
from numpy import linalg as LA
import numpy as np
import cmath as cmat
#import matplotlib.pyplot as plt
#import matplotlib as mpl
from scipy import linalg as LA
parser = argparse.ArgumentParser(description='Produces eigenvalues for a kicked top')
parser.add_argument("-j","--j", help="Set the value for the maximum angular momentum on the z component",
                    type=float, default=10.0)
parser.add_argument("-k","--k", help="Set the constant k",
                    type=float, default=10.0)
parser.add_argument("-p","--p", help="Set the constant p",
                    type=float, default=1.7)        
parser.add_argument("-pos","--pos", help="Print only the eigenvalues with positive parity respect to the R_y trasnformation",
                    action='store_true')                              
parser.add_argument("-neg","--neg", help="Print only the eigenvalues with positive parity respect to the R_y trasnformation",
                    action='store_true')                     
args = parser.parse_args()  
l=args.j #Valor maximo del momento angular en la direccion z
p=args.p #parametro del hamiltoniano
k=args.k #parametro del Hamiltoniano
def print_array(x):
    for i in np.arange(0,np.size(x)):
        print(x[i])
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
def flo(j,k_var,p_var): # construye el operador de floquet para el hamiltoniano dado, ya devuleve la matriz
    return np.dot(LA.expm(complex(0,-k_var/(2*j))*mat(j,square_J_z)),LA.expm(complex(0,-p_var)*mat(j,J_y)))
def R(j,ope): #operador que genera la transformacion R(y),la cual es la matriz dentro de la funcion
    return LA.expm(complex(0,-pi)*mat(j,ope))
def hermitian(matri): #calculando el hermitiano de un operador o estado
    return np.transpose(np.conj(matri))    
def neg_block(j,n): #construccion de los bloques postivo y negativo del operador de floquet
    if j%2==0 and (j).is_integer():
       return n[int(j):int(2*j+1),int(j):int(2*j+1)]
    elif (j).is_integer():
       return n[0:int(j+1),0:int(j+1)]
    else:
       return n[int(j+0.5):int(2*j+1),int(j+0.5):int(2*j+1)]
def upper_zero_block(j,n):
    if j%2==0 and (j).is_integer():
        return n[0:int(j),int(j):int(2*j+1)]
    elif (j).is_integer():
        return n[0:int(j+1),int(j+1):int(2*j+1)]
    else:
        return n[int(j+0.5):int(2*j+1),0:int(j+0.5)]
def lower_zero_block(j,n):
    if j%2==0 and (j).is_integer():
        return n[int(j):int(2*j+1),0:int(j)]
    elif (j).is_integer():
        return n[int(j+1):int(2*j+1),0:int(j+1)]
    else:
        return n[0:int(j+0.5),int(j+0.5):int(2*j+1)]
def pos_block(j,n):
    if j%2==0 and (j).is_integer():
       return n[0:int(j),0:int(j)]
    elif (j).is_integer():
       return n[int(j+1):int(2*j+1),int(j+1):int(2*j+1)]
    else:
       return n[0:int(j+0.5),0:int(j+0.5)]
def norm_mat(x):
    return np.sum(np.square(np.absolute(x)))
 #esta funcion crea las quasienergias del operador de floquet, las quasienergias estan separadas en 
def sort_order(j):
    x1,y=LA.eig(R(j,J_y)) # un bloque de paridad positiva y otro de paridad negativa, al final regresa un vector que 
    if (j).is_integer():
        return np.real(x1)
    else:
        return np.imag(x1)    
def new_flo(j,k_var,p_var):
    x1,y=LA.eig(R(j,J_y))
    y_sort=y[:,sort_order(j).argsort()] # las quasienergias hacer return dist_eig
    return chabas(flo(j,k_var,p_var),y_sort)
def block_eig(pari):
    x,y=LA.eig(pari)
    r,eig_ene=cart2pol(np.real(x),np.imag(x))
    return np.sort(eig_ene)
def block_r(pari):
    x,y=LA.eig(pari)
    r,eig_ene=cart2pol(np.real(x),np.imag(x))
    return r    
def block_dim_pos(j):
    if j%2==0 and (j).is_integer():
        return int(j)
    elif (j).is_integer():
        return int(j)
    else:
        return int(j+0.5)
def block_dim_neg(j):
    if j%2==0 and (j).is_integer():
        return int(j+1)
    elif (j).is_integer():
        return int(j+1)
    else:
        return int(j+0.5)
print('This results was obtained with this values:')
print("j=%f" % args.j)
print("k=%f"%args.k)
print("p=%f"%args.p)
if args.pos:
    print('Positive block eigenvalues')
    print_array(block_eig(pos_block(l,new_flo(l,k,p))))
elif args.neg:
    print('Negative block eigenvalues')
    print_array(block_eig(neg_block(l,new_flo(l,k,p))))
else:
    print('Positive block eigenvalues')
    print_array(block_eig(pos_block(l,new_flo(l,k,p))))
    print('Negative block eigenvalues')
    print_array(block_eig(neg_block(l,new_flo(l,k,p))))    
#plt.matshow(np.real(new_flo(l,k,p)),vmin=-1.5,vmax=1.5)
#plt.colorbar()
#plt.matshow(np.real(pos_block(l,new_flo(l,k,p))),vmin=-1.5,vmax=1.5)
#plt.colorbar()
#plt.matshow(np.real(neg_block(l,new_flo(l,k,p))),vmin=-1.5,vmax=1.5)
#plt.colorbar()
#plt.show()
