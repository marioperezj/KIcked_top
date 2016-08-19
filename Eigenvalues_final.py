import argparse
from math import *
from numpy import linalg as LA
import numpy as np
import cmath as cmat
#import matplotlib.pyplot as plt
#import matplotlib as mpl
from scipy import linalg as LA
parser = argparse.ArgumentParser(description='Produces eigenvalues for a kicked top')
"""Necesary arguments to import the parameters from the command line"""
parser.add_argument("-j","--j", help="Set the value for the maximum angular momentum on the z component",
                    type=float, default=10.0)
"""Importing the maximum value for the angular momentum"""
parser.add_argument("-k","--k", help="Set the constant k",
                    type=float, default=10.0)
"""Importing the value of the kick paramater"""
parser.add_argument("-p","--p", help="Set the constant p",
                    type=float, default=1.7)
"""Importing the value for the rotation parameter."""        
parser.add_argument("-pos","--pos", help="Print only the eigenvalues with positive parity respect to the R_y trasnformation",
                    action='store_true')          
"""Gives the option of print only the positive parity eigenvalues"""                    
parser.add_argument("-neg","--neg", help="Print only the eigenvalues with positive parity respect to the R_y trasnformation",
                    action='store_true')                     
"""Gives the option of print only the negative parity eigenvalues"""
args = parser.parse_args()  
l=args.j 
p=args.p 
k=args.k 
"""Giving short names to the parameters"""
def print_array(x):
    """This function allows the user to print an array as a list"""
    for i in np.arange(0,np.size(x)):
        print(x[i])
def unitary(x):
    """This functions test if an operator is unitary"""
    return np.dot(x,hermitian(x))
def cart2pol(x, y):
    """This function applies a change of basis from cartesian to polar coordinates. this function takes the values x and y and return r and theta"""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)+pi
    return(rho, phi)
def chabas(x,y):       
    """This function applies a change of representation of an operator under a change of basis using the transformation y^-1xy where x is the operator  in the old basis and y is the new basis"""
    z=np.zeros_like(x) 
    z=np.dot(LA.inv(y),np.dot(x,y))
    return z
def ket(j_var,m_var): 
    """This function generates the matrix representation of a ket with the basis of eigenvalues of J^2 and j_z it means |jm>"""
    x=np.zeros(int(round((2*j_var+1))),dtype=complex)
    x[int(round(j_var+m_var))]=1
    return x
def square_J_z(j_var,m_var): 
    """This fucntion generates the J_z operator in the |jm> basis."""
    return ((m_var)**2)*ket(j_var,m_var)
def J_z(j_var,m_var): 
    """This fucntion generates the J_z operator in the |jm> basis."""
    return (m_var)*ket(j_var,m_var)    
def J_plus(j_var,m_var):
    """This fucntion generates the J_+ operator in the |jm> basis."""
    if (j_var==m_var):
        return 0*np.zeros(int(round(2*j_var+1)))
    else:
        return (sqrt((j_var-m_var)*(j_var+m_var+1)))*ket(j_var,m_var+1)    
def J_minus(j_var,m_var):
    """This fucntion generates the J- operator in the |jm> basis."""
    if (j_var==-m_var):
        return 0*np.zeros(int(round(2*j_var+1)))
    else:
        return (sqrt((j_var+m_var)*(j_var-m_var+1)))*ket(j_var,m_var-1)   
def J_x(j_var,m_var):
    """This fucntion generates the J_x operator in the |jm> basis."""
    if (j_var==m_var):
        return (0.5)*(sqrt((j_var+m_var)*(j_var-m_var+1)))*ket(j_var,m_var-1)
    elif (j_var==-m_var):
        return (0.5)*(sqrt((j_var-m_var)*(j_var+m_var+1)))*ket(j_var,m_var+1)
    else:
        return (0.5)*(sqrt((j_var-m_var)*(j_var+m_var+1)))*ket(j_var,m_var+1)+(0.5)*(sqrt((j_var+m_var)*(j_var-m_var+1)))*ket(j_var,m_var-1)
def J_y(j_var,m_var):
    """This fucntion generates the J_y operator in the |jm> basis."""
    if (j_var==m_var):
        return (complex(0,0.5))*(sqrt((j_var+m_var)*(j_var-m_var+1)))*ket(j_var,m_var-1)
    elif (j_var==-m_var):
        return (complex(0,-0.5))*(sqrt((j_var-m_var)*(j_var+m_var+1)))*ket(j_var,m_var+1)
    else:
        return (complex(0,0.5))*(sqrt((j_var+m_var)*(j_var-m_var+1)))*ket(j_var,m_var-1)+(complex(0,-0.5))*(sqrt((j_var-m_var)*(j_var+m_var+1)))*ket(j_var,m_var+1)
def mat(j,ope):
    """This functions generates the matrix representation of the above operators in the |jm> basis. IMPORTANT: Check that the "higgest" value is the lowest one """
    matrix=np.zeros((int(round(2*j+1)),int(round(2*j+1))),dtype=complex)
    for i in range(int(round(2*j+1))):
        for k in range(int(round(2*j+1))):
            matrix[i,k]=np.dot(ket(j,i-j),ope(j,k-j))
    return matrix     
def flo(j,k_var,p_var): 
    """This function returns the Floquet operator for the Kicked top model, this function return the operator as a matrix in the |jm> basis."""
    return np.dot(LA.expm(complex(0,-k_var/(2*j))*mat(j,square_J_z)),LA.expm(complex(0,-p_var)*mat(j,J_y)))
def R(j,ope): 
    """This function returns the matrix respresentation of the operator R_y. This operator is a rotation around the y axis by 180 degrees."""
    return LA.expm(complex(0,-pi)*mat(j,ope))
def hermitian(matri): 
    """This function returns the hermitian conjugate of an operator or ket."""
    return np.transpose(np.conj(matri))    
def neg_block(j,n): 
    """This function select and returns the negative eigenvalues block of the Floquet operator """
    if j%2==0 and (j).is_integer():
       return n[int(j):int(2*j+1),int(j):int(2*j+1)]
    elif (j).is_integer():
       return n[0:int(j+1),0:int(j+1)]
    else:
       return n[int(j+0.5):int(2*j+1),int(j+0.5):int(2*j+1)]
def upper_zero_block(j,n):
    """The function selects the upper block of the matrix. This block should be full with zeros"""
    if j%2==0 and (j).is_integer():
        return n[0:int(j),int(j):int(2*j+1)]
    elif (j).is_integer():
        return n[0:int(j+1),int(j+1):int(2*j+1)]
    else:
        return n[int(j+0.5):int(2*j+1),0:int(j+0.5)]
def lower_zero_block(j,n):
    """The function selects the upper block of the matrix. This block should be full with zeros""" 
    if j%2==0 and (j).is_integer():
        return n[int(j):int(2*j+1),0:int(j)]
    elif (j).is_integer():
        return n[int(j+1):int(2*j+1),0:int(j+1)]
    else:
        return n[0:int(j+0.5),int(j+0.5):int(2*j+1)]
def pos_block(j,n):
    """This block is the positive parity part of the Floquet operator"""
    if j%2==0 and (j).is_integer():
       return n[0:int(j),0:int(j)]
    elif (j).is_integer():
       return n[int(j+1):int(2*j+1),int(j+1):int(2*j+1)]
    else:
       return n[0:int(j+0.5),0:int(j+0.5)]
def norm_mat(x):
    """This function returns the norm of a matrix calculated as the sum of every element in absolute value and squared."""
    return np.sum(np.square(np.absolute(x)))
 #esta funcion crea las quasienergias del operador de floquet, las quasienergias estan separadas en 
def sort_order(j):
    """This function defines the necessary order to transform the Floquet operator in a Block Matrix"""
    x1,y=LA.eig(R(j,J_y)) # un bloque de paridad positiva y otro de paridad negativa, al final regresa un vector que 
    if (j).is_integer():
        return np.real(x1)
    else:
        return np.imag(x1)    
def new_flo(j,k_var,p_var):
    """This is the Floquet operator divided into Blocks"""
    x1,y=LA.eig(R(j,J_y))
    y_sort=y[:,sort_order(j).argsort()] # las quasienergias hacer return dist_eig
    return chabas(flo(j,k_var,p_var),y_sort)
def block_eig(pari):
    """This function return the eigenvalues of a specific part of the FLoquet operator, the arguments into the function could be 
the functions pos_block or neg_block"""
    x,y=LA.eig(pari)
    r,eig_ene=cart2pol(np.real(x),np.imag(x))
    return np.sort(eig_ene)
def block_r(pari):
    """This funcion return the norm of the complex eigenvalues of a specific block of the Floquet operator."""
    x,y=LA.eig(pari)
    r,eig_ene=cart2pol(np.real(x),np.imag(x))
    return r    
def block_dim_pos(j):
    """This function return the dimension of the matrix of the positive Block of the Floquet operator"""
    if j%2==0 and (j).is_integer():
        return int(j)
    elif (j).is_integer():
        return int(j)
    else:
        return int(j+0.5)
def block_dim_neg(j):
    """This function return the dimension of the matrix of the negative Block of the Floquet operator"""
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
