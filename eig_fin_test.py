# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 14:56:42 2016

@author: mario
"""
from math import *
from numpy import linalg as LA
import numpy as np
import cmath as cmat
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg as LA
import Eigenvalues_final
import unittest

class KnownValues(unittest.TestCase):
    """Test known values for many functions defined in Eigenvalues_final.py"""
    KnownValuesket= ((0,0,np.array([complex(1,0)])),
                  (1,1,np.array([complex(0,0),complex(0,0),complex(1,0)])))
    KnownValuesJ_x= np.array([[complex(0,0),complex(0.5,0.0)],[complex(0.5,0.0),complex(0.0,0.0)]])
    KnownValuesJ_y= np.array([[complex(0,0),complex(0.0,0.5)],[complex(0.0,-0.5),complex(0.0,0.0)]])
    KnownValuesJ_z= np.array([[complex(-0.5,0),complex(0.0,0.0)],[complex(0.0,0.0),complex(0.5,0.0)]])

    def testketvalues(self):
        """Test ket values for some knownvalues"""
        for j , m , mat in self.KnownValuesket:
            result=Eigenvalues_final.ket(j,m)
            self.assertTrue((mat==result).all())
    def testJ_xvalues(self):
        """Test J_x values for some knownvalues"""
        self.assertTrue((self.KnownValuesJ_x==Eigenvalues_final.mat(0.5,Eigenvalues_final.J_x)).all())
    def testJ_yvalues(self):
        """Test J_y values for some knownvalues"""
        self.assertTrue((self.KnownValuesJ_y==Eigenvalues_final.mat(0.5,Eigenvalues_final.J_y)).all())        
    def testJ_zvalues(self):
        """Test J_z values for some knownvalues"""
        self.assertTrue((self.KnownValuesJ_z==Eigenvalues_final.mat(0.5,Eigenvalues_final.J_z)).all())
        
class Sanity(unittest.TestCase):
    val_flo= ((0.5,10.0,1.7),
              (1.0,10.0,1.7),
              (1.5,10.0,1.7),
              (2.0,5.0,1.7),
              (2.5,0.3,1.7))
    generic= np.array([[1,1],[1,1]])
    generic1=np.array([[2,1],[2,2]])
    def testflounitarity(self):
        """Test unitarity of the Floquet operator"""
        for j , k , p in self.val_flo:
            self.assertTrue((np.around(np.dot(Eigenvalues_final.flo(j,k,p),Eigenvalues_final.hermitian(Eigenvalues_final.flo(j,k,p))))==np.identity(int(2*j+1),dtype=complex)).all())              
    def testhermitian(self):
        """Test hermitian function"""
        self.assertTrue((Eigenvalues_final.hermitian(Eigenvalues_final.hermitian(self.generic))==self.generic).all())   
    def testchabas(self):
        """Test change of basis function"""
        self.assertTrue((Eigenvalues_final.chabas(Eigenvalues_final.chabas(self.generic,self.generic1),LA.inv(self.generic1))==self.generic).all())
    def testmat(self):
        """Test if the matrix fuction returns a square matrix"""
        self.assertTrue(np.shape(Eigenvalues_final.mat(0.5,Eigenvalues_final.J_minus))[0]==np.shape(Eigenvalues_final.mat(0.5,Eigenvalues_final.J_minus))[1])
class Block(unittest.TestCase):
    ref_val= 1.0e-24
    spin=((0.5,0.3,1.7),
          (1.0,0.5,1.7),
          (1.5,10.0,1.7),
          (2.0,5.0,1.7),
          (2.5,0.3,1.7)
          )
    """This Class contains the functions that test the block form of the Floquet matrix"""
    def testblockupper(self):
        """Test upper Floquet block"""
        for j,k,p in self.spin:
            self.assertTrue(Eigenvalues_final.norm_mat(Eigenvalues_final.upper_zero_block(j,Eigenvalues_final.new_flo(j,k,p)))<=self.ref_val)
    def testblocklower(self):
        """Test lower Floquet block"""
        for j,k,p in self.spin:
            self.assertTrue(Eigenvalues_final.norm_mat(Eigenvalues_final.lower_zero_block(j,Eigenvalues_final.new_flo(j,k,p)))<=self.ref_val)
class Quasi(unittest.TestCase):
    spin=((0.5,0.3,1.7),
          (1.0,0.5,1.7),
          (1.5,10.0,1.7),
          (2.0,5.0,1.7),
          (2.5,0.3,1.7)
          )
    def test_ones_neg(self):            
        """Test r vales of the eigenvalues equal to one in the negative block"""
        for j,k,p in self.spin:
            self.assertTrue(np.array_equal(np.around(Eigenvalues_final.block_r(Eigenvalues_final.neg_block(j,Eigenvalues_final.new_flo(j,k,p)))),np.ones(Eigenvalues_final.block_dim_neg(j))))
    def test_ones_pos(self):            
        """Test r vales of the eigenvalues equal to one in the positive block"""
        for j,k,p in self.spin:
            self.assertTrue(np.array_equal(np.around(Eigenvalues_final.block_r(Eigenvalues_final.pos_block(j,Eigenvalues_final.new_flo(j,k,p)))),np.ones(Eigenvalues_final.block_dim_pos(j))))
    def test_angle_neg(self):
        for j,k,p in self.spin:
            self.assertTrue(0.0<(Eigenvalues_final.block_eig(Eigenvalues_final.neg_block(j,Eigenvalues_final.new_flo(j,k,p)))).all()<2*pi)
    def test_angle_pos(self):
        for j,k,p in self.spin:
            self.assertTrue(0.0<(Eigenvalues_final.block_eig(Eigenvalues_final.pos_block(j,Eigenvalues_final.new_flo(j,k,p)))).all()<2*pi)
if __name__ == "__main__":
     unittest.main()
