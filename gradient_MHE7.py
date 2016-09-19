import dolfin as dl
import numpy as np
import scipy.sparse as sp
import math
import h5py
import pickle
import math
import optwrapper as ow
import ipdb
lib_load = 1

import dolfin_MHE
import qp_MHE
reload( dolfin_MHE )
reload( qp_MHE )

def kernal_djdk( filename, ke_list, ke_values ):
    # kernal matrix for djdk
    mesh = dl.Mesh( filename+'.xml' )
    T = dl.FunctionSpace( mesh, "CG", 2 )
    V0 = dl.FunctionSpace( mesh, "DG", 0 )
    ke = dl.Function( V0 )
    de = dl.TestFunction( T )
    te = dl.TrialFunction( T )
    n_v0 = V0.dofmap().ownership_range()[1] - V0.dofmap().ownership_range()[0] 
    coeff = np.zeros( n_v0 )
    coeff[ ke_list ] = ke_values[0] - ke_values[1]
    ke.vector().set_local( coeff )
    # dl.plot( ke, title="ke_i", interactive=True )     
    ltt_express = ke*dl.inner( dl.grad(te), dl.grad(de) )*dl.dx
    # ltt[i,j]: i, de( lambda_1 ); j, te( T )
    ltt_mat = dl.assemble( ltt_express )
    return ltt_mat.array()

def kernal_djda( filename, alpha_list, alpha_values ):
    # kernal matrix for djda
    mesh = dl.Mesh( filename+'.xml' )
    V = dl.VectorFunctionSpace( mesh,"CG",2 )
    V0 = dl.FunctionSpace( mesh, "DG", 0 )
    alpha = dl.Function( V0 )
    de = dl.TestFunction( V )
    te = dl.TrialFunction( V )
    n_v0 = V0.dofmap().ownership_range()[1] - V0.dofmap().ownership_range()[0]
    coeff = np.zeros( n_v0 )
    coeff[ alpha_list ] = alpha_values[0] - alpha_values[1]
    alpha.vector().set_local( coeff )
    # dl.plot( alpha, title="alpha_i", interactive=True )
    ltt_express = alpha*dl.inner( de,te )*dl.dx
    # ltt[i,j]: i, de( lambda_2 ); j, te( u )
    ltt_mat = dl.assemble( ltt_express )
    return ltt_mat.array()

def kernal_djdi( filename, bp0, bp1, br=1.0 ):
    # kernal matrix for partial djdi0
    mesh = dl.Mesh( filename+'.xml' )
    g1 = dolfin_MHE.bump_sensor( bp0, bp1, br )
    V = dl.FunctionSpace( mesh,"CG",2 )
    de = dl.TestFunction( V )
    te = dl.TrialFunction( V )
    l_express = g1*dl.inner( de,te )*dl.dx
    # l[i,j]: i, i0-i^0; j, delta_i0
    l_mat = dl.assemble( l_express )
    return l_mat.array()
    
def assemble_djdk( kernal_matrix, t_v, t_adj, dt, Tf ):
    # assemble djdk for theta_ke[i]
    djdk = 0
    i = 1
    while ( i*dt <= Tf ): # backward Eular
        djdk += dt*t_adj[i,:].dot(
            kernal_matrix.dot( t_v[i,:] ) )
        i += 1
    return djdk
        
def assemble_djda( kernal_matrix, u_v, u_adj ):
    # assemble djda for theta_alpha[i]
    djda = u_adj.dot( kernal_matrix.dot( u_v ) )   
    return djda

def assemble_djdi( pjpi_list, kernal_lii, t_adj7, t_v0, t_s0, yeta0, yeta1 ):
    # assemble djdi for delta_i0
    djdi = -1.0*t_adj7.dot( kernal_lii )
    dt_v0 = t_v0 - t_s0
    for kernal_pjpi in pjpi_list:
        djdi += 2.0 * yeta0 * dt_v0.dot( kernal_pjpi )
    djdi += 2.0 * yeta1 * t_v0.dot( kernal_lii )
    # djdi += 2.0 * yeta1 * dt_v0.dot( kernal_lii )
    return djdi

def assemble_qpLt( djdk_list, djda_list, t_v,
                   u_v, t_adj, u_adj, dt, Tf ):
    # assemble L vector for QP of theta
    qpL = np.zeros( len( djdk_list ) )
    for i in range( len( djdk_list ) ):
        qpL[ i ] = assemble_djdk( djdk_list[i],
                                  t_v, t_adj, dt, Tf ) + assemble_djda(
                                      djda_list[i], u_v, u_adj )        
    return qpL

def assemble_qpLi( pjpi_list, kernal_lii, t_v, u_v,
                   t_adj, u_adj, t_s, yeta0, yeta1 ):
    # assemble L vector for QP of i0
    return assemble_djdi( pjpi_list,
                          kernal_lii, t_adj[0], t_v[0],
                          t_s[0], yeta0, yeta1 )

def assemble_qpLa( pjpi_list, kernal_lii, t_v, u_v,
                   t_adj, u_adj, t_s, e_x, tin_range,
                   yeta0, yeta1, indx ):
    tmp_qpL = assemble_djdi( pjpi_list, kernal_lii, t_adj[0],
                             t_v[0], t_s[0], yeta0, yeta1 )
    qpL = np.zeros( ( indx.n_t, ) )
    for i in range( indx.n_t ):
        qpL[ indx.i0( i+1 ) ] = tmp_qpL[ tin_range ].dot( e_x[ i,tin_range ] )
    return qpL

def assemble_qpL( djdk_list, djda_list, pjpi_list,
                  kernal_lii, t_v, u_v, t_adj, u_adj,
                  t_s, e_x, tin_range, 
                  dt, Tf, yeta0, yeta1, indx ):
    # assemble L vector for QP, whole
    qpL = np.zeros( indx.num_v )
    qpL_tmp = assemble_qpLa( pjpi_list, kernal_lii, t_v, u_v,
                             t_adj, u_adj, t_s, e_x, tin_range,
                             yeta0, yeta1, indx )
    qpL[ 0:indx.theta(1) ] = qpL_tmp # reshape the points
    for i in range( len( djdk_list ) ):
        qpL[ indx.theta( i+1 ) ] = assemble_djdk(
            djdk_list[i], t_v, t_adj, dt, Tf ) + assemble_djda(
                djda_list[i], u_v, u_adj )        
    return qpL

class qp_indx:
    def __init__( self, n_t ):
        self.n_t = n_t
        self.num_v = n_t + 4
    def i0( self, i ):
        # I0 index, [ 0, n_t )
        return i-1
    def theta( self, i ):
        # alpha index, [ n_t, n_t+4 )
        return self.n_t + ( i-1 )

def wrap_var( x_p, n_t ):
    # qpindx and ke, alpha values are global variables, also e_x
    alpha = []
    ke = []
    for i in range( qpindx.num_v - qpindx.n_t ):
        alpha.append( ke_values[0] * x_p[ qpindx.theta( i+1 ) ] +
                      ke_values[1] * ( 1 - x_p[ qpindx.theta( i+1 ) ] ) )
        ke.append( ke2_values[0] * x_p[ qpindx.theta( i+1 ) ] +
                   ke2_values[1] * ( 1 - x_p[ qpindx.theta( i+1 ) ] ) )
    i0_value = np.zeros( n_t )
    for i in range( qpindx.n_t ):
        i0_value[ tin_pt ] += x_p[ qpindx.i0(i+1) ]*e_x[ i,tin_pt ]
    return ( i0_value, alpha, ke )

class fx:
    # get the cost functions value
    def __init__( self, mesh_drt, indx, yeta0, yeta1,
                  ke_values, ke2_values,
                  pjpi_list, kernal_lii, bp0, bp1 ):
        self.mesh_drt = mesh_drt
        self.indx = indx
        self.yeta0 = yeta0
        self.yeta1 = yeta1
        self.pjpi_list = pjpi_list
        self.lii = kernal_lii
        self.ke_values = ke_values
        self.ke2_values = ke2_values
        self.bp0 = bp0
        self.bp1 = bp1
    def value( self, x_p, t_s ):
        '''
        i0_value = x_p[ 0:self.indx.theta(1) ]
        alpha_value = []
        ke_value = []
        for i in range( 4 ):
            alpha_value.append( self.ke_values[0]*x_p[ self.indx.theta(i+1) ]
                                + ( 1 - x_p[ self.indx.theta(i+1) ] )
                                *self.ke_values[1] )
            ke_value.append( self.ke2_values[0]*x_p[ self.indx.theta(i+1) ]
                             + ( 1 - x_p[ self.indx.theta(i+1) ] )
                             *self.ke2_values[1] )
        '''
        # call wrap_var()
        [ i0_value, alpha_value, ke_value ] = wrap_var( x_p, n_t )  
        # ipdb.set_trace()
        [ t_v, u_v, p_v,
          t_ad, u_ad, p_ad ] = dolfin_MHE.dolfin_mhe(
              self.mesh_drt, i0_value, alpha_value,
              ke_value, t_s, self.bp0, self.bp1, 0 )
        out_put = 0
        # dt = 10 # time step, use the global variable
        for pjpi in self.pjpi_list: # several sensors
            for i in range( 1, t_s.shape[0] ): # backward Eular
                out_put += dt * ( t_v[i,:] - t_s[i,:] ).dot( pjpi.dot(
                    t_v[i,:] - t_s[i,:] ) )
            out_put += self.yeta0 * ( t_v[0,:] - t_s[0,:] ).dot( pjpi.dot(
                t_v[0,:] - t_s[0,:] ) )
        # out_put += self.yeta1 * ( t_v[0,:] - t_s[0,:] ).dot( self.lii.dot( t_v[0,:] - t_s[0,:] ) )
        out_put += self.yeta1 * t_v[0,:].dot( self.lii.dot( t_v[0,:] ) ) 
        return out_put

def objf(out, x):
    # ipdb.set_trace()
    tem0 = x[0]*ax0[0,:] + x[1]*ax1[0,:] + x[2]*axc[0,:]
    # initial time
    out[0] = yeta0 * ( tem0 - t_s[0,tin_pt] ).dot(
        pjpi_tmp.dot( tem0 - t_s[0,tin_pt] ) )
    out[0] += yeta1 * tem0.dot( lii[ np.ix_( tin_pt, tin_pt ) ].dot(
        tem0 ) ) 
    # later time
    for i in range( 1,Tf/dt+1 ):
        tmp_m = x[0]*ax0[i,:] + x[1]*ax1[i,:] + x[2]*axc[i,:] + b_o[i-1,tin_pt]
        out[0] += dt*( tmp_m-t_s[i,tin_pt] ).dot(
            pjpi_tmp.dot( tmp_m-t_s[i,tin_pt] ) )
        
def objg(out, x):
    # ipdb.set_trace()
    tem0 = x[0]*ax0[0,:] + x[1]*ax1[0,:] + x[2]*axc[0,:]
    ## initial time, ax+by+c
    out[ 0 ] = 2.0 * yeta0 * ax0[0,:].dot( pjpi_tmp.dot(
        tem0 - t_s[0,tin_pt] ) ) + 2.0 * yeta1 * ax0[0,:].dot(
            lii[ np.ix_( tin_pt, tin_pt ) ].dot( tem0 ) ) 
    out[ 1 ] = 2.0 * yeta0 * ax1[0,:].dot( pjpi_tmp.dot(
        tem0 - t_s[0,tin_pt] ) ) + 2.0 * yeta1 * ax1[0,:].dot(
            lii[ np.ix_( tin_pt, tin_pt ) ].dot( tem0 ) )
    out[ 2 ] = 2.0 * yeta0 * axc[0,:].dot( pjpi_tmp.dot(
        tem0 - t_s[0,tin_pt] ) ) + 2.0 * yeta1 * axc[0,:].dot(
            lii[ np.ix_( tin_pt, tin_pt ) ].dot( tem0 ) )
    ## later time
    for i in range( 1,Tf/dt+1 ):
        tmp_m = x[0]*ax0[i,:] + x[1]*ax1[i,:] + x[2]*axc[i,:] + b_o[i-1,tin_pt]
        out[ 0 ] += ( 2.0*dt ) * ax0[i,:].dot(
            pjpi_tmp.dot( tmp_m-t_s[i,tin_pt] ) )
        out[ 1 ] += ( 2.0*dt ) * ax1[i,:].dot(
            pjpi_tmp.dot( tmp_m-t_s[i,tin_pt] ) )
        out[ 2 ] += ( 2.0*dt ) * axc[i,:].dot(
            pjpi_tmp.dot( tmp_m-t_s[i,tin_pt] ) )

    
def grad_armijo( x0, t_s, fx, grad_f0, qpval,
                 alpha, beta, k_max = 30 ):
    # x0, x position
    # t_s, simulation data
    # fx, costfunction, function
    # grad_f0, costfunction's gradient at x0 position
    # alpha, Armijo's parameter, (0,1)
    # beta, Armijo's step size, (0,1)
    # k_max, gurantee stop
    # ipdb.set_trace()
    # [ i0_gd, alpha_gd, ke_gd ] = wrap_var( grad_f0, n_t )
    fx0 = fx.value( x0, t_s )
    k = 0
    fx_tmp = fx.value( x0 + ( beta**k )*grad_f0, t_s )
    # ipdb.set_trace()
    while ( fx_tmp - fx0 > alpha*( beta**k )*qpval
        and k < k_max ):
        # ipdb.set_trace()
        k = k+1
        fx_tmp = fx.value( x0 + ( beta**k )*grad_f0, t_s )
        # print str(k)
    if k >= k_max :
        print "Armijo reaches step limit."
        # ipdb.set_trace()
    return ( k, fx_tmp )

mesh_drt = "./test_geo/seperateTT3"
mesh = dl.Mesh( mesh_drt+'.xml' )
subdomains = dl.MeshFunction( "size_t", mesh, ( mesh_drt + "_physical_region.xml" ) )
boundaries = dl.MeshFunction( "size_t", mesh, ( mesh_drt + "_facet_region.xml" ) )

### cost function's (OP) parameter
yeta0 = 1.0
yeta1 = 0.08
bp0 = [ 13.0, 13.0, 12.0 ]
bp1 = [ 3.0, 6.0, 11.0 ]
alpha = 0.1
beta = 0.7
alpha_i = 0.05 # 1E-5
beta_i = 0.75
epsl = 1E-6 # 1E-5
lp_step = 20
# ipdb.set_trace()
V0 = dl.FunctionSpace( mesh,"DG",0 )
T = dl.FunctionSpace( mesh, "CG", 2 )
V = dl.VectorFunctionSpace( mesh,"CG",2 )
P = dl.FunctionSpace( mesh,"CG",1 )
W = V * P
n_t = T.dofmap().ownership_range()[1] - T.dofmap().ownership_range()[0] 

#################
### Generate simulation results
#################
i0_value = np.zeros( ( n_t, ) )
### 1 is open, 0 is closed
alpha_init = [ 1000, 1000, 1000, 1000 ]
ke_init = [ 0.001, 0.001, 0.001, 0.001 ]
drt = "./results/qh_0000_n"
# all open
adj_pt = 0
dt = 10
Tf = 300
ts_tmp = np.zeros( ( Tf/dt+1, n_t ) )
[ t_s, u_s, p_s,
  t_ad, u_ad, p_ad ] = dolfin_MHE.dolfin_mhe( mesh_drt, i0_value, alpha_init,
                                              ke_init, ts_tmp, bp0,
                                              bp1, adj_pt )
i0_value = t_s[1,:]
[ t_s, u_s, p_s,
  t_ad, u_ad, p_ad ] = dolfin_MHE.dolfin_mhe( mesh_drt, i0_value, alpha_init,
                                              ke_init, ts_tmp, bp0,
                                              bp1, adj_pt )
###############
### plot initial
###############
# de = dl.Function( T )
# de.vector()[:] = i0_value
# vit = dl.plot( de, interactive=True )
# vit.write_pdf( drt + '/init' )
# ipdb.set_trace()
#############
## checking the plots
'''
ke = dl.Function( T )

alpha_init = [ 0.01, 1000, 1000, 1000 ]
ke_init = [ 0.01, 0.001, 0.001, 0.001 ]
adj_pt = 1
[ t_v, u_v, p_v,
  t_ad, u_ad, p_ad ] = dolfin_MHE.dolfin_mhe( mesh_drt, i0_value, alpha_init,
                                              ke_init, t_s, bp0, bp1, adj_pt )
i = 0
t = dt
while t <= Tf:
    t += dt
    ke.vector().set_local( t_s[i,:] )
    dl.plot( ke, interactive=True )
    i += 1

i = 0
t = dt
while t <= Tf:
    t += dt
    ke.vector().set_local( t_v[i,:] )
    dl.plot( ke, interactive=True )
    i += 1

i = 0
t = dt
while t <= Tf:
    t += dt
    ke.vector().set_local( t_ad[i,:] )
    dl.plot( ke, interactive=True )
    i += 1

u = dl.Function( V )
u.vector().set_local( u_v )
dl.plot( u, interactive=True )

u.vector().set_local( u_ad )
dl.plot( u, interactive=True )

ipdb.set_trace()
'''
###########
### class indix 
###########
# mark unknown alpha, and initial conditions
ke = dl.Function(V0)
ke_values = [0.01, 1000]  # values of re in the two subdomains
alpha2_list = []
alpha4_list = []
alpha5_list = []
alpha6_list = []
for cell_no in range(len(subdomains.array())):
    if subdomains.array()[cell_no] == 518:
        ke.vector()[cell_no] = ke_values[1]
    else:
        ke.vector()[cell_no] = ke_values[0]
    if subdomains.array()[cell_no] == 471:
        ke.vector()[cell_no] = ke_values[1]
    if (subdomains.array()[cell_no] == 472):
        alpha2_list.append( cell_no )
    if (subdomains.array()[cell_no] == 474):
        alpha4_list.append( cell_no )
    if (subdomains.array()[cell_no] == 475):
        alpha5_list.append( cell_no )
    if (subdomains.array()[cell_no] == 476):
        alpha6_list.append( cell_no )
    if ( (subdomains.array()[cell_no] == 488) or
         (subdomains.array()[cell_no] == 489) or
         (subdomains.array()[cell_no] == 490) or
         (subdomains.array()[cell_no] == 491) or
         (subdomains.array()[cell_no] == 492) or
         (subdomains.array()[cell_no] == 493) or
         (subdomains.array()[cell_no] == 494) or
         # (subdomains.array()[cell_no] == 495) or
         (subdomains.array()[cell_no] == 496)
    ):
        ke.vector()[cell_no] = ke_values[1]
# dl.plot( ke, interactive=True )

ke2 = dl.Function(V0)
ke2_values = [0.01, 1.0/(100.0)*0.1]
ke2_local_range = V0.dofmap().ownership_range()
ke2_list = []
ke4_list = []
ke5_list = []
ke6_list = []
for cell_no in range(len(subdomains.array())):
    if subdomains.array()[cell_no] == 518:
        ke2.vector()[cell_no] = ke2_values[1]
    else:
        ke2.vector()[cell_no] = ke_values[0]
    if subdomains.array()[cell_no] == 471:
        ke2.vector()[cell_no] = ke2_values[1]
    if ( (subdomains.array()[cell_no] == 488) or
         (subdomains.array()[cell_no] == 489) or
         (subdomains.array()[cell_no] == 490) or
         (subdomains.array()[cell_no] == 491) or
         (subdomains.array()[cell_no] == 492) or
         (subdomains.array()[cell_no] == 493) or
         (subdomains.array()[cell_no] == 494) or
         (subdomains.array()[cell_no] == 496)
    ):
        ke2.vector()[cell_no] = ke2_values[1]
    if (subdomains.array()[cell_no] == 472):
        ke2_list.append( cell_no )
    if (subdomains.array()[cell_no] == 474):
        ke4_list.append( cell_no )
    if (subdomains.array()[cell_no] == 475):
        ke5_list.append( cell_no )
    if (subdomains.array()[cell_no] == 476):
        ke6_list.append( cell_no )    
# dl.plot( ke2, interactive=True )

##############
#### remove boundary points in temperature
te = dl.TrialFunction( T )
de = dl.TestFunction( T )
tbc_express = dl.Constant( 0 )*dl.inner( te,de )*dl.dx
tbc_mat = dl.assemble( tbc_express )
noslip = dl.Constant( 0 )
tbc1 = dl.DirichletBC( T, dl.Constant(0.0), boundaries, 498 )
tbc2 = dl.DirichletBC( T, dl.Constant(0.0), boundaries, 499 )
tbc3 = dl.DirichletBC( T, dl.Constant(0.0), boundaries, 500 )
tbc4 = dl.DirichletBC( T, dl.Constant(0.0), boundaries, 501 )
tbc5 = dl.DirichletBC( T, dl.Constant(0.0), boundaries, 502 )
tbc6 = dl.DirichletBC( T, dl.Constant(0.0), boundaries, 503 )
tbc7 = dl.DirichletBC( T, dl.Constant(0.0), boundaries, 504 )
tbc8 = dl.DirichletBC( T, dl.Constant(0.0), boundaries, 505 )
tbc9 = dl.DirichletBC( T, dl.Constant(0.0), boundaries, 506 )
tbc10 = dl.DirichletBC( T, dl.Constant(0.0), boundaries, 507 )
tbc11 = dl.DirichletBC( T, dl.Constant(0.0), boundaries, 508 )
tbcs = [ tbc1,tbc2,tbc3,
         tbc4,tbc5,tbc6,
         tbc7,tbc8,tbc9,
         tbc10,tbc11 ]    
for bco in tbcs:
    bco.apply( tbc_mat )
tbc = tbc_mat.array()
tbc_pt = []
tin_pt = []
for i in range( n_t ):
    if tbc[i][i] > 0.5:
        tbc_pt.append( i )
    else:
        tin_pt.append( i )

# number of variables, I0 + theta_list, only four doors unknowm,
n_t = T.dofmap().ownership_range()[1] - T.dofmap().ownership_range()[0] 
n_i = n_t - len( tbc_pt ) # remove boundary points from QP

#############
### linearize initial temperature distribution
#############
qpindx = qp_indx( 3 )

express_x0 = dolfin_MHE.e_x0( 1.0 )
t_x0 = dl.interpolate( express_x0, T )
express_x1 = dolfin_MHE.e_x1( 1.0 )
t_x1 = dl.interpolate( express_x1, T )
e_x = np.ones( ( 3, n_t ) )
e_x[0,:] = t_x0.vector().array()
e_x[1,:] = t_x1.vector().array()
##########
## matrix for QP
##########
# I0 (lambda_7) and dI0
de = dl.TestFunction( T )
te = dl.TrialFunction( T )
lii_express = dl.inner(te,de)*dl.dx
lii_mat = dl.assemble( lii_express )
lii = lii_mat.array()

### generate qpQ and kernal matrix, qpQ is psd (checked)
ker_djdk1 = kernal_djdk( mesh_drt, ke2_list, ke2_values )
ker_djdk2 = kernal_djdk( mesh_drt, ke4_list, ke2_values )
ker_djdk3 = kernal_djdk( mesh_drt, ke5_list, ke2_values )
ker_djdk4 = kernal_djdk( mesh_drt, ke6_list, ke2_values )
djdk_list = [ ker_djdk1, ker_djdk2, ker_djdk3, ker_djdk4 ]

ker_djda1 = kernal_djda( mesh_drt, alpha2_list, ke_values )
ker_djda2 = kernal_djda( mesh_drt, alpha4_list, ke_values )
ker_djda3 = kernal_djda( mesh_drt, alpha5_list, ke_values )
ker_djda4 = kernal_djda( mesh_drt, alpha6_list, ke_values )
djda_list = [ ker_djda1, ker_djda2, ker_djda3, ker_djda4 ]

ker_djdi1 = kernal_djdi( mesh_drt, bp0[0], bp1[0] )
ker_djdi2 = kernal_djdi( mesh_drt, bp0[1], bp1[1] )
ker_djdi3 = kernal_djdi( mesh_drt, bp0[2], bp1[2] )
pjpi_list = [ ker_djdi1, ker_djdi2, ker_djdi3 ]
# ### only one sensor, the dolfin_MHE should be changed as well
# pjpi_list = [ ker_djdi1 ]
#############
### test the kernals, pass
#############
# g1 = dolfin_MHE.bump_sensor( bp0[0], bp1[0] )
# g2 = dolfin_MHE.bump_sensor( bp0[1], bp1[1] )
# g3 = dolfin_MHE.bump_sensor( bp0[2], bp1[2] )
# te0 = dl.Function( T )
# te1 = dl.Function( T )
# t_v0 = np.random.rand( n_t )
# t_v1 = np.random.rand( n_t )
# te0.vector()[:] = t_v0
# te1.vector()[:] = t_v1
# l_express = ( g1*dl.inner( te1,te0 )*dl.dx
#               + g2*dl.inner( te1,te0 )*dl.dx
#               + g3*dl.inner( te1,te0 )*dl.dx )
# l_mat = dl.assemble( l_express )
# ker_t = ker_djdi1 + ker_djdi2 + ker_djdi3
# ker_ans = t_v0.dot( ker_t.dot( t_v1 ) )
# ipdb.set_trace()

qpQi = 0.5 * np.eye( qpindx.num_v )
# qpQi = 0.5 * lii[ np.ix_( tin_pt,tin_pt ) ]
qpQt = 0.5 * np.eye( len( djda_list ) )
qpQa = 0.5 * np.eye( qpindx.n_t )
# ipdb.set_trace()

########
## k: theta and I0, start at original
## Dolfin solution 
## assembling parameters for QP in each step
## QP, call qp_MHE( Q, L, indx, x_lb, x_ub )
## Armijo after QP
## k+1: theta and I0
## judgement 
########

### step 0
init_x = 0.0*np.ones( qpindx.num_v )
# init_x[ 0:qpindx.theta(1) ] = t_s[ 0,tin_pt ]
init_x[ -4 ] = 0.2
init_x[ -2 ] = 0.2
init_x[ -1 ] = 0.2
[ i0_value, alpha_value, ke_value ] = wrap_var( init_x, n_t )

adj_pt = 1
[ t_v, u_v, p_v,
  t_ad, u_ad, p_ad ] = dolfin_MHE.dolfin_mhe( mesh_drt, i0_value,
                                              alpha_value, ke_value, t_s,
                                              bp0, bp1, adj_pt )

qpLt = assemble_qpLt( djdk_list, djda_list, t_v,
                      u_v, t_ad, u_ad, dt, Tf )

xt_lb = np.ones( qpindx.num_v - qpindx.n_t )
xt_ub = np.ones( qpindx.num_v - qpindx.n_t )
for i in range( qpindx.num_v - qpindx.n_t ):
    xt_lb[ i ] = -1.0 * init_x[ qpindx.theta( i+1 ) ]
    xt_ub[ i ] = 1.0 - init_x[ qpindx.theta( i+1 ) ]

[ del_t, qp_val, qp_rtv ] = qp_MHE.qp_t( qpQt,
                                         qpLt, qpindx, xt_lb, xt_ub )
del_x = np.zeros( qpindx.num_v )
del_x[ qpindx.theta(1): ] = del_t

fxr = fx( mesh_drt, qpindx, yeta0, yeta1,
          ke_values, ke2_values,
          pjpi_list, lii, bp0, bp1 )

fx0 = fxr.value( init_x, t_s )
# iteration list
itr_list = []
lp_num = 1
# ipdb.set_trace()
# armijo
[ k, fx_tmp ] = grad_armijo( init_x, t_s, fxr, del_x,
                             qp_val, alpha, beta )
tmp_x0 = init_x + ( beta**k ) * del_x

itr_list.append( { "Itr No": lp_num,
                   "Fx": fx_tmp,
                   "QP value": qp_val,
                   "dx": 1.0,
                   "Return value": qp_rtv,
                   "k value": k,
                   "i0": tmp_x0[ 0:qpindx.theta(1) ],
                   "theta 1": tmp_x0[ qpindx.theta(1) ],
                   "theta 2": tmp_x0[ qpindx.theta(2) ],
                   "theta 3": tmp_x0[ qpindx.theta(3) ],
                   "theta 4": tmp_x0[ qpindx.theta(4) ] } )
lp_num += 1

[ i0_value, alpha_value, ke_value ] = wrap_var( tmp_x0, n_t )
adj_pt = 1
[ t_v, u_v, p_v,
  t_ad, u_ad, p_ad ] = dolfin_MHE.dolfin_mhe( mesh_drt, i0_value,
                                              alpha_value, ke_value, t_s,
                                              bp0, bp1, adj_pt )
#############
### plot adjoint
#############
# te0 = dl.Function( T )
# for i in range( Tf/dt+1 ):
#     te0.vector()[:] = t_ad[i,:]
#     dl.plot( te0, interactive=True )
# ipdb.set_trace()
qpLa =  assemble_qpLa( pjpi_list, lii, t_v, u_v,
                       t_ad, u_ad, t_s, e_x, tin_pt,
                       yeta0, yeta1, qpindx )

xi_lb = -0.5 * np.ones( qpindx.n_t )
xi_ub = 0.5 * np.ones( qpindx.n_t )

[ del_a, qp_val, qp_rtv ] = qp_MHE.qp_i( qpQa, qpLa,
                                         qpindx, xi_lb, xi_ub )
del_x = np.zeros( qpindx.num_v )
del_x[ :qpindx.theta(1) ] = del_a

# ipdb.set_trace()
# epsl = 1e-6
# lambda_fun = np.zeros( ( 100, ) )
# for i in range( 100 ):
#     fx_lambda = fxr.value( tmp_x0 + epsl*i*del_x, t_s )
#     lambda_fun[i] = fx_lambda - fx_tmp - epsl*i*qpLa.dot(
#         del_x[ :qpindx.theta(1) ] )
# ipdb.set_trace()
[ k, fx_tmp ] = grad_armijo( tmp_x0, t_s, fxr, del_x,
                             qp_val, alpha_i, beta_i )
tmp_x0 = tmp_x0 + ( beta_i**k ) * del_x

# ipdb.set_trace()
itr_list.append( { "Itr No": lp_num,
                   "Fx": fx_tmp,
                   "QP value": qp_val,
                   "dx": 1.0,
                   "Return value": qp_rtv,
                   "k value": k,
                   "i0": tmp_x0[ 0:qpindx.theta(1) ],
                   "theta 1": tmp_x0[ qpindx.theta(1) ],
                   "theta 2": tmp_x0[ qpindx.theta(2) ],
                   "theta 3": tmp_x0[ qpindx.theta(3) ],
                   "theta 4": tmp_x0[ qpindx.theta(4) ] } )
lp_num += 1
# ipdb.set_trace()
# while ( lp_num < lp_step ):
# while ( ( abs( fx_tmp - fx0 ) > epsl ) and ( lp_num < lp_step ) ):
while ( ( del_x.dot( del_x ) > epsl ) and ( lp_num < lp_step ) ):    
    fx0 = fx_tmp
    # theta
    [ i0_value, alpha_value, ke_value ] = wrap_var( tmp_x0, n_t )
    [ t_v, u_v, p_v,
      t_ad, u_ad, p_ad ] = dolfin_MHE.dolfin_mhe( mesh_drt, i0_value,
                                                  alpha_value, ke_value,
                                                  t_s, bp0, bp1, 1 )    
    qpLt = assemble_qpLt( djdk_list, djda_list, t_v,
                          u_v, t_ad, u_ad, dt, Tf )
    for i in range( qpindx.num_v - qpindx.n_t ):
        xt_lb[ i ] = -1.0 * tmp_x0[ qpindx.theta( i+1 ) ]
        xt_ub[ i ] = 1.0 - tmp_x0[ qpindx.theta( i+1 ) ]

    [ del_t, qp_val, qp_rtv ] = qp_MHE.qp_t( qpQt,
                                             qpLt, qpindx, xt_lb, xt_ub )
    del_x = np.zeros( qpindx.num_v )
    del_x[ qpindx.theta(1): ] = del_t

    [ k, fx_tmp ] = grad_armijo( tmp_x0, t_s, fxr, del_x,
                                 qp_val, alpha, beta )

    tmp_x0 = tmp_x0 + ( beta**k ) * del_x
    itr_list.append( { "Itr No": lp_num,
                       "Fx": fx_tmp,
                       "QP value": qp_val,
                       "dx": del_x.dot( del_x ),
                       "Return value": qp_rtv,
                       "k value": k,
                       "i0": tmp_x0[ 0:qpindx.theta(1) ],
                       "theta 1": tmp_x0[ qpindx.theta(1) ],
                       "theta 2": tmp_x0[ qpindx.theta(2) ],
                       "theta 3": tmp_x0[ qpindx.theta(3) ],
                       "theta 4": tmp_x0[ qpindx.theta(4) ] } )
    lp_num += 1
    if ( qp_val > -1E-5 ):
        print "*" * 32
        print "Jumping"
        print "*" * 32
        break
    
    # i0
    [ i0_value, alpha_value, ke_value ] = wrap_var( tmp_x0, n_t )
    [ t_v, u_v, p_v,
      t_ad, u_ad, p_ad ] = dolfin_MHE.dolfin_mhe( mesh_drt, i0_value,
                                                  alpha_value, ke_value,
                                                  t_s, bp0, bp1, adj_pt )
    
    qpLa =  assemble_qpLa( pjpi_list, lii, t_v, u_v,
                           t_ad, u_ad, t_s, e_x, tin_pt,
                           yeta0, yeta1, qpindx )
    xi_lb = -0.5 * np.ones( qpindx.n_t )
    xi_ub = 0.5 * np.ones( qpindx.n_t )

    [ del_a, qp_val, qp_rtv ] = qp_MHE.qp_i( qpQa, qpLa,
                                             qpindx, xi_lb, xi_ub )
    del_x = np.zeros( qpindx.num_v )
    del_x[ :qpindx.theta(1) ] = del_a

    # ipdb.set_trace()
    [ k, fx_tmp ] = grad_armijo( tmp_x0, t_s, fxr, del_x,
                                 qp_val, alpha_i, beta_i )
    tmp_x0 = tmp_x0 + ( beta_i**k ) * del_x
    # itr_list.append( { "Itr No": lp_num,
    #                    "Fx": fx_tmp,
    #                    "QP value": qp_val,
    #                    "dx": 1.0,
    #                    "Return value": qp_rtv,
    #                    "k value": k,
    #                    "i0": tmp_x0[ 0:qpindx.theta(1) ],
    #                    "theta 1": tmp_x0[ qpindx.theta(1) ],
    #                    "theta 2": tmp_x0[ qpindx.theta(2) ],
    #                    "theta 3": tmp_x0[ qpindx.theta(3) ],
    #                    "theta 4": tmp_x0[ qpindx.theta(4) ] } )
    # lp_num += 1
    if ( qp_val > -1E-5 ):
        print "*" * 32
        print "Jumping"
        print "*" * 32
        break
    
    # theta
    [ i0_value, alpha_value, ke_value ] = wrap_var( tmp_x0, n_t )
    [ t_v, u_v, p_v,
      t_ad, u_ad, p_ad ] = dolfin_MHE.dolfin_mhe( mesh_drt, i0_value,
                                                  alpha_value, ke_value,
                                                  t_s, bp0, bp1, 1 )    
    qpLt = assemble_qpLt( djdk_list, djda_list, t_v,
                          u_v, t_ad, u_ad, dt, Tf )
    for i in range( qpindx.num_v - qpindx.n_t ):
        xt_lb[ i ] = -1.0 * tmp_x0[ qpindx.theta( i+1 ) ]
        xt_ub[ i ] = 1.0 - tmp_x0[ qpindx.theta( i+1 ) ]

    [ del_t, qp_val, qp_rtv ] = qp_MHE.qp_t( qpQt,
                                             qpLt, qpindx, xt_lb, xt_ub )
    del_x = np.zeros( qpindx.num_v )
    del_x[ qpindx.theta(1): ] = del_t

    [ k, fx_tmp ] = grad_armijo( tmp_x0, t_s, fxr, del_x,
                                 qp_val, alpha, beta )

    tmp_x0 = tmp_x0 + ( beta**k ) * del_x
    itr_list.append( { "Itr No": lp_num,
                       "Fx": fx_tmp,
                       "QP value": qp_val,
                       "dx": del_x.dot( del_x ),
                       "Return value": qp_rtv,
                       "k value": k,
                       "i0": tmp_x0[ 0:qpindx.theta(1) ],
                       "theta 1": tmp_x0[ qpindx.theta(1) ],
                       "theta 2": tmp_x0[ qpindx.theta(2) ],
                       "theta 3": tmp_x0[ qpindx.theta(3) ],
                       "theta 4": tmp_x0[ qpindx.theta(4) ] } )
    lp_num += 1
    if ( qp_val > -1E-5 ):
        print "*" * 32
        print "Jumping"
        print "*" * 32
        break

##################
##### pre the data
##################
# ipdb.set_trace()
lp_num = 0
lp_step = 20
while ( lp_num < lp_step ):
    
    [ i0_value, alpha_value, ke_value ] = wrap_var( tmp_x0, n_t )
    [ t_v, u_v, p_v,
      t_ad, u_ad, p_ad ] = dolfin_MHE.dolfin_mhe( mesh_drt, i0_value,
                                                  alpha_value, ke_value,
                                                  t_s, bp0, bp1, 0 )
    # dolfin_tem( filename, alpha_value, ke_value, u_s )
    [ A_o, b_o ] = dolfin_MHE.dolfin_tem( mesh_drt, alpha_value,
                                          ke_value, u_v )
    
    ax0 = np.zeros( ( t_s.shape[0], A_o.shape[0] ) )
    ax1 = np.zeros( ax0.shape )
    axc = np.ones( ax0.shape )
    ax0[ 0,: ] = e_x[0,:]
    ax1[ 0,: ] = e_x[1,:]
    for i in range( 1, t_s.shape[0] ):
        ax0[ i,: ] = A_o.dot( ax0[ i-1,: ] )
        ax1[ i,: ] = A_o.dot( ax1[ i-1,: ] )
        axc[ i,: ] = A_o.dot( axc[ i-1,: ] )
    ax0 = ax0[:,tin_pt]
    ax1 = ax1[:,tin_pt]
    axc = axc[:,tin_pt]
    # cost function
    pjpi_tmp = ( pjpi_list[0][ np.ix_( tin_pt, tin_pt ) ]
                + pjpi_list[1][ np.ix_( tin_pt, tin_pt ) ]
                + pjpi_list[2][ np.ix_( tin_pt, tin_pt ) ] )

    prob = ow.nlp.Problem( N=3, Ncons=0 )
    prob.initPoint( tmp_x0[0:qpindx.theta(1)] )
    x_bnd_l = -0.05*np.ones( 3 )
    x_bnd_h =  0.05*np.ones( 3 )
    prob.consBox( x_bnd_l, x_bnd_h )
    prob.objFctn( objf )
    prob.objGrad( objg )
    # ipdb.set_trace()
    # test_pt = np.random.randn( 3 )
    # if( not prob.checkGrad( debug=True ) ):
    #     sys.exit( "Gradient check failed" )
    # ipdb.set_trace()
    solver = ow.ipopt.Solver( prob )
    solver.debug = True
    solver.solve()
    print( prob.soln.getStatus() )
    print( "Value: " + str( prob.soln.value ) )
    print( "Retval: " + str( prob.soln.retval ) )
    # ipdb.set_trace()
    tmp_x0[ 0:qpindx.theta(1) ] = prob.soln.final
    fx_tmp = fxr.value( tmp_x0, t_s )

    # theta
    [ i0_value, alpha_value, ke_value ] = wrap_var( tmp_x0, n_t )
    [ t_v, u_v, p_v,
      t_ad, u_ad, p_ad ] = dolfin_MHE.dolfin_mhe( mesh_drt, i0_value,
                                                  alpha_value, ke_value,
                                                  t_s, bp0, bp1, 1 )    
    qpLt = assemble_qpLt( djdk_list, djda_list, t_v,
                          u_v, t_ad, u_ad, dt, Tf )
    for i in range( qpindx.num_v - qpindx.n_t ):
        xt_lb[ i ] = -1.0 * tmp_x0[ qpindx.theta( i+1 ) ]
        xt_ub[ i ] = 1.0 - tmp_x0[ qpindx.theta( i+1 ) ]

    [ del_t, qp_val, qp_rtv ] = qp_MHE.qp_t( qpQt,
                                             qpLt, qpindx, xt_lb, xt_ub )
    del_x = np.zeros( qpindx.num_v )
    del_x[ qpindx.theta(1): ] = del_t

    [ k, fx_tmp ] = grad_armijo( tmp_x0, t_s, fxr, del_x,
                                 qp_val, alpha, beta )

    tmp_x0 = tmp_x0 + ( beta**k ) * del_x
    itr_list.append( { "Itr No": lp_num,
                       "Fx": fx_tmp,
                       "QP value": qp_val,
                       "dx": del_x.dot( del_x ),
                       "Return value": qp_rtv,
                       "k value": k,
                       "i0": tmp_x0[ 0:qpindx.theta(1) ],
                       "theta 1": tmp_x0[ qpindx.theta(1) ],
                       "theta 2": tmp_x0[ qpindx.theta(2) ],
                       "theta 3": tmp_x0[ qpindx.theta(3) ],
                       "theta 4": tmp_x0[ qpindx.theta(4) ] } )
    lp_num += 1
    if ( qp_val > -1E-5 ):
        print "*" * 32
        print "Jumping"
        print "*" * 32
        break
    if ( del_x.dot( del_x ) < 1E-6 ):
        break
    
################
##### final QP
ipdb.set_trace()

lp_num = 0 # reset lp_num
lp_step = 80 # lp_step = 100 # reset lp_step
beta = 0.65 # beta = 0.7
armijo_num = 35 # armijo_num = 40

qpQ = np.eye( qpindx.num_v )
# qpQ[ 0:qpindx.theta(1), 0:qpindx.theta(1) ] = lii[ np.ix_( tin_pt, tin_pt ) ]
qpQ = 0.5*qpQ

fx0 = fx_tmp

[ i0_value, alpha_value, ke_value ] = wrap_var( tmp_x0, n_t )
[ t_v, u_v, p_v,
t_ad, u_ad, p_ad ] = dolfin_MHE.dolfin_mhe( mesh_drt, i0_value,
                                            alpha_value, ke_value,
                                            t_s, bp0, bp1, 1 )
 
qpL = assemble_qpL( djdk_list, djda_list, pjpi_list,
                    lii, t_v, u_v, t_ad,
                    u_ad, t_s, e_x, tin_pt,
                    dt, Tf, yeta0, yeta1, qpindx )

# def assemble_qpL( djdk_list, djda_list, pjpi_list,
#                   kernal_lii, t_v, u_v, t_adj, u_adj,
#                   t_s, e_x, tin_range, 
#                   dt, Tf, yeta0, yeta1, indx ):
    
x_lb = -0.5 * np.ones( qpindx.num_v )
x_ub = 0.5 * np.ones( qpindx.num_v ) 

for i in range( qpindx.num_v - qpindx.n_t ):
    x_lb[ qpindx.theta(i+1) ] = -1.0 * tmp_x0[ qpindx.theta( i+1 ) ]
    x_ub[ qpindx.theta(i+1) ] = 1.0 - tmp_x0[ qpindx.theta( i+1 ) ]

[ del_x, qp_val, qp_rtv ] = qp_MHE.qp_mhe( qpQ,
                                           qpL, qpindx, x_lb, x_ub )
# ipdb.set_trace()
[ k, fx_tmp ] = grad_armijo( tmp_x0, t_s, fxr, del_x,
                             qp_val, alpha_i, beta )
tmp_x0 = tmp_x0 + ( beta**k ) * del_x
# del_x_norm = del_x[ 0:qpindx.theta(1) ].dot(
#     lii[ np.ix_( tin_pt, tin_pt ) ].dot(
#         del_x[ 0:qpindx.theta(1) ] ) ) + del_x[ qpindx.theta(1): ].dot(
#             del_x[ qpindx.theta(1): ] )
del_x_norm = del_x.dot( del_x )
itr_list.append( { "Itr No": lp_num,
                   "Fx": fx_tmp,
                   "QP value": qp_val,
                   "dx": del_x_norm,
                   "Return value": qp_rtv,
                   "k value": k,
                   "i0": tmp_x0[ 0:qpindx.theta(1) ],
                   "theta 1": tmp_x0[ qpindx.theta(1) ],
                   "theta 2": tmp_x0[ qpindx.theta(2) ],
                   "theta 3": tmp_x0[ qpindx.theta(3) ],
                   "theta 4": tmp_x0[ qpindx.theta(4) ] } )
lp_num += 1
# ipdb.set_trace()
while ( ( del_x_norm > epsl ) and ( lp_num < lp_step ) ):
    fx0 = fx_tmp

    [ i0_value, alpha_value, ke_value ] = wrap_var( tmp_x0, n_t )
    [ t_v, u_v, p_v,
      t_ad, u_ad, p_ad ] = dolfin_MHE.dolfin_mhe( mesh_drt, i0_value,
                                                  alpha_value, ke_value,
                                                  t_s, bp0, bp1, 1 )
 
    qpL = assemble_qpL( djdk_list, djda_list, pjpi_list,
                        lii, t_v, u_v, t_ad,
                        u_ad, t_s, e_x, tin_pt,
                        dt, Tf, yeta0, yeta1, qpindx )
    
    x_lb = -0.05 * np.ones( qpindx.num_v )
    x_ub = 0.05 * np.ones( qpindx.num_v ) 

    for i in range( qpindx.num_v - qpindx.n_t ):
        x_lb[ qpindx.theta(i+1) ] = -1.0 * tmp_x0[ qpindx.theta( i+1 ) ]
        x_ub[ qpindx.theta(i+1) ] = 1.0 - tmp_x0[ qpindx.theta( i+1 ) ]

    [ del_x, qp_val, qp_rtv ] = qp_MHE.qp_mhe( qpQ,
                                               qpL, qpindx, x_lb, x_ub )
        
    [ k, fx_tmp ] = grad_armijo( tmp_x0, t_s, fxr, del_x,
                                 qp_val, alpha_i, beta, armijo_num )
    tmp_x0 = tmp_x0 + ( beta**k ) * del_x
    # del_x_norm = del_x[ 0:qpindx.theta(1) ].dot(
    #     lii[ np.ix_( tin_pt, tin_pt ) ].dot(
    #         del_x[ 0:qpindx.theta(1) ] ) ) + del_x[ qpindx.theta(1): ].dot(
    #             del_x[ qpindx.theta(1): ] )
    del_x_norm = del_x.dot( del_x )
    itr_list.append( { "Itr No": lp_num,
                       "Fx": fx_tmp,
                       "QP value": qp_val,
                       "dx": del_x_norm,
                       "Return value": qp_rtv,
                       "k value": k,
                       "i0": tmp_x0[ 0:qpindx.theta(1) ],
                       "theta 1": tmp_x0[ qpindx.theta(1) ],
                       "theta 2": tmp_x0[ qpindx.theta(2) ],
                       "theta 3": tmp_x0[ qpindx.theta(3) ],
                       "theta 4": tmp_x0[ qpindx.theta(4) ] } )
    lp_num += 1
  
    if ( qp_val > -1*epsl ):
        break

# if lp_num >= lp_step:
#     print "The algorithm has not been convergent after maximun number of steps."
# else:
#     print "The algorithm reaches the convergent condition."
    
###############
### plot estimation results
ipdb.set_trace()
ke = dl.Function( T )
u = dl.Function( V )
p = dl.Function( P )

[ i0_value, alpha_value, ke_value ] = wrap_var( tmp_x0, n_t )
[ t_v, u_v, p_v,
  t_ad, u_ad, p_ad ] = dolfin_MHE.dolfin_mhe( mesh_drt, i0_value, alpha_value,
                                              ke_value, t_s, bp0, bp1, 0 )
i = 0
t = dt
while t <= Tf:
    t += dt
    ke.vector().set_local( t_s[i,:] )
    wr_t = dl.plot( ke, title="real temperature", interactive=True )
    wr_t.write_pdf( ( drt + '/real_t' + str(i) ) )
    i += 1

u.vector().set_local( u_s )
wr_u = dl.plot( u, title="real velocity", interactive=True )
wr_u.write_pdf( ( drt + '/real_v' ) )

p.vector().set_local( p_s )
wr_p = dl.plot( p, title="real pressure", interactive=True )
wr_p.write_pdf( ( drt + '/real_p') )

i = 0
t = dt
while t <= Tf:
    t += dt
    ke.vector().set_local( t_v[i,:] )
    wr_t = dl.plot( ke, title="estimation temperature", interactive=True )
    wr_t.write_pdf( ( drt + '/estimate_t' + str(i) ) )
    i += 1

u.vector().set_local( u_v )
wr_u = dl.plot( u, title="estimation velocity", interactive=True )
wr_u.write_pdf( ( drt + '/estimate_v') )

p.vector().set_local( p_v )
wr_p = dl.plot( p, title="estimate pressure", interactive=True )
wr_p.write_pdf( ( drt + '/estimate_p') )

np.save( ( drt + "/itr_list_new" ), itr_list )

##################################
### temp trush
##################################

'''
############
### 3d tensor generator/loader, no need for a 3d tensor
############
def vtt_assemble( mesh_drt, i=0, n_u=1 ):
    import dolfin as dl
    import numpy as np
    import scipy.sparse as sp
    mesh = dl.Mesh( mesh_drt )
    T = dl.FunctionSpace( mesh, "CG", 2 )
    V0 = dl.FunctionSpace( mesh, "DG", 0 )
    u = dl.Function( V0 )
    de = dl.TestFunction( T )
    te = dl.TrialFunction( T )
    coeff = np.zeros( n_u )
    coeff[i] = 1.0
    u.vector().set_local( coeff )
    ltt_express = u*dl.div( dl.grad(te) )*de*dl.dx
    ltt_mat = dl.assemble( ltt_express )
    ltt_sparse = sp.csr_matrix( ltt_mat.array() )
    return ltt_sparse

def vuu_assemble( mesh_drt, i=0, n_u=1 ):
    import dolfin as dl
    import numpy as np
    import scipy.sparse as sp
    mesh = dl.Mesh( mesh_drt )
    V = dl.VectorFunctionSpace( mesh,"CG",2 )
    V0 = dl.FunctionSpace( mesh, "DG", 0 )
    u = dl.Function( V0 )
    de = dl.TestFunction( V )
    te = dl.TrialFunction( V )
    coeff = np.zeros( n_u )
    coeff[i] = 1.0
    u.vector().set_local( coeff )
    ltt_express = u*dl.inner( de,te )*dl.dx
    ltt_mat = dl.assemble( ltt_express )
    ltt_sparse = sp.csr_matrix( ltt_mat.array() )
    return ltt_sparse

ipdb.set_trace()
from IPython import parallel
####################
### slow parallel 
####################
n_v0 = V0.dofmap().ownership_range()[1] - V0.dofmap().ownership_range()[0] 
filename_vtt = mesh_drt + "_cross/vtt"
# list vtt, 0: len(alpha2_list): len(alpha4_list)+len(alpha2_list): len(alpha4_list)+len(alpha2_list)+len(alpha5_list): len(alpha_list) 

try:
    with open( filename_vtt, "rb" ) as data_vtt:
        vtt = pickle.load( data_vtt )
except IOError:
    rc = parallel.Client()
    lview = rc.load_balanced_view()
    ltt_para = []
    for i in range( len( alpha_list ) ):
        ar = lview.apply( vtt_assemble, mesh_drt, alpha_list[i], n_v0 )
        ltt_para.append( ar )
        rc.wait( ltt_para )
    vtt = []
    for ele in ltt_para:
        ltt.append( ele.get() )
    with open( filename_vtt, "w" ) as lut_output:
        pickle.dump( vtt, lut_output )
    print "vtt has been saved into: " + filename_vtt

filename_vuu = mesh_drt + "_cross/vuu"
# list vuu, 0: len(alpha2_list): len(alpha4_list)+len(alpha2_list): len(alpha4_list)+len(alpha2_list)+len(alpha5_list): len(alpha_list) 
try:
    with open( filename_vuu, "rb" ) as data_vuu:
        vuu = pickle.load( data_vuu )
except IOError:
    rc = parallel.Client()
    lview = rc.load_balanced_view()
    ltt_para = []
    for i in range( len( ke_list ) ):
        ar = lview.apply( vuu_assemble, mesh_drt, ke_list[i], n_v0 )
        ltt_para.append( ar )
        rc.wait( ltt_para )
    vuu = []
    for ele in ltt_para:
        vuu.append( ele.get() )
    with open( filename_vuu, "w" ) as lut_output:
        pickle.dump( vuu, lut_output )
    print "vuu has been saved into: " + filename_vuu


'''
