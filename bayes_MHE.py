import numpy as np
import scipy.sparse as sp
import math
import h5py
import pickle
import math
import itertools
from optwrapper import nlp, npsol, snopt
from datetime import date
import os
import ipdb

import qp_MHE
reload( qp_MHE )

# retrieve data
def retrieve_data( filename ):
    '''
    itr_list.append( { "Tf": Tf,
    "dt": dt,
    "n_t": n_t,
    "vx0": vx0,
    "vx1": vx1,
    "vc": np.ones( ( n_t, ) ),
    "pjpi": pjpi,
    "tin_pt": tin_pt,
    "b_0": b_o,
    "A_o": A_o,
    "theta_list": np.array( theta_list ),
    "t_s": t_original } )
    '''
    # ipdb.set_trace()
    tmp_list = np.load( filename )
    tmp_list = tmp_list[-1]
    Tf = tmp_list[ "Tf" ] # time num
    dt = tmp_list[ "dt" ]
    n_t = tmp_list[ "n_t" ] # temperature num
    vx = tmp_list[ "vx0" ] # position vector
    vy = tmp_list[ "vx1" ]
    vc = tmp_list[ "vc" ]
    pjpi = tmp_list[ "pjpi" ] # sensor matrix, multi in one, symmetric
    tin_pt = tmp_list[ "tin_pt" ] # tin_pt array
    b_o = tmp_list[ "b_o" ]
    A_o = tmp_list[ "A_o" ]
    theta_list = tmp_list[ "theta_list" ]
    t_s = tmp_list[ "t_s" ]
    return ( Tf, dt, n_t,
             vx, vy, vc,
             pjpi, tin_pt, b_o,
             A_o, theta_list, t_s )

( Tf, dt, n_t,
  vx, vy, vc,
  pjpi, tin_pt, b_o,
A_o, theta_list, t_s ) = retrieve_data(
    "./results/bayes/nonhomo/bayes0000.npy" )
result_file = './results/bayes/nonhomo/0000_n/result0000_1.npy'
# ipdb.set_trace()

### cost function's (OP) parameter
yeta_0 = 1.0
yeta_1 = 1.0
n_theta = theta_list.shape[0]
#############
#### generate the NLP
#############
t_idx = np.arange( 0, 3 ) # temperature indx, ax+by+c, only three variables
theta_idx = t_idx.size + np.arange( 0, n_theta )  # velocity indx

A_sparse = []
for i in range( n_theta ):
    A_sparse.append( np.eye( n_t ) )
    for j in range( 1, Tf/dt+1 ):
        A_sparse.append( A_sparse[-1].dot( A_o[i,:,:] ) )
        # print "i: " + str( i ) + ", j: " + str( j )

a_indx  = np.arange( 0, n_theta*( Tf/dt+1 ) ).reshape( (n_theta,Tf/dt+1) )
n_total = 3+n_theta
# x_init = np.random.randn( n_total )
x_init = 0.4*np.ones( ( n_total, ) )
# x_init[-3] = 0.0
# ipdb.set_trace()

################
### testing the data recieved
################
# cost function
def objf(out, x):
    tem0 = x[ t_idx[0] ]*vx + x[ t_idx[1] ]*vy + x[ t_idx[2] ]*vc
    # initial time
    out[0] = ( 1.0/n_t ) * yeta_0 * ( tem0 - t_s[0,:] ).dot(
        pjpi.dot( tem0 - t_s[0,:] ) )
    # later time
    for i in range( 1,Tf/dt+1 ):
        tmp_m = np.zeros( ( n_t, ) )
        for j in range( n_theta ):
            tmp_m += x[ theta_idx[j] ] * A_sparse[ a_indx[j,i] ].dot(
                tem0 ) + x[ theta_idx[j] ] * b_o[j,i-1,:]
        out[0] += ( 1.0/n_t ) * ( tmp_m-t_s[i,:] ).dot(
            pjpi.dot( tmp_m-t_s[i,:] ) )    
        
def objg(out, x):
    # ipdb.set_trace()
    tem0 = x[ t_idx[0] ]*vx + x[ t_idx[1] ]*vy + x[ t_idx[2] ]*vc
    ## initial time, ax+by+c
    out[ t_idx[0] ] = ( 2.0/n_t ) * yeta_0 * ( vx ).dot( pjpi.dot( tem0 - t_s[0] ) )
    out[ t_idx[1] ] = ( 2.0/n_t ) * yeta_0 * ( vy ).dot( pjpi.dot( tem0 - t_s[0] ) )
    out[ t_idx[2] ] = ( 2.0/n_t ) * yeta_0 * ( vc ).dot( pjpi.dot( tem0 - t_s[0] ) )
    for i in theta_idx:
        out[ i ] = 0.0 # theta
    ## later time
    for i in range( 1,Tf/dt+1 ):
        tmp_m = np.zeros( ( n_t, ) )
        tmp_x = np.zeros( ( n_t, ) )
        tmp_y = np.zeros( ( n_t, ) )
        tmp_c = np.zeros( ( n_t, ) )
        for j in range( n_theta ):
            tmp_m += x[theta_idx[j]] * A_sparse[ a_indx[j,i] ].dot(
                tem0 ) + x[theta_idx[j]] * b_o[j,i-1,:]
            tmp_x += x[theta_idx[j]] * A_sparse[ a_indx[j,i] ].dot(
                vx )
            tmp_y += x[theta_idx[j]] * A_sparse[ a_indx[j,i] ].dot(
                vy )
            tmp_c += x[theta_idx[j]] * A_sparse[ a_indx[j,i] ].dot(
                vc )
        out[ t_idx[0] ] += ( 2.0/n_t ) * ( tmp_x ).dot(
            pjpi.dot( tmp_m-t_s[i,:] ) )
        out[ t_idx[1] ] += ( 2.0/n_t ) * ( tmp_y ).dot(
            pjpi.dot( tmp_m-t_s[i,:] ) )
        out[ t_idx[2] ] += ( 2.0/n_t ) * ( tmp_c ).dot(
            pjpi.dot( tmp_m-t_s[i,:] ) )
        # theta
        for j in range( n_theta ):
            tmp_theta = A_sparse[ a_indx[j,i] ].dot(
                tem0 ) + b_o[j,i-1,:]
            out[ theta_idx[ j ] ] += ( 2.0/n_t ) * tmp_theta.dot(
                pjpi.dot( tmp_m - t_s[i,:] ) )

def consf(out, x):
    out[0] = 0.0
    for i in theta_idx:
        out[0] += x[i]

def consg(out, x):
    out[ 0, t_idx[0] ] = 0.0
    out[ 0, t_idx[1] ] = 0.0
    out[ 0, t_idx[2] ] = 0.0
    for i in theta_idx:
        out[ 0, i ] = 1.0
    
prob = nlp.Problem( N=n_total, Ncons=1 )
# x_init = np.random.randn( n_total )
prob.initPoint( x_init )

# set box constraint
x_bnd_l = np.zeros( 3+n_theta )
x_bnd_h = np.ones( 3+n_theta )
prob.consBox( x_bnd_l, x_bnd_h )
# constraints' range
consf_bounds_low = np.ones( (1, ) )
consf_bounds_upper = np.ones( (1, ) )
# cost and constraints
prob.objFctn( objf )
prob.objGrad( objg )
prob.consFctn( consf, lb=consf_bounds_low, ub=consf_bounds_upper )
prob.consGrad( consg )
# ipdb.set_trace()
# test_pt = np.random.randn( n_total )
# if( not prob.checkGrad( debug=True ) ):
#     sys.exit( "Gradient check failed" )
# ipdb.set_trace()

solver = snopt.Solver( prob )
solver.debug = True
solver.options[ "printLevel" ] = 10
drt = './results/bayes/'+'1'
if not os.path.exists(drt):
    os.makedirs(drt)
solver.options["printFile"] = (drt + "/optwrp1.txt")
solver.options["minorPrintLevel"] = 10
solver.options["printLevel"] = 10
solver.options["qpSolver"] = "cg"

solver.solve()
print( prob.soln.getStatus() )
print( "Value: " + str( prob.soln.value ) )
print( "Retval: " + str( prob.soln.retval ) )
# ipdb.set_trace()

#########
### saving results
#########
x_final = prob.soln.final
dr1 = 0
dr2 = 0
dr3 = 0
dr4 = 0
for i in range( n_theta ):
    dr1 += theta_list[i][0]*x_final[ theta_idx[i] ]
    dr2 += theta_list[i][1]*x_final[ theta_idx[i] ]
    dr3 += theta_list[i][2]*x_final[ theta_idx[i] ]
    dr4 += theta_list[i][3]*x_final[ theta_idx[i] ]
  
# tem0 = x_final[ t_idx[0] ]*vx + x_final[ t_idx[1] ]*vy + x_final[ t_idx[2] ]*vc
# t_t = np.zeros( t_s.shape )
# t_t[0] = tem0
# for i in range( 1,Tf/dt+1 ):
#     for j in range( n_theta ):
#             t_t[i] += x_final[ theta_idx[j] ]*A_sparse[
#                 a_indx[j,i] ].dot( tem0 ) + x_final[
#                     theta_idx[j] ] * b_o[j,i-1,:]
                 
print "Starting writting the data ... ..."
result_list = []
result_list.append( { 'dr1': dr1,
                      'dr2': dr2,
                      'dr3': dr3,
                      'dr4': dr4,
                      'ea': x_final[ t_idx[0] ],
                      'eb': x_final[ t_idx[1] ],
                      'ec': x_final[ t_idx[2] ],
                      't_original': t_s } )
np.save( result_file, result_list )
# fdata = h5py.File( result_file, "w" )
# fdata[ "dr1" ] = dr1 # for doors
# fdata[ "dr2" ] = dr2
# fdata[ "dr3" ] = dr3 
# fdata[ "dr4" ] = dr4
# fdata[ "ea" ] = x_final[ t_idx[0] ] # temperature estimation
# fdata[ "eb" ] = x_final[ t_idx[1] ]
# fdata[ "ec" ] = x_final[ t_idx[2] ]
# fdata[ "t_original" ] = t_s 
# fdata[ "t_estimation" ] = t_t
print( "Data is written to: " + result_file + '...' )
# np.save(result_ts, t_t)
