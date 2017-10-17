from numpy import *
import matplotlib.pyplot as plt
import os
import pdb
import pickle

import PyXFocus.sources as source
import PyXFocus.surfaces as surf
import PyXFocus.analyses as anal
import PyXFocus.transformations as tran
import PyXFocus.grating as grat
import PyXFocus.conicsolve as conic

import arcusTrace.arcusUtilities as ArcUtil
import arcusTrace.arcusPerformance as ArcPerf
import arcusTrace.ParamFiles.arcus_params_rev1p7 as cfpar

####################################################################
# CAT Grating Related Functions

def grat_ray_select(rays,xgrat,ygrat,zgrat,tgrat,pgrat,ngrat,xwidth,ywidth):
    grat_rays = ArcUtil.copy_rays(rays)
    R = make_rot_matrix(tgrat,pgrat,ngrat)
    a1,a2,a3 = tran.tr.euler_from_matrix(R,'sxyz')
    
    tran.transform(grat_rays,xgrat,ygrat,zgrat,a1,a2,a3)
    surf.flat(grat_rays)
    
    ind = (grat_rays[1] < xwidth/2)*(grat_rays[1] > -xwidth/2)*(grat_rays[2] < ywidth/2)*(grat_rays[2] > -ywidth/2)
    return [asarray(rays)[i][ind] for i in range(len(rays))]

def prop_to_grat(rays,xgrat,ygrat,zgrat,normgrat):
    # Loosely based on:
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    
    grat_rays = ArcUtil.copy_rays(rays)
    
    zhat = array([0,0,1])
    cvec,ctheta = cross(zhat,normgrat),dot(zhat,normgrat)
    skew_mat = tran.skew(cvec)
    I = array([[1,0,0],[0,1,0],[0,0,1]])
    R = I + skew_mat + dot(skew_mat,skew_mat)*1/(1 + ctheta)
    a1,a2,a3 = tran.tr.euler_from_matrix(R,'sxyz')
    
    tran.transform(grat_rays,xgrat,ygrat,zgrat,a1,a2,a3)
    surf.flat(grat_rays)
    tran.transform(grat_rays,0,0,0,0, 0,-a3)
    tran.transform(grat_rays,0,0,0,0,-a2,0)
    tran.transform(grat_rays,0,0,0,-a1,0,0)
    tran.transform(grat_rays,-xgrat,-ygrat,-zgrat,0,0,0)
    return grat_rays

def make_rot_matrix(tgrat,pgrat,ngrat):
    return transpose([tgrat,pgrat,ngrat])

def CATgratTrace(rays,order,xgrat,ygrat,zgrat,tgrat,pgrat,ngrat,dgrat = 2.00e-4):
    '''
    Loosely based on: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    '''
    grat_rays = ArcUtil.copy_rays(rays)
    R = make_rot_matrix(tgrat,pgrat,ngrat)
    a1,a2,a3 = tran.tr.euler_from_matrix(R,'sxyz')
    
    tran.transform(grat_rays,xgrat,ygrat,zgrat,a1,a2,a3)
    surf.flat(grat_rays)
    
    tran.grat(grat_rays,dgrat,order,grat_rays[0])
    tran.reflect(grat_rays)
    
    tran.transform(grat_rays,0,0,0,0, 0,-a3)
    tran.transform(grat_rays,0,0,0,0,-a2,0)
    tran.transform(grat_rays,0,0,0,-a1,0,0)
    tran.transform(grat_rays,-xgrat,-ygrat,-zgrat,0,0,0)
    return grat_rays

def GratPetalTrace(rays,order_select = None,apply_support_structure = True):
    '''
    Inputs:
    order_select: an integer that sets what order is raytraced for all gratings within the petal. If 'None',
    the order selection is done by the grating efficiency function linked to the Arcus parameters.
    '''
    # If the order selection is specified by a grating efficiency function,
    # calculate the interpolation function once for all gratings to be traced (outside loop).
    if order_select == None:
        geff_func = ArcPerf.make_geff_interp_func()
    
    for j in range(cfpar.N_grats):
        # Selecting the rays hitting the jth grating and creating a separate selection of "single grating rays" -- sgrat_rays
        sgrat_rays = grat_ray_select(rays,cfpar.xgrats[j],cfpar.ygrats[j],cfpar.zgrats[j],\
                                                cfpar.tgrats[j],cfpar.pgrats[j],cfpar.ngrats[j],cfpar.grat_dims[0],cfpar.grat_dims[1])
        
        if apply_support_structure == True:
            sgrat_rays,L1_vig_ind = ArcPerf.apply_support_structure(sgrat_rays,support = 'L1')
            sgrat_rays,L2_vig_ind = ArcPerf.apply_support_structure(sgrat_rays,support = 'L2')
            
        # Negative sign necessary due to definition of grating normal away from telescope focus, and ray direction towards the focus.
        thetas = anal.indAngle(sgrat_rays,normal = -cfpar.ngrats[j])*180/pi
        
        # Creating the order vectors -- they can be specified to have the appropriate order distribution based on a grating efficiency
        # function or given by the user by itself.
        if isinstance(order_select,int):
            order = ones(len(sgrat_rays[0]))*order_select
        elif order_select is None:
            # The geff_func is typically specified in terms of nanometers. The geff_func is currently programmed to output
            # an order of -1000 if the ray would be absorbed.
            order,crit,order_cdf = ArcPerf.pick_order(geff_func,thetas,sgrat_rays[0]*10**6)
            nonabsorbed_ind = order != -1000
            sgrat_rays = tran.vignette(sgrat_rays,ind = nonabsorbed_ind)
            order = order[nonabsorbed_ind]
        else:
            raise TypeError("Variable 'order_select' is not an integer or None -- this is an issue.")
        
        # Now actually diffracting just the rays hitting the jth grating.
        sdiff_rays = CATgratTrace(sgrat_rays,order,cfpar.xgrats[j],cfpar.ygrats[j],cfpar.zgrats[j],\
                                         cfpar.tgrats[j],cfpar.pgrats[j],cfpar.ngrats[j])
    
        #pdb.set_trace()
        # Now tracking the output of diffracted rays (diff_rays) and which ray hits which grating. This effectively "auto-vignettes"
        # rays that don't pass through a grating, so some care is required.
        if j == 0:
            diff_rays = sdiff_rays
            diff_orders = order
            grat_hit = ones(len(sdiff_rays[0]))*j
        else:
            diff_rays = hstack((diff_rays,sdiff_rays))
            diff_orders = hstack((diff_orders,order))
            grat_hit = hstack((grat_hit,ones(len(sdiff_rays[0]))*j))
    return diff_rays,diff_orders,grat_hit


