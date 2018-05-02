from numpy import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d,UnivariateSpline
from scipy.special import wofz,beta,gamma
import time
import os
import pdb
import pickle

import PyXFocus.sources as source
import PyXFocus.surfaces as surf
import PyXFocus.analyses as anal
import PyXFocus.transformations as tran
import PyXFocus.grating as grat
import PyXFocus.conicsolve as conic
import PyXFocus.transformMod as transM
import PyXFocus.analyses as anal

import arcusTrace.arcusPerformance as ArcPerf
import arcusTrace.arcusUtilities as ArcUtil
import arcusTrace.arcusRays as ArcRays
import arcusTrace.ParamFiles.arcus_params_rev1p8 as cfpar

####################################################################
# Probability Functions
####################################################################

def lorentzian(x,loc,l):
    return 1./(pi*l*(1 + ((x - loc)/l)**2))
    
def voigt(x, alpha = 0.019949, gamma = 0.0282896):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / sqrt(2 * log(2))

    return real(wofz((x + 1j*gamma)/sigma/sqrt(2))) / (sigma*sqrt(2*pi))

def laplace(x,loc,b):
    return 1./(2*b)*exp(-abs(x - loc)/b)

def logistic(x,loc,s):
    return 1./(4*s)/(cosh((x - loc)/2/s)**2)

def pearson_iv(x,loc,m,nu,alpha):
    norm = real(gamma(m + nu/2*1j)/gamma(m))**2/alpha/beta(m - 0.5,m)
    return norm*((1 + ((x - loc)/alpha)**2)**-m)*exp(-nu*arctan((x - loc)/alpha))

def make_cdf(func,args,sample_factor):
    x = linspace(-sample_factor,sample_factor,20001)
    pdf = func(x,*args)
    cdf = cumsum(pdf)/sum(pdf)
    # Needed to ensure that cdf starts at 0 outside the bounds of the sample factor.
    x_mod,cdf_mod = hstack((x[0] - (x[0] + x[1]),x)),hstack((array([0.0]),cdf))
    return x_mod,cdf_mod

def lorentzian_draw(size,scale):
    args = [0.,scale]
    x,cdf = make_cdf(lorentzian,args,scale*20)
    try:
        cdf_draw = interp1d(cdf,x,kind = 'cubic')
    except ValueError:
        pdb.set_trace()
    return cdf_draw(random.random(size)) 

def voigt_draw(size,alpha,gamma):
    args = [alpha,gamma]
    x,cdf = make_cdf(voigt,args,20*max(args))
    try:
        cdf_draw = interp1d(cdf,x,kind = 'cubic')
    except ValueError:
        pdb.set_trace()
    return cdf_draw(random.random(size))

def laplace_draw(size,scale):
    args = [0.,scale]
    x,cdf = make_cdf(laplace,args,20*scale)
    try:
        cdf_draw = interp1d(cdf,x,kind = 'cubic')
    except ValueError:
        pdb.set_trace()
    return cdf_draw(random.random(size))

def logistic_draw(size,scale):
    args = [0.,scale]
    x,cdf = make_cdf(logistic,args,20*scale)
    try:
        cdf_draw = interp1d(cdf,x,kind = 'cubic')
    except ValueError:
        pdb.set_trace()
    return cdf_draw(random.random(size))

def pearson_iv_draw(size,m,nu,alpha):
    args = [0.0,m,nu,alpha]
    x,cdf = make_cdf(pearson_iv,args,100*alpha)
    try:
        cdf_draw = interp1d(cdf,x,kind = 'cubic')
    except ValueError:
        pdb.set_trace()
    return cdf_draw(random.random(size))

def pearson_vii_draw(size,m,alpha):
    ''' When nu fixed at 0.0, becomes Pearson VII distribution'''
    args = [0.0,m,0.0,alpha]
    x,cdf = make_cdf(pearson_iv,args,20*alpha)
    try:
        cdf_draw = interp1d(cdf,x,kind = 'cubic')
    except ValueError:
        pdb.set_trace()
    return cdf_draw(random.random(size))

def empirical_draw(size,scale,fn = '/Users/Casey/Dropbox/Arcus/GWAT/Raytraces/gwatAnalTrace/180325_ComboFit_Rev10.txt'):
    pdf = loadtxt(fn)
    x = linspace(-20*scale,20*scale,len(pdf))
    cdf = cumsum(pdf)/sum(pdf)
    # Needed to ensure that cdf starts at 0 outside the bounds of the sample factor.
    x_mod,cdf_mod = hstack((x[0] - 40*scale/len(pdf),x)),hstack((array([0.0]),cdf))
    # Checking that this is strictly increasing:
    compare_adjacent = [cdf_mod[i+1] <= cdf_mod[i] for i in range(len(cdf_mod) - 1)]
    bad_ind = [i for i, x in enumerate(compare_adjacent) if x]
    for ind in bad_ind:
        next_good_ind = next(i for i in range(len(cdf_mod[ind:])) if cdf_mod[ind + i] > cdf_mod[ind])
        cdf_mod[ind:ind + next_good_ind + 1] = linspace(cdf_mod[ind],cdf_mod[ind + next_good_ind],next_good_ind + 1) 
    
    cdf_draw = interp1d(cdf_mod,x_mod,kind = 'cubic')
    return cdf_draw(random.random(size))

####################################################################
# SPO-Related Functions
####################################################################

def id_XOU_for_rays(ray_object,xou_dict):
    rays = ray_object.yield_prays()
    xou_hit = empty(len(rays[0]))
    xou_hit.fill(NaN)
    
    for key in xou_dict.keys():
        clocked_rays = ArcUtil.do_ray_transform_to_coordinate_system(rays,xou_dict[key].xou_coords)
        surf.flat(clocked_rays)
        radii = sqrt(clocked_rays[1]**2 + clocked_rays[2]**2)
        bool_list = logical_and(logical_and(radii > xou_dict[key].inner_radius,radii < xou_dict[key].outer_radius),abs(clocked_rays[1]) <= xou_dict[key].azwidth/2)
        xou_hit[bool_list] = xou_dict[key].xou_num
    ray_object.xou_hit = xou_hit

def XOUTrace(ray_object,xou):
    """
    Trace a set of rays through an XOU. Incorporates ray plate selection and
    plate-by-plate vignetting, but does not yet incorporate pore width effects.
    #Inputs:
    # xou -- an xou object from ArcusComponents.
    # xou_rays -- a PyXFocus ray list.
    #Outputs:
    # xou_rays = a ray object centered on the SPO, with:
    #    x - the radial coordinates, zeroed at the center of the SPO radius,
    #    y - the azimuthal coordinates, zeroed at the center of the SPO module
    #    z - the axial coordinate, with positive z pointed towards the source,
    #    and zeroed at the center of the SPO interface.
    """

    # Generating an internal ray object to play nicely with PyXFocus routines.
    input_rays = ray_object.yield_prays()
    wavelength = ray_object.wave
    int_rays = ArcUtil.do_ray_transform_to_coordinate_system(input_rays,xou.xou_coords)
    
    surf.flat(int_rays)
    
    ray_radius = sqrt(int_rays[1]**2+int_rays[2]**2)
    v_ind_all = zeros(len(int_rays[1]),dtype = bool)
    # Bookkeeping variables in case checking how many rays get vignetted (and through what process) is necessary.
    geo_v,ref_v = 0,0
    
    if xou.pore_vignette == True:
        v_ind_all = ArcPerf.apply_support_vignetting(int_rays,support = 'PoreStructure')
    
    for r in xou.plate_radii:
        # Collect relevant rays: tr_ind are the rays that should be traced,
        # v_ind are the rays that should be vignetted (impacting plates).
        # Note that the sidewall vignetting is completely ignored.
        tr_ind = logical_and(ray_radius > r,ray_radius < r + xou.pore_space)
        v_ind = logical_and(ray_radius > r + xou.pore_space, ray_radius < r + xou.plate_height)
        geo_v = geo_v + sum(v_ind)
        v_ind_all = logical_or(v_ind,v_ind_all)
        
        # In case our sampling is so sparse that there are no rays fulfilling this condition.
        if sum(tr_ind) == 0: 
            continue
            
        surf.spoPrimary(int_rays,r,xou.focal_length,ind = tr_ind)
        tran.reflect(int_rays,ind = tr_ind)
        if xou.ref_func is not None:
            # Calculating the rays lost to absorption. This enters as vignetting after passing through the entire SPO.
            v_ind_ref = ArcPerf.ref_vignette_ind(int_rays,wavelength,xou.ref_func,ind = tr_ind)
            ref_v = ref_v + sum(v_ind_ref)
            v_ind_all = logical_or(v_ind_ref,v_ind_all)
        
        surf.spoSecondary(int_rays,r,xou.focal_length,ind = tr_ind)
        tran.reflect(int_rays,ind = tr_ind)
        if xou.ref_func is not None:
            # Calculating the rays lost to absorption. This enters as vignetting after passing through the entire SPO.
            v_ind_ref = ArcPerf.ref_vignette_ind(int_rays,wavelength,xou.ref_func,ind = tr_ind)
            ref_v = ref_v + sum(v_ind_ref)
            v_ind_all = logical_or(v_ind_ref,v_ind_all)
    
    # Add normalized scatter to the direction cosines if desired.
    if xou.scatter[0] == 'Gaussian':
        int_rays[4] = int_rays[4] + random.normal(scale=xou.dispdir_scatter_val,size=shape(int_rays)[1])
    elif xou.scatter[0] == 'Lorentz':
        int_rays[4] = int_rays[4] + lorentzian_draw(shape(int_rays)[1],xou.dispdir_scatter_val)
    elif xou.scatter[0] == 'Voigt':
        int_rays[4] = int_rays[4] + voigt_draw(shape(int_rays)[1],xou.dispdir_scatter_val[0],xou.dispdir_scatter_val[1])
    elif xou.scatter[0] == 'Laplace':
        int_rays[4] = int_rays[4] + laplace_draw(shape(int_rays)[1],xou.dispdir_scatter_val)
    elif xou.scatter[0] == 'Logistic':
        int_rays[4] = int_rays[4] + logistic_draw(shape(int_rays)[1],xou.dispdir_scatter_val)
    elif xou.scatter[0] == 'PearsonIV':
        int_rays[4] = int_rays[4] + pearson_iv_draw(shape(int_rays)[1],xou.dispdir_scatter_val[0],xou.dispdir_scatter_val[1],xou.dispdir_scatter_val[2])
    elif xou.scatter[0] == 'PearsonVII':
        int_rays[4] = int_rays[4] + pearson_vii_draw(shape(int_rays)[1],xou.dispdir_scatter_val[0],xou.dispdir_scatter_val[1])
    elif xou.scatter[0] == 'Empirical':
        int_rays[4] = int_rays[4] + empirical_draw(shape(int_rays)[1],xou.dispdir_scatter_val)


    if xou.scatter[1] == 'Gaussian':
        int_rays[5] = int_rays[5] + random.normal(scale=xou.crossdispdir_scatter_val,size=shape(int_rays)[1])
    elif xou.scatter[1] == 'Lorentz':
        int_rays[5] = int_rays[5] + lorentzian_draw(shape(int_rays)[1],xou.crossdispdir_scatter_val)
    elif xou.scatter[1] == 'Voigt':
        int_rays[5] = int_rays[5] + voigt_draw(shape(int_rays)[1],xou.crossdispdir_scatter_val[0],xou.crossdispdir_scatter_val[1])
    elif xou.scatter[1] == 'Laplace':
        int_rays[5] = int_rays[5] + laplace_draw(shape(int_rays)[1],xou.crossdispdir_scatter_val)
    elif xou.scatter[1] == 'Logistic':
        int_rays[5] = int_rays[5] + logistic_draw(shape(int_rays)[1],xou.crossdispdir_scatter_val)
    elif xou.scatter[1] == 'PearsonIV':
        int_rays[5] = int_rays[5] + pearson_iv_draw(shape(int_rays)[1],xou.crossdispdir_scatter_val[0],xou.crossdispdir_scatter_val[1],xou.crossdispdir_scatter_val[2])
    elif xou.scatter[1] == 'PearsonVII':
        int_rays[5] = int_rays[5] + pearson_vii_draw(shape(int_rays)[1],xou.crossdispdir_scatter_val[0],xou.crossdispdir_scatter_val[1])
    elif xou.scatter[1] == 'Empirical':
        int_rays[5] = int_rays[5] + empirical_draw(shape(int_rays)[1],xou.crossdispdir_scatter_val)

    int_rays[6] = -sqrt(1.- int_rays[5]**2 - int_rays[4]**2)
    # Preventing the above from being poorly defined (in case int_rays[5]**2 + int_rays[6]**2 greater than 1)
    v_ind_all = logical_or(v_ind_all,isnan(int_rays[6]))
    
    # Now we apply the vignetting to all the PyXFocus rays, and undo the original
    # coordinate transformation (i.e. undoing the transform to the xou_coords done
    # at the start of this function)
    raw_xou_rays = tran.vignette(int_rays,ind = ~v_ind_all)
    reref_xou_rays = ArcUtil.undo_ray_transform_to_coordinate_system(raw_xou_rays,xou.xou_coords)
    
    # Finally, reconstructing a vignetted ray object with the correctly tracked parameters, and
    # setting the ray objects PyXFocus rays to be those traced here.
    xou_ray_object = ray_object.yield_object_indices(ind = ~v_ind_all)
    xou_ray_object.set_prays(reref_xou_rays)

    return xou_ray_object

def SPOPetalTrace(ray_object,xou_dict):
    # Identifying the XOUs hit for all of the input rays.
    id_XOU_for_rays(ray_object,xou_dict)

    petal_ray_dict = dict()
    # Looping through the entire dictionary of XOUs. 
    for key in xou_dict.keys():
        ray_ind_this_xou = ray_object.xou_hit == xou_dict[key].xou_num
        xou_ray_object = ray_object.yield_object_indices(ind = ray_ind_this_xou)
        if size(xou_ray_object.x) != 0:
            try:
                petal_ray_dict[key] = XOUTrace(xou_ray_object,xou_dict[key])
            except:
                pdb.set_trace()
                continue       
    petal_ray_object = ArcRays.merge_ray_object_dict(petal_ray_dict)
    return petal_ray_object
