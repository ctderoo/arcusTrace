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

####################################################################
# Utility-related functions.

def copy_rays(rays):
    return [rays[i].copy() for i in range(len(rays))]

####################################################################
# Performance-related functions.

def compute_cdf_pdf(rays1d):
    hist,bin_edges = histogram(rays1d,bins = 10**3)
    x_locs = (bin_edges[:-1] + bin_edges[1:])/2
    pdf = hist.astype(float)/sum(hist)
    cdf = cumsum(hist).astype(float)/sum(hist)
    return cdf,pdf,x_locs
 
def hpd_cdf_intersection(cdf,x_locs):
    ind1,ind2 = argmin(abs(cdf - 0.25)),argmin(abs(cdf - 0.75))
    return x_locs[ind2] - x_locs[ind1]

def compute_hpd(rays1d):
    cdf,pdf,x_locs = compute_cdf_pdf(rays1d)
    return hpd_cdf_intersection(cdf,x_locs)

#############################################
# Ray picking functions. 

def select_xou_rays(rays,xou_hit,ind):
    return asarray(rays)[:,where(xou_hit == ind)[0]]

def select_grat_rays(rays,grat_hit,ind):
    return asarray(rays)[:,where(grat_hit == ind)[0]]

def check_normals(rays,grat_hit,ind):
    sample_rays = select_grat_rays(rays,ind)
    norms = vstack((sample_rays[4],sample_rays[5],sample_rays[6]))
    norms = transpose(norms)
    ngrat = cfpar.ngrats[ind]
    return norms,ngrat
