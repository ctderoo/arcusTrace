from numpy import *
import matplotlib.pyplot as plt
import os
import pdb
import pickle
import glob
from scipy.interpolate import RegularGridInterpolator as RGI

import PyXFocus.sources as source
import PyXFocus.surfaces as surf
import PyXFocus.analyses as anal
import PyXFocus.transformations as tran
import PyXFocus.grating as grat
import PyXFocus.conicsolve as conic

home_directory = os.getcwd()

########################################################################
# CALDB Paths and Functions
########################################################################

# Overall responses are contained here.
caldb_directory = '/Users/Casey/Software/Bitbucket/caldb-inputdata'

# Contains grating-related performance files, e.g. transmission from L1, L2 filters, order efficiency, etc.
grat_directory = caldb_directory + '/gratings'
grat_eff_fn = grat_directory + '/' + 'efficiency.csv'
grat_L1vig_fn = grat_directory + '/' + 'L1support.csv'
grat_L2vig_fn = grat_directory + '/' + 'L2support.csv'

# Contains detector-related performance files, e.g. contamination and QE
detector_directory = caldb_directory + '/detectors'
det_qe_fn = detector_directory + '/' + 'qe.csv'
det_contam_fn = detector_directory + '/' + 'contam.csv'

# Contains filter-related performance files, e.g. tranmissivity of the filters selected for Arcus.
filter_directory = caldb_directory + '/' + '/filters'
opt_block_fn = filter_directory + '/' + '/opticalblocking.csv'

def read_caldb_csvfile(fn):
    file_contents = genfromtxt(fn,comments = '#',dtype = str)
    header,data = file_contents[0],file_contents[1:].astype('float')
    return header,data

def make_geff_interp_func(grat_eff_file = grat_eff_fn):
    geff_header,geff_data = read_caldb_csvfile(grat_eff_file)
    # Hardcoding for now.
    graze,wave,order = arange(0.7,5.05,0.05),arange(0.8,2.85,0.05),range(0,13,1)
    geff = geff_data[:,2:].reshape(len(graze),len(wave),len(order))
    return RGI(points = (graze,wave,order),values = geff)

geff_func = make_geff_interp_func()

def pick_order(graze,wave,orders = range(0,13,1)):#grat_eff_file = grat_eff_fn):
    N = len(graze)
    crit = random.random(N)
    
    if len(graze) != len(wave):
        raise NameError('Graze angle and wavelength vectors are not matched')
    
    gmesh,omesh = meshgrid(graze,orders,indexing = 'ij')
    wmesh,omesh = meshgrid(wave,orders,indexing = 'ij')
    order_cdf = cumsum(geff_func((gmesh,wmesh,omesh)),axis = 1)
    
    order_vec = zeros(N) - 1
    for i in range(N):
        try:
            order_vec[i] = orders[sum(~(crit[i] < order_cdf[i]))]
        except IndexError:
            pass
    return order_vec,crit,order_cdf
    
########################################################################
# SPO MM Reflectivity
########################################################################
 
# Overall responses are contained here.
reflib_directory = '/Users/Casey/Software/ReflectLib'

def return_ref_data(material,roughness):
    '''
    Inputs:
    material -- string specifying formula as given to CXRO
    roughness -- integer specifying coating roughness in Angstroms
    Outputs:
    ref -- reflectivity of the specified thick mirror as a function of energy and graze angle
    energy -- energy (eV) matrix matched to the reflectivity
    graze -- graze angle (degrees) matrix matched to the reflectivity
    '''
    fn = glob.glob(reflib_directory + '/' + material + '*' + 'Rough' + "{0:02d}".format(roughness) + '*.npy')
    if len(fn) != 1:
        raise PrintError('Reflectivity file is not found uniquely - please check inputs.')
        pdb.set_trace()
    data = load(fn[0])
    energy,ref,graze = data[:,:,0],data[:,:,1],data[:,:,2]
    return energy,ref,graze

def make_reflectivity_func(material,roughness):
    '''
    '''
    energy,ref,graze = return_ref_data(material,roughness)
    earray = energy[0]
    garray = graze[:,0]
    ref_func = RGI(points = (garray,earray),values = ref)
    return ref_func

def ref_vignette_ind(rays,ref_func,ind = None):
    if ind is not None:
        N = len(rays[0][ind])
        crit = random.random(N)
        energy = (1240./10**6)/rays[0][ind]
        graze = anal.grazeAngle(rays,ind = ind)*180/pi
        threshold = ref_func((graze,energy))
        # Finding the numbered indices within the selection of rays given by ind.
        ind_locs = array([i for i, x in enumerate(ind) if x])
        # Defining a new vignetting vector that runs over all the rays (not just those where ind = True)
        vig_locs = zeros(len(rays[0]),dtype = bool)
        # Now selecting those rays within ind AND where the random number doesn't clear the reflectivity.
        vig_locs[ind_locs[crit > threshold]] = True
    else:
        N = len(rays[0])
        crit = random.random(N)
        energy = (1240./10**6)/rays[0]
        graze = anal.grazeAngle(rays)*180/pi
        threshold = ref_func((graze,energy))
        vig_locs = crit > threshold
    return vig_locs
#def ref_vignette_ind(graze,wave,ref_func):
#    '''
#    Generate N random numbers (crit). If these random numbers are below the reflectivity
#    (threshold) for the graze angle and energy of the ray, then the ray is kept; if it is
#    above, it is vignetted.
#    '''
#    if len(graze) != len(wave):
#        raise NameError('Graze angle and wavelength vectors are not matched')
#    
#    N = len(graze)
#    crit = random.random(N)
#    energy = 1240./wave
#    threshold = ref_func((graze,energy))
#    return crit > threshold

