from numpy import *
import matplotlib.pyplot as plt
import os
import pdb
import pickle
import glob
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import interp1d

import PyXFocus.sources as source
import PyXFocus.surfaces as surf
import PyXFocus.analyses as anal
import PyXFocus.transformations as tran
import PyXFocus.grating as grat
import PyXFocus.conicsolve as conic
from arcusTrace.ParamFiles.pointers import *

home_directory = os.getcwd()

########################################################################
# CALDB Paths and Functions
########################################################################

# Overall responses are contained here.
# caldb_directory = 'C:/Users/swarm/Software/Bitbucket/caldb-inputdata'

# Contains grating-related performance files, e.g. transmission from L1, L2 filters, order efficiency, etc.
spo_directory = caldb_directory + '/spos'
pore_transmission_fn = spo_directory + '/' + 'porespecifications.csv'

# Contains grating-related performance files, e.g. transmission from L1, L2 filters, order efficiency, etc.
grat_directory = caldb_directory + '/gratings'
grat_eff_fn = grat_directory + '/' + 'efficiency.csv'
grat_L1vig_fn = grat_directory + '/' + 'L1support.csv'
grat_L2vig_fn = grat_directory + '/' + 'L2support.csv'
grat_debye_waller_fn = grat_directory + '/' + 'debyewaller.csv'

# Contains detector-related performance files, e.g. contamination and QE
detector_directory = caldb_directory + '/detectors'
det_qe_fn = detector_directory + '/' + 'qe.csv'
det_contam_fn = detector_directory + '/' + 'contam.csv'

# Contains filter-related performance files, e.g. tranmissivity of the filters selected for Arcus.
filter_directory = caldb_directory + '/' + '/filters'
opt_block_fn = filter_directory + '/' + '/opticalblocking.csv'
Si_mesh_block_fn = filter_directory + '/' + '/sifilter.csv'
uv_block_fn = filter_directory + '/' + '/uvblocking.csv'

def read_caldb_csvfile(fn):
    '''
    Read in the .csv files from the CALDB format.
    Input:
    fn -- filename to be read in
    Output:
    header -- the column labels as strings
    data -- the contents of the csvfile as a float.
    '''
    file_contents = genfromtxt(fn,comments = '#',dtype = str)
    header,data = file_contents[0],file_contents[1:].astype('float')
    return header,data

########################################################################
# Applications of straightforward geometric loss.
########################################################################

def apply_support_weighting(rays,support = 'L1',L1_support_file = grat_L1vig_fn,\
                             L2_support_file = grat_L2vig_fn,pore_transmission_file = pore_transmission_fn):
    # We are extracting the transmission efficiency of the SPO structure
    if support == 'L1':
        structure_header,structure_data = read_caldb_csvfile(L1_support_file)
    elif support == 'L2':
        structure_header,structure_data = read_caldb_csvfile(L2_support_file)
    elif support == 'PoreStructure':
        structure_header,structure_data = read_caldb_csvfile(pore_transmission_file)
    else:
        pdb.set_trace()
        
    transmission = structure_data[0]
    #structure_rays = tran.vignette(rays,ind = vig_locs)
    return transmission #structure_rays,vig_locs

def apply_support_vignetting(rays,support = 'L1',L1_support_file = grat_L1vig_fn,\
                             L2_support_file = grat_L2vig_fn,pore_transmission_file = pore_transmission_fn):
    N = len(rays[0])
    crit = random.random(N)
    if support == 'L1':
        structure_header,structure_data = read_caldb_csvfile(L1_support_file)
    elif support == 'L2':
        structure_header,structure_data = read_caldb_csvfile(L2_support_file)
    elif support == 'PoreStructure':
        structure_header,structure_data = read_caldb_csvfile(pore_transmission_file)
    else:
        pdb.set_trace()
        
    transmission = structure_data[0]
    vig_locs = crit > transmission
    #structure_rays = tran.vignette(rays,ind = vig_locs)
    return vig_locs #structure_rays,vig_locs

########################################################################
# Grating efficiency, order selection, and structure vignetting.
########################################################################

def apply_debye_waller(rays,debye_waller_file = grat_debye_waller_fn):
    def debye_waller(d,sigma):
        return exp(-2*pi*sigma/d)
    
    header,data = read_caldb_csvfile(debye_waller_file)
    d,sigma = data[0,0],data[0,1]
    threshold = debye_waller(d,sigma)
    
    N = len(rays[0])
    crit = random.random(N)
    vig_locs = crit > threshold
    #pdb.set_trace()
    #debyewaller_rays = tran.vignette(rays,ind = vig_locs) return debyewaller_rays
    return vig_locs

def apply_debye_waller_weighting(rays,debye_waller_file = grat_debye_waller_fn):
    def debye_waller(d,sigma):
        return exp(-2*pi*sigma/d)
    header,data = read_caldb_csvfile(debye_waller_file)
    d,sigma = data[0,0],data[0,1]
    threshold = debye_waller(d,sigma)*ones(len(rays[0]))
    return threshold

def make_geff_interp_func(grat_eff_file = grat_eff_fn,style = 'old'):
    '''
    From the grating efficiency file in the CALDB directory, produce an interpolation function for
    grating efficiency lookup as a function of incidence angle, wavelength, and order.
    Input:
    grat_eff_file -- path to the grating efficiency file in CALDB .csv format
    Output:
    geff_func -- a lookup function as a function of incidence angle, wavelength, and order.
    '''
    if style == 'old':
        geff_header,geff_data = read_caldb_csvfile(grat_eff_file)
    else:
        geff_data = genfromtxt(grat_eff_file,delimiter = '\t',dtype = float)
    # Sorting the list.
    wave,theta,order = unique(geff_data[:,0]),unique(geff_data[:,1]),range(-4,16,1)
    geff = geff_data[:,2:].reshape(len(wave),len(theta),len(order))
    geff_func = RGI(points = (wave,theta,order),values = geff)
    return geff_func

def pick_order(geff_func,theta,wave,orders = range(-4,16,1)):
    '''
    '''
    N = len(theta)
    crit = random.random(N)
    
    if len(theta) != len(wave):
        raise IndexError('Graze angle and wavelength vectors are not matched')
    
    gmesh,omesh = meshgrid(theta,orders,indexing = 'ij')
    wmesh,omesh = meshgrid(wave,orders,indexing = 'ij')
    try:
        order_cdf = cumsum(geff_func((wmesh,gmesh,omesh)),axis = 1)
    except ValueError:
        pdb.set_trace()
        
    order_vec = zeros(N) - 1000
    for i in range(N):
        try:
            order_vec[i] = orders[sum(~(crit[i] < order_cdf[i]))]
        except IndexError:
            pass
    #pdb.set_trace()
    return order_vec,crit,order_cdf
    
########################################################################
# SPO MM Reflectivity
########################################################################
 
# Overall responses are contained here.
# reflib_directory = 'C:/Users/swarm/Software/ReflectLib'

def return_ref_data(pointer):#,mirror_type = 'Thick',layer_thickness = 1.0):
    '''
    Inputs:
    material -- string specifying formula as given to CXRO
    roughness -- integer specifying coating roughness in Angstroms
    Outputs:
    ref -- reflectivity of the specified thick mirror as a function of energy and graze angle
    energy -- energy (eV) matrix matched to the reflectivity
    graze -- graze angle (degrees) matrix matched to the reflectivity
    '''

    fn = pointer
    try:
        data = load(fn)
    except:
        pdb.set_trace()
    energy,ref,graze = data[:,:,0],data[:,:,1],data[:,:,2]
    return energy,ref,graze

def make_reflectivity_func(pointer):
    '''
    '''
    energy,ref,graze = return_ref_data(pointer)
    #pdb.set_trace()
    earray = energy[:,0]
    garray = graze[0]
    ref_func = RGI(points = (earray,garray),values = ref)
    #earray = energy[0]
    #garray = graze[:,0]
    #ref_func = RGI(points = (garray,earray),values = ref)
    return ref_func

def ref_vignette_ind(rays,wave,ref_func,ind = None):
    if ind is not None:
        N = len(rays[0][ind])
        crit = random.random(N)
        energy = (1240./10**6)/wave[ind]
        graze = anal.grazeAngle(rays,ind = ind)*180/pi
        threshold = ref_func((energy,graze))
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
        threshold = ref_func((energy,graze))
        vig_locs = crit > threshold
    return vig_locs

def ref_weighting_ind(rays,wave,ref_func,ind = None):
    if ind is not None:
        energy = (1240./10**6)/wave[ind]
        graze = anal.grazeAngle(rays,ind = ind)*180/pi
        threshold = ref_func((energy,graze))
        # Finding the numbered indices within the selection of rays given by ind.
        ind_locs = array([i for i, x in enumerate(ind) if x])
        # Defining a new vignetting vector that runs over all the rays (not just those where ind = True)
        vig_locs = ones(len(rays[0]),dtype = float)
        # Now selecting those rays within ind AND where the random number doesn't clear the reflectivity.
        vig_locs[ind_locs] = threshold
    else:
        # N = len(rays[0])
        # crit = random.random(N)
        energy = (1240./10**6)/rays[0]
        graze = anal.grazeAngle(rays)*180/pi
        threshold = ref_func((energy,graze))
        vig_locs = threshold
    return vig_locs

########################################################################
# Detector Absorption and QE
########################################################################

def make_detector_effect_func(eff_caldb_fn):
    header,data = read_caldb_csvfile(eff_caldb_fn)
    wave,effect = (1.240/data[:,0]),data[:,1]
    interp_func = interp1d(wave,effect,kind = 'cubic')
    return interp_func

def apply_detector_effect_vignetting(rays,wave,interp_func):
    N = len(rays[0])
    crit = random.random(N)
    wave_nm = wave*10**6
    threshold = interp_func(wave_nm)
    vig_locs = crit > threshold
    #pdb.set_trace()
    #effect_applied_rays = tran.vignette(rays,ind = vig_locs) # effect_applied_rays,
    return vig_locs

def apply_detector_effect_weighting(rays,wave,interp_func):
    wave_nm = wave*10**6
    threshold = interp_func(wave_nm)
    #pdb.set_trace()
    #effect_applied_rays = tran.vignette(rays,ind = vig_locs) # effect_applied_rays,
    return threshold