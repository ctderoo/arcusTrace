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
opt_block_fn = filter_directory + '/' + '/sifilter.csv'
opt_block_fn = filter_directory + '/' + '/uvblocking.csv'

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
# Grating efficiency, order selection, and structure vignetting.
########################################################################

def apply_support_vignetting(rays,support = 'L1',L1_support_file = grat_L1vig_fn,L2_support_file = grat_L2vig_fn):
    N = len(rays[0])
    crit = random.random(N)
    if support == 'L1':
        structure_header,structure_data = read_caldb_csvfile(L1_support_file)
    elif support == 'L2':
        structure_header,structure_data = read_caldb_csvfile(L2_support_file)
    transmission = structure_data[0]
    vig_locs = crit < threshold
    structure_rays = tran.vignette(rays,ind = vig_locs)
    return structure_rays,vig_locs

def make_geff_interp_func(grat_eff_file = grat_eff_fn):
    '''
    From the grating efficiency file in the CALDB directory, produce an interpolation function for
    grating efficiency lookup as a function of incidence angle, wavelength, and order.
    Input:
    grat_eff_file -- path to the grating efficiency file in CALDB .csv format
    Output:
    geff_func -- a lookup function as a function of incidence angle, wavelength, and order.
    '''
    geff_header,geff_data = read_caldb_csvfile(grat_eff_file)
    # Hardcoding for now.
    wave,theta,order = arange(0.7,5.05,0.05),arange(0.8,2.85,0.05),range(0,13,1)
    geff = geff_data[:,2:].reshape(len(wave),len(theta),len(order))
    geff_func = RGI(points = (wave,theta,order),values = geff)
    return geff_func

#geff_func = make_geff_interp_func()

def pick_order(geff_func,theta,wave,orders = range(0,13,1)):
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

########################################################################
# Detector Absorption and QE
########################################################################

################
# Detector QE

def make_det_qe_func(det_qe_file = det_qe_fn):
    qe_header,qe_data = read_caldb_csvfile(det_qe_file)
    wave,qe = (1.240/qe_data[:,0]),qe_data[:,1]
    qe_func = interp1d(wave,qe,kind = 'cubic')
    return qe_func

def det_qe_vignette(rays,qe_func):
    N = len(rays[0])
    crit = random.random(N)
    wave_nm = rays[0]*10**6
    threshold = qe_func(wave_nm)
    vig_locs = crit < threshold
    qe_applied_rays = tran.vignette(rays,ind = vig_locs)
    return qe_applied_rays,vig_locs

################
# Detector Contamination

def make_det_contam_func(det_contam_file = det_contam_fn):
    contam_header,contam_data = read_caldb_csvfile(det_contam_file)
    wave,contam = (1.240/contam_data[:,0]),contam_data[:,1]
    contam_func = interp1d(wave,contam,kind = 'cubic')
    return contam_func

def det_contam_vignette(rays,contam_func):
    N = len(rays[0])
    crit = random.random(N)
    wave_nm = rays[0]*10**6
    threshold = contam_func(wave_nm)
    vig_locs = crit < threshold
    contam_applied_rays = tran.vignette(rays,ind = vig_locs)
    return contam_applied_rays,vig_locs

################
# Filter Absorption

def make_filter_abs_func(filter_abs_file = opt_block_fn):
    filter_header,filter_data = read_caldb_csvfile(filter_abs_file)
    wave,filter_trans = (1.240/filter_data[:,0]),filter_data[:,1]
    filter_func = interp1d(wave,filter_trans,kind = 'cubic')
    return filter_func

def filter_abs_vignette(rays,filter_func):
    N = len(rays[0])
    crit = random.random(N)
    wave_nm = rays[0]*10**6
    threshold = filter_func(wave_nm)
    vig_locs = crit < threshold
    filter_applied_rays = tran.vignette(rays,ind = vig_locs)
    return filter_applied_rays,vig_locs