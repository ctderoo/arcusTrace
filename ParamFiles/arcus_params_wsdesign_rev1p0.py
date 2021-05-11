from numpy import *
import PyXFocus.transformations as tran
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy

import pdb
from scipy.optimize import root
from arcusTrace.ParamFiles.pointers import *

####################################################################
# Petal Optical Design Configuration File
# Arcus Rev. 1p8 -- Implementing final torus radii, new SPO table and grating points file.
####################################################################
F = 12000.
pore_space = 0.665
plate_space = 0.775

# Enabling either SiC, Ir, or the "uncoated" variant with lower roughness.
MM_coat_mat = 'SingleLayerMirror_SiO2Layer_SiSubstrate1p0nmThick_' #'SiC'
MM_coat_rough = 4 #10

#################
# Lookup functionality.
def row_param_construct(row_param1,row_param2):
    xou_param = []
    for i in row_number:
        xou_param.append(row_param1[i - 1])
        xou_param.append(row_param2[i - 1])
    return xou_param

#################
# Row parameters.
number_of_mms = 24
row_num = arange(1,7)

# Inner and outer radii are taken from "Arcus SPO Layout - ARCUS-200-P-30Mar2020".
# Inner stacks are denoted MMO, outer stacks are denoted MM1.
# Inner radii come from column "Inner Radius (mm)"
# Outer radii come from column "Outer Radius (mm).

row_MM0_ir = array([ 385.418,   453.018,    520.618,    588.118,    655.718,    723.318])
row_MM0_or = array([ 414.868,   482.468,    550.068,    617.568,    685.168,    752.768])
row_MM1_ir = array([ 415.533,   483.133,    550.733,    618.233,    685.833,    753.433])
row_MM1_or = array([ 444.983,   512.583,    580.183,    647.683,    715.283,    782.883])

# Inner and outer radii are taken from "Arcus 2020 - SPO MM geometry 25Mar2020"
# Kink radii come from "Radius at Kink of Reference (Plate #18)".
# Lengths come from Stack length axial (Z) direction
# Widths come from "Stack width azimuthal (X) direction"
row_MM0_kink = array([ 400.143,   467.743,    535.343,    602.843,    670.443,    738.043])
row_MM1_kink = array([ 430.258,   497.858,    565.458,    632.958,    700.558,    768.158])

row_MM0_length = array([82.819, 70.461, 61.312, 54.275, 48.679, 44.130])
row_MM1_length = array([76.817, 66.069, 57.959, 51.631, 46.542, 42.366])

row_MM0_w = array([76.7715, 71.6688, 68.2668, 80.0597, 75.9448, 86.0527])
row_MM1_w = array([76.7715, 71.6688, 68.2668, 80.0597, 75.9448, 86.0527])

# Finally, a calculation of the z0_focal length, as defined for Wolter-I types. This corresponds to 
# the z-placement on the optical axis.
row_MM0_z0 = sqrt(F**2 - row_MM0_kink**2) 
row_MM1_z0 = sqrt(F**2 - row_MM1_kink**2) 
#################
# SPO MM parameters.

# Clocking angles are taken from "Arcus SPO Layout - ARCUS-200-P-30Mar2020".
# Clocking angles from centerline.
mm_cangles = array([-23.602,-7.867,7.867,23.602,\
                 -20.062,-6.687,6.687,20.062,\
                 -17.448,-5.816,5.816,17.448,\
                 -15.439,-5.146,5.146,15.439,\
                 -13.844,-4.615,4.615,13.844,\
                 -12.548,-4.183,4.183,12.548])

# Offsets along the optical axis.
zoffsets = zeros(number_of_mms)

#################
# XOU parameters.
N_xous = number_of_mms*2
# Ordered XOU Numbers.
xou_number = arange(N_xous)

# Row number of each XOU - employed for lookup.
num_of_rows = 6
row_number = repeat(arange(1,7),4)

arcus_xou_row_number = repeat(arange(1,7),8) - 1
MM_number = repeat(arange(number_of_mms),2)
chan_number = ones(N_xous,dtype = int)

xou_irs = row_param_construct(row_MM0_ir,row_MM1_ir)
xou_ors = row_param_construct(row_MM0_or,row_MM1_or)
xou_kinkr = row_param_construct(row_MM0_kink,row_MM1_kink)

xou_lengths = row_param_construct(row_MM0_length,row_MM1_length)
xou_widths = row_param_construct(row_MM0_w,row_MM1_w)
xou_cangles = repeat(mm_cangles,2)
xou_z0 = row_param_construct(row_MM0_z0,row_MM1_z0)
xou_zoffsets = zeros(N_xous)
xou_rtilts = zeros(N_xous)

#################
# Grating placement equations.

# The periodicity of the gratings (200 nm) in mm.
dgrat = 2.00e-4

# The dimensions of the gratings in the dispersion direction and cross-dispersion direction respectively.
grat_dims = array([28,28.5])

#def compute_xy_grat_locs(inr,outr,cangle,row):
#    if row >= 5:
#        return (inr + outr)/2*sin(cangle*pi/180) + array([-grat_dims[0],0,grat_dims[0]]), (inr + outr)/2*cos(cangle*pi/180) + array([0,0,0])
#    else:
#        return (inr + outr)/2*sin(cangle*pi/180) + array([-grat_dims[0]/2,grat_dims[0]/2]), (inr + outr)/2*cos(cangle*pi/180) + array([0,0])
#
#xgrats,ygrats = array([]),array([])
#
#for i in range(len(row_number)):
#    xtemp,ytemp = compute_xy_grat_locs(xou_irs[i],xou_ors[i],xou_cangles[i],row_number[i])
#    xgrats = hstack((xgrats,xtemp))
#    ygrats = hstack((ygrats,ytemp))

grat_path = 'C:/Users/Casey/Software/python_repository/arcusTrace/ParamFiles/200409_GratPoints_OC13_XYZ.csv'
grat_locs = loadtxt(grat_path,delimiter = ',')
xgrats,ygrats,zgrats = grat_locs[:,0],grat_locs[:,1],grat_locs[:,2]

N_grats = len(xgrats)

# Computing the zposition of the gratings based on the Rowland Torus prescription.
'''
Double-Tilted Rowland Torus (origin at channel):
xhat and zhat are in the plane of the Rowland circle, and xhat is the hinge of revolution.
alpha = blaze angle (independent from the tilt of the torus).
xi = torus tilt angle (driven by desired channel spacing)
r = the radius of the Rowland Circle.
R = the radius from the center of the Rowland circle to the "hinge".

Canonical Torus Parameters (origin at center hinge point):
theta = the angle about z' as measured from the x' axis. 
phi = the angle about y' as measured up from x' 
'''
alpha = 1.8*pi/180
xi = 2.892138*pi/180
r = 5945.787
R = 5915.513
zg = (r + R)/cos(xi)

# Derived values:
#xd = (r + R)*sin(xi)
#zd = (r + R)*tan(xi)*sin(xi)
#
#def torus_equation(xp,yp,zp):
#    return (xp**2 + yp**2 + zp**2 + R**2 - r**2)**2 - 4*R**2*(xp**2 + zp**2)
#
#def get_ct_coords(theta,phi):
#    xprime = (R + r*cos(theta))*cos(phi)
#    yprime = r*sin(theta)
#    zprime = (R + r*cos(theta))*sin(phi)
#    return x,y,z
#
#def get_ct_angs(xprime,yprime,zprime):
#    theta = arcsin(xprime/r)
#    phi = arcsin(zprime/((r + R)*cos(theta)))
#    return theta,phi
#
#def construct_ct_transform():
#    exprime = array([sin(xi),0,cos(xi),0])
#    eyprime = array([cos(xi),0,-sin(xi),0])
#    ezprime = array([0,1,0,0])
#    c = array([-xd,0,zd,1])
#    return transpose(vstack((exprime,eyprime,ezprime,c)))
#
#def comp_z(x,y):
#    mat_ct2arcus = construct_ct_transform()
#    
#    def min_func(k,x,y):
#        vec = array([x,y,k,1])
#        xp,yp,zp,nomatter = dot(linalg.inv(mat_ct2arcus),vec)    
#        return torus_equation(xp,yp,zp)
#    
#    output = root(min_func,r + R,args = (x,y))
#    return output.x
#    
#zgrats = asarray([comp_z(xgrats[i],ygrats[i])[0] for i in range(N_grats)])

'''
Now to establish the grating orientations.
'''

# First, pointing the gratings towards the telescope focus -- setting normal incidence while being positioned on the Rowland torus.
# The nominal normal of the transmission gratings, focus_ngrats, points towards the focus of telescope.
norm_ind_grats = asarray([array([xgrats[i],ygrats[i],zgrats[i]]) for i in range(N_grats)])
focus_ngrats = asarray([norm_ind_grats[i]/linalg.norm(norm_ind_grats[i]) for i in range(N_grats)])

# Constructing a grating bar vector that's 1) orthogonal to the normal vector and 2) orthogonal to x_hat. The
# weighting done in the third slot ensures orthogonality.
gbars = asarray([array([0,1.,-focus_ngrats[i][1]/focus_ngrats[i][2]]) for i in range(N_grats)])
gbars = asarray([gbars[i]/linalg.norm(gbars[i]) for i in range(N_grats)])

# Now rotating about the grating bars to set the desired blaze angle. First, we produce a rotation matrix to
# rotate another vector about the grating bar direction by the desired incidence angle alpha. Then we rotate
# the nominal transmission grating normal (pointed towards the focus) about the grating bar direction by the
# incidence angle.
ngrats = array([dot(tran.tr.rotation_matrix(alpha,gbars[i])[:3,:3],focus_ngrats[i]) for i in range(len(gbars))])

# Finally, constructing the orthogonal vector for each grating representing the local dispersion direction.
gdisp = asarray([cross(gbars[i],ngrats[i]) for i in range(N_grats)])

# Setting up these definitions as related to the torus -- the dispersion direction is related to the theta
# direction of the Rowland torus, while the gbars direction is related to the phi direction on the torus.
tgrats,pgrats = gdisp,gbars

grat_num_by_xou = []
grat_count = 0

for i in range(N_xous):
    #if row_number[i] < 5:
    #    temp = [grat_count,grat_count + 1]
    #else:
    #    temp = [grat_count,grat_count + 1,grat_count + 2]
    temp = [grat_count,grat_count + 1,grat_count + 2]
    grat_count = grat_count + len(temp)
    grat_num_by_xou.append(temp)

#det_xstart = 540
#det_xstep = 2048*0.024 + 0.5
#det_xlocs = -(det_xstart + arange(8)*det_xstep)
#det_RoC = mean(zgrats)*2

#def define_det_array(xlocs,RoC):
#    '''
#    From the channel origin, explicitly defines the detector array in the plane y = 0
#    from the x dimension locations specified. Number of detectors is given by the length
#    of the location array.
#    ''' 
#    zlocs = RoC - sqrt(RoC**2 - xlocs**2)
#    locs = array([array([xlocs[i],0,zlocs[i]]) for i in range(len(xlocs))])
    
#    normals = array([array([-xlocs[i],0,RoC - zlocs[i]])/RoC for i in range(len(xlocs))])
#    cross_disp_dir = array([array([0,1,0]) for i in range(len(xlocs))])
#    disp_dir = array([cross(cross_disp_dir[i],normals[i]) for i in range(len(xlocs))])
    
#    det_vecs = array([vstack((disp_dir[i],cross_disp_dir[i],normals[i])) for i in range(len(xlocs))])
#    return locs,det_vecs

#s1det_locs,s1det_vecs = define_det_array(det_xlocs,det_RoC)
#s2det_locs,s2det_vecs = define_det_array(det_xlocs - 5,det_RoC)

##s1det_locs,s2det_locs = copy.deepcopy(s1det_locs),copy.deepcopy(s2det_locs)
##s1det_vecs,s2det_vecs = copy.deepcopy(s1det_vecs),copy.deepcopy(s2det_vecs)

#s1det_locs[:,0] = s1det_locs[:,0] + 300
#s2det_locs[:,0] = -(s2det_locs[:,0] + 300)

#flip_matrix = array([[1,1,-1],[1,1,1],[-1,1,1]])
#s2det_vecs = asarray([flip_matrix*s2det_vecs[i] for i in range(len(s2det_vecs))])

#instrum_det_locs,instrum_det_vecs = vstack((s1det_locs,s2det_locs)),vstack((s1det_vecs,s2det_vecs))
