from numpy import *
import PyXFocus.transformations as tran
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy

import pdb
from scipy.optimize import root

####################################################################
# Petal Optical Design Configuration File
# Arcus Rev. 1p8 -- Implementing final torus radii, new SPO table and grating points file.
####################################################################
F = 12000.
pore_space = 0.605
plate_space = 0.775

# Enabling either SiC, Ir, or the "uncoated" variant with lower roughness.
MM_coat_mat = 'SingleLayerMirror_SiO2Layer_SiSubstrate1p0nmThick_' #'SiC'
MM_coat_rough = 4 #10

#################
# Lookup functionality.
def row_param_construct(row_param1,row_param2):
    xou_param = []
    for i in range(0,number_of_mms*2,2):
        xou_param.append(row_param1[row_number[i] - 2])
        xou_param.append(row_param2[row_number[i + 1] - 2])
    return xou_param

#################
# Row parameters.
number_of_mms = 34
row_num = arange(2,10)

# Inner and outer radii are taken from Cosine-SPO-geometry-summary-11Oct2017.xls.
# Inner stacks are denoted MMO, outer stacks are denoted MM1.
# Inner radii come from column "Stack Inner Radius (Plate #34)".
# Outer radii come from column "Radius of Outer Mirror (Plate #1)".
# Stack lengths come from "Stack length axial (Z) direction".
# Stack widths come from "Stack width azimuthal (X) direction"

row_MM0_ir = array([ 321.145,  389.145,  457.145,  525.145,  593.145,  661.145, 729.145,  797.145])
row_MM0_or = array([ 346.72,  414.72,  482.72,  550.72,  618.72,  686.72,  754.72, 822.72])
row_MM1_ir = array([ 348.875,  416.875,  484.875,  552.875,  620.875,  688.875, 756.875,  824.875])
row_MM1_or = array([ 374.45,  442.45,  510.45,  578.45,  646.45,  714.45,  782.45, 850.45])

row_MM0_length = array([ 83.756,  70.023,  60.159,  52.731,  46.936,  42.288,  38.478, 35.298])
row_MM1_length = array([ 77.554,  65.635,  56.891,  50.203,  44.922,  40.647,  37.114, 34.147])

row_MM0_w = array([50.0,50.0,50.0,83.0,83.0,83.0,83.0,83.0])
row_MM1_w = array([50.0,50.0,50.0,83.0,83.0,83.0,83.0,83.0])

#################
# SPO MM parameters.

# Clocking angles from centerline.
mm_cangles = array([-19.2,-6.5,6.5,19.2,\
                 -15.8,-5.4,5.4,15.8,\
                 -22.5,-13.5,-4.6,4.6,13.5,22.4,\
                 -17.1,-5.8,5.8,17.1,\
                 -15.2,-5.1,5.1,15.2,\
                 -13.6,-4.6,4.6,13.6,\
                 -12.3,-4.2,4.2,12.3,\
                 -11.3,-3.9,3.9,11.3])

# Offsets along the optical axis.
zoffsets = zeros(number_of_mms)

#################
# XOU parameters.
N_xous = number_of_mms*2
# Ordered XOU Numbers.
xou_number = arange(N_xous)

# Row number of each XOU - employed for lookup.
row_number = array([2,2,2,2,2,2,2,2,
                    3,3,3,3,3,3,3,3,
                    4,4,4,4,4,4,4,4,4,4,4,4,
                    5,5,5,5,5,5,5,5,
                    6,6,6,6,6,6,6,6,
                    7,7,7,7,7,7,7,7,
                    8,8,8,8,8,8,8,8,
                    9,9,9,9,9,9,9,9])

arcus_row_number = row_number - 1
MM_number = repeat(arange(34),2)
chan_number = ones(N_xous,dtype = int)

xou_irs = row_param_construct(row_MM0_ir,row_MM1_ir)
xou_ors = row_param_construct(row_MM0_or,row_MM1_or)
xou_lengths = row_param_construct(row_MM0_length,row_MM1_length)
xou_widths = row_param_construct(row_MM0_w,row_MM1_w)
xou_cangles = repeat(mm_cangles,2)
xou_zoffsets = zeros(N_xous)
xou_rtilts = zeros(N_xous)

#################
# Grating placement equations.

# The periodicity of the gratings (200 nm) in mm.
dgrat = 2.00e-4

# The dimensions of the gratings in the dispersion direction and cross-dispersion direction respectively.
grat_dims = array([27,26])

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


grat_locs = loadtxt('/Users/Casey/Software/python_repository/arcusTrace/ParamFiles/171019_GratPoints_OC13_XYZ.txt',delimiter = ',')
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
    if row_number[i] < 5:
        temp = [grat_count,grat_count + 1]
    else:
        temp = [grat_count,grat_count + 1,grat_count + 2]
    grat_count = grat_count + len(temp)
    grat_num_by_xou.append(temp)

det_xstart = 540
det_xstep = 2048*0.024 + 0.5
det_xlocs = -(det_xstart + arange(8)*det_xstep)
det_RoC = mean(zgrats)*2

def define_det_array(xlocs,RoC):
    '''
    From the channel origin, explicitly defines the detector array in the plane y = 0
    from the x dimension locations specified. Number of detectors is given by the length
    of the location array.
    ''' 
    zlocs = RoC - sqrt(RoC**2 - xlocs**2)
    locs = array([array([xlocs[i],0,zlocs[i]]) for i in range(len(xlocs))])
    
    normals = array([array([-xlocs[i],0,RoC - zlocs[i]])/RoC for i in range(len(xlocs))])
    cross_disp_dir = array([array([0,1,0]) for i in range(len(xlocs))])
    disp_dir = array([cross(cross_disp_dir[i],normals[i]) for i in range(len(xlocs))])
    
    det_vecs = array([vstack((disp_dir[i],cross_disp_dir[i],normals[i])) for i in range(len(xlocs))])
    return locs,det_vecs

s1det_locs,s1det_vecs = define_det_array(det_xlocs,det_RoC)
s2det_locs,s2det_vecs = define_det_array(det_xlocs - 5,det_RoC)

#s1det_locs,s2det_locs = copy.deepcopy(s1det_locs),copy.deepcopy(s2det_locs)
#s1det_vecs,s2det_vecs = copy.deepcopy(s1det_vecs),copy.deepcopy(s2det_vecs)

s1det_locs[:,0] = s1det_locs[:,0] + 300
s2det_locs[:,0] = -(s2det_locs[:,0] + 300)

flip_matrix = array([[1,1,-1],[1,1,1],[-1,1,1]])
s2det_vecs = asarray([flip_matrix*s2det_vecs[i] for i in range(len(s2det_vecs))])

instrum_det_locs,instrum_det_vecs = vstack((s1det_locs,s2det_locs)),vstack((s1det_vecs,s2det_vecs))
