from numpy import *
import PyXFocus.transformations as tran
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pdb
from scipy.optimize import root

####################################################################
# Petal Optical Design Configuration File
# Arcus Rev. 1p7 -- implementing import of Hertz CAD geometry/grating placement and new torus radii.
####################################################################
F = 12000.
pore_space = 0.605
plate_space = 0.775
MM_coat_mat = 'SiC'
MM_coat_rough = 10
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
row_MM0_ir = array([320.0,388.0,456.0,524.0,592.0,660.0,728.0,796.0])
row_MM0_or = array([347.2,415.2,483.2,551.2,619.2,687.2,755.2,823.2])
row_MM1_ir = array([348.8,416.8,484.8,552.8,620.8,688.8,756.8,824.8])
row_MM1_or = array([376.0,444.0,512.0,580.0,648.0,716.0,784.0,852.0])

row_MM0_sh = array([83.4,70.9,61.9,54.9,49.4,44.9,40.9,37.9])
row_MM1_sh = array([83.4,70.9,61.9,54.9,49.4,44.9,40.9,37.9])

row_MM0_w = array([50.0,50.0,50.0,83.0,83.0,83.0,83.0,83.0])
row_MM1_w = row_MM0_w

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

xou_irs = row_param_construct(row_MM0_ir,row_MM1_ir)
xou_ors = row_param_construct(row_MM0_or,row_MM1_or)
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


grat_locs = loadtxt('/Users/Casey/Software/python_repository/arcusTrace/ParamFiles/171013_GratPoints_OC13_XYZ.txt',delimiter = ',')
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
xi = 2.88*pi/180
r = 5961.00
R = 5930.80
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

