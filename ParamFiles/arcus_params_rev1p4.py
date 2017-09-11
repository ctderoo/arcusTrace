from numpy import *
import PyXFocus.transformations as tran
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

####################################################################
# SPO Petal Optical Design Configuration File
# Arcus Rev. 1p4 - studying "quasiconfocal" SPOs.
####################################################################
# Focal length of all the SPOs.

F = 12000.
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

xou_loc_irs = row_param_construct(row_MM0_ir,row_MM1_ir)
xou_loc_ors = row_param_construct(row_MM0_or,row_MM1_or)

xou_opt_irs = row_param_construct(row_MM0_ir,row_MM0_ir)
xou_opt_ors = row_param_construct(row_MM0_or,row_MM0_or)

xou_rad_offsets = asarray(xou_loc_irs) - asarray(xou_opt_irs)

xou_widths = row_param_construct(row_MM0_w,row_MM1_w)
xou_cangles = repeat(mm_cangles,2)
xou_zoffsets = zeros(N_xous)
xou_tilts = -arctan(xou_rad_offsets/F)#/4

####################################################################
####################################################################
## Grating placement equations.
#dgrat = 2.00e-4 # The periodicity of the gratings (200 nm) in mm.
#grat_dims = array([27,26])
#
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
#
## Computing the zposition of the gratings based on the Rowland Torus prescription.
#'''
#Grating Rowland Torus Design:
#xhat and zhat are in the plane of the Rowland circle, and xhat is the hinge of revolution.
#bbeta = blaze angle (determines the tilt of the torus).
#r = the radius of the Rowland Circle.
#R = the radius of the "Hinge" Rotation.
#zg = the focal distance between the telescope focus and the "prime" grating.
#theta = the angle about the Rowland circle, as defined by the righthand rotation angle about yhat from the xhat axis
#phi = the angle about the "hinge", 
#'''
#
#bbeta = 1.9*pi/180
#zg = 11866.6
#
#x0,y0,z0 = zg*tan(2*bbeta)/2,0,0
#r = zg/2/cos(2*bbeta)
#R = zg/2
#
#def get_z_front_surface_toroid(x,y):
#    return sqrt((R + sqrt(r**2 - (x - x0)**2))**2 - (y**2))
#
#def get_torus_pangles(x,y,z):
#    '''
#    Returns theta, phi for a given position/
#    '''
#    try:
#        phi = arctan2(-y,z)
#    except ZeroDivisionError:
#        if x - x0 > 0:
#            phi = 0.0
#        else:
#            phi = pi
#    if z < R:
#        theta = arccos((x + x0)/r)
#    else:
#        theta = 2*pi - arccos((x + x0)/r)
#    return theta,phi
#
#def get_toroid_pvecs(x,y,z):
#    theta,phi = get_torus_pangles(x,y,z)
#    vtheta = array([-r*sin(theta),r*cos(theta)*sin(phi),-r*cos(theta)*cos(phi)])
#    vphi = array([0,-(R - r*sin(theta))*cos(phi),-(R - r*sin(theta))*sin(phi)])
#    vtheta,vphi = vtheta/linalg.norm(vtheta),vphi/linalg.norm(vphi)
#    return vtheta,vphi
#
#def get_toroid_n(x,y,z):
#    vtheta,vphi = get_toroid_pvecs(x,y,z)
#    return cross(vphi,vtheta)
#
#def plot_toroid():
#    # Generate torus mesh
#    angle1 = linspace(0, 2*pi, 32)
#    angle2 = linspace(0, -1.*pi/180, 32)
#    theta, phi = meshgrid(angle1, angle2)
#    
#    X = r*cos(theta) - x0
#    Y = -(R - r*sin(theta))*sin(phi) - y0
#    Z = (R - r*sin(theta))*cos(phi) - z0
#    
#    # Display the mesh
#    plt.ioff()
#    fig = plt.figure()
#    ax = fig.gca(projection = '3d')
#    ax.pbaspect = [1.,1.,1.]
#    ax.set_xlabel('Dispersion Direction (mm)')
#    ax.set_ylabel('Petal Height (mm)')
#    ax.set_zlabel('Optical Axis (mm)')
#    ax.plot_surface(X, Y, Z, color = 'w', rstride = 1, cstride = 1)
#    plt.show()
#
#zgrats = asarray([get_z_front_surface_toroid(xgrats[i],ygrats[i]) for i in range(len(xgrats))])
#ngrats = asarray([array([xgrats[i] - x0,ygrats[i] - y0,zgrats[i] - z0])/linalg.norm(array([xgrats[i] - x0,ygrats[i] - y0,zgrats[i] - z0])) for i in range(len(xgrats))])
##tgrats = asarray([array([1,0,0]) for i in range(len(xgrats))])
#tgrats = asarray([array([cos(bbeta),0,sin(bbeta)]) for i in range(len(xgrats))])
#pgrats = asarray([cross(ngrats[i],tgrats[i]) for i in range(len(xgrats))])
#
#grat_num_by_xou = []
#grat_count = 0
#
#for i in range(N_xous):
#    if row_number[i] < 5:
#        temp = [grat_count,grat_count + 1]
#    else:
#        temp = [grat_count,grat_count + 1,grat_count + 2]
#    grat_count = grat_count + len(temp)
#    grat_num_by_xou.append(temp)

#
#ngrat = ngrats[0]
#zhat = array([0,0,1])
#cvec,ctheta = cross(zhat,ngrat),dot(zhat,ngrat)
#skew_mat = tran.skew(cvec)
#I = array([[1,0,0],[0,1,0],[0,0,1]])
#
#R = I + skew_mat + dot(skew_mat,skew_mat)*1/(1 + ctheta)
#a1,a2,a3 = tran.tr.euler_from_matrix(R)

## Checking the position of the gratings.
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#
#fig = plt.figure(figsize = (8,8))
#ax1 = fig.add_subplot(111,aspect = 'equal')
#
#for i in range(12):
#    ax1.add_patch(patches.Rectangle((xgrats[i] - grat_xy[0]/2,ygrats[i] - grat_xy[1]/2),grat_xy[0],grat_xy[1],alpha = 0.3))



