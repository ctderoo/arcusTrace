from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pdb
import copy
import cPickle
import string
import time
import csv

import arcusTrace.SPOPetal as ArcSPO
import arcusTrace.CATPetal as ArcCAT
import arcusTrace.DetectorArray as ArcDet
import arcusTrace.arcusUtilities as ArcUtil
import arcusTrace.arcusPerformance as ArcPerf
import arcusTrace.arcusComponents as ArcComp
import arcusTrace.arcusRays as ArcRays

import arcusTrace.MCPerformPredict.MCPerformanceComputation as ArcMCComp

################################################
# Setting up the performance computation.
################################################

# Arcus configuration set-up.
opt_coords = [ArcComp.OC1_coords,ArcComp.OC2_coords,ArcComp.OC3_coords,ArcComp.OC4_coords]
uncoated_opt_chans = [ArcComp.ArcusChannel(i + 1,chan_coords = opt_coords[i]) for i in range(len(opt_coords))]
fpa = ArcComp.ArcusFPA()

# Monte Carlo calculation set-up.
N = int(2.5*10**4)
low_eV_space,high_eV_space = 10.,100.
lowhigh_boundary = 1500
low_energies = arange(250,lowhigh_boundary + low_eV_space,low_eV_space)
high_energies = arange(lowhigh_boundary, 10000 + high_eV_space,high_eV_space)
energies = hstack((low_energies,high_energies))
wavelengths = (1240./energies)*10**-6

uncoated_fileend = '190215_ArcusPerf_Uncoated_N10^5'
csv_description = 'Effective Areas and Resolving Powers by Order for Arcus Configuration, 12/11/17'
uncoated_ea_title_description = '4 Channels (Traced Separately),\nUncoated Plates, 1 um SiO2 atop Si,\n Full Arcus Focal Plane, No Alignment/Jitter Errors'

################################################
# Performing the Monte Carlo computation.
################################################
times = []

times.append(time.time())
ArcMCComp.ArcusConfigPerfCalc(uncoated_opt_chans,fpa,wavelengths,N,fileend = uncoated_fileend,csv_description = csv_description,ea_title_description = uncoated_ea_title_description)
times.append(time.time())

[wavelength,orders,ArcusR,ArcusEA] = cPickle.load(open(os.getcwd() + '/ResultPickles/IndividualChannels/' + uncoated_fileend + '.pk1','rb'))