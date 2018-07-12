import glob
import os
import numpy as np
import pyreflectivity.mirror as mirror
import pdb

reflib_dir = r'C:\Users\Casey\Software\ReflectLib'

# Making the needed reflectivity libraries for the Arcus uncoated baseline case (1 nm SiO2 atop Si) and 
# the Arcus coated baseline case (6 nm SiC atop 10 nm Ir atop Si). 

e_step = float(2.5)
max_step = int(500)
angles = np.arange(0.1,2.0 + 0.05,0.05)
energies = np.arange(100.,12000. + e_step,e_step)
roughness = np.arange(0.0,1.0 + 0.1,0.1)

def make_that_data_stack(mirror_object,angle,energies = energies):
    e_min = energies[0] 
    e_max = energies[-1]
    e_step = energies[1] - energies[0]

    e0 = e_min
    while e0 < e_max:
        e1 = e0 + (max_step - 1)*e_step
        e1 = min([e_max + e_step,e1])
        num_steps = (e1 - e0)/e_step + 1
        if e0 == e_min:
            data = mirror_object.calculate_energy_scan(e0,e1,num_steps,angle)[:,:2]
        else:
            data = np.vstack((data,mirror_object.calculate_energy_scan(e0,e1,num_steps,angle)[:,:2]))
        e0 = e1
    data = np.hstack((data,np.ones((np.shape(data)[0],1))*angle))
    return data

#########################################################
# First, building the Arcus plate model with no coating (uncoated baseline case).
Arcus_NoCoat = mirror.SingleLayerMirror()
Arcus_NoCoat.layer_chemical_formula = 'Si02'        # Layer Material
Arcus_NoCoat.layer_density_gm_cm3 = -1              # Layer Density
Arcus_NoCoat.layer_thickness_nm = 1.0               # Layer Thickness
Arcus_NoCoat.layer_roughness_nm = 0.0               # Layer Roughness
Arcus_NoCoat.sub_chemical_formula = 'Si'            # Substrate Material
Arcus_NoCoat.sub_density_gm_cm3 = -1                # Substrate Density
Arcus_NoCoat.sub_roughness_nm = 0.0                 # Substrate Roughness (nm)
Arcus_NoCoat.polarization = 1                       # Polarization of Incoming Light

## We've got a tri-nested loop -- one with roughness, one with angle, one with energy.
#for rough in roughness:
#    fn = reflib_dir + '\SL_LSi02_LThick1p0_LRough' + "{:2.1f}".format(rough).replace('.','p') + '_SSi_SRough0p0_Pol1_CXRO.npy'
#    Arcus_NoCoat.layer_roughness_nm = rough
#    #save_data = np.zeros((len(energies),len(angles),3))
#    for i in range(len(angles)):
#        print('Working on Layer Roughness: ' + "{:2.1f}".format(rough) + ' nm, Angle: ' + "{:3.2f}".format(angles[i]) + ' deg...' )
#        temp = make_that_data_stack(Arcus_NoCoat,angles[i])
#        if i == 0:
#            save_data = np.zeros((np.shape(temp)[0],len(angles),3))
#        save_data[:,i,:] = temp
#    np.save(fn,save_data)

Arcus_Coat = mirror.BiLayerMirror()
Arcus_Coat.top_chemical_formula = 'SiC'             # Top Layer Material
Arcus_Coat.top_density_gm_cm = -1                   # Top Layer Density
Arcus_Coat.top_thickness_nm  = 6.0                  # Top Layer Thickness (nm)
Arcus_Coat.top_roughness_nm= 0.0                    # Top Layer Roughness (nm)
Arcus_Coat.bot_chemical_formula = 'Ir'              # Bottom Layer Material
Arcus_Coat.bot_density_gm_cm3 = -1                  # Bottom Layer Density
Arcus_Coat.bot_thickness_nm = 10                    # Bottom Layer Thickness (nm)
Arcus_Coat.bot_roughness_nm = 0.0                   # Bottom Layer Roughness (nm)
Arcus_Coat.sub_chemical_formula = 'Si'              # Substrate Material
Arcus_Coat.sub_density_gm_cm3 = -1                  # Substrate Density
Arcus_Coat.sub_roughness_nm = 0.0                   # Substrate Roughness (nm)
Arcus_Coat.polarization = 1                         # Polarization State

# We've got a tri-nested loop -- one with roughness, one with angle, one with energy.
for rough in roughness:
    fn = reflib_dir + '\BiL_TLSiC_TLThick6p0_TLRough' + "{:2.1f}".format(rough).replace('.','p') + 'BLIr_BLThick10p0_BLRough0p0_SSi_SRough0p0_Pol1_CXRO.npy'
    Arcus_Coat.layer_roughness_nm = rough
    #save_data = np.zeros((len(energies),len(angles),3))
    for i in range(len(angles)):
        print('Working on Layer Roughness: ' + "{:2.1f}".format(rough) + ' nm, Angle: ' + "{:3.2f}".format(angles[i]) + ' deg...' )
        temp = make_that_data_stack(Arcus_NoCoat,angles[i])
        if i == 0:
            save_data = np.zeros((np.shape(temp)[0],len(angles),3))
        save_data[:,i,:] = temp
    np.save(fn,save_data)