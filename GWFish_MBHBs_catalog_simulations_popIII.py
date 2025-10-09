#!/usr/bin/env python
# coding: utf-8

# suppress warning outputs for using lal in jupuyter notebook
import warnings 
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import GWFish.modules as gw
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import corner
import numpy as np
import pandas as pd
import json
import os
from astropy.cosmology import Planck18


# In[2]:


from catalog_MBHBs_popIII import N_drawn, generate_synthetic_MBHB_population 

#from catalog_MBHBs import N_expected


# In[3]:


N_obs = N_drawn

# for reproducibility for generator (it uses np.random.*)
np.random.seed(42626)

# --- draw population directly from function (note the time window) ---
t0 = 1187008882
T  = t0 + 31536000
pop = generate_synthetic_MBHB_population(
    N=N_obs,
    geoctime_i=t0,
    geoctime_f=T,    
    alpha1=1.5, alpha2=2.5,
    m_min=5e2, m_break=1e3, m_max=5e3, #m_min=5e5, m_break=1e6, m_max=5e6
    q_min=0.5, q_max=1.0, #q_min=0.5, q_max=1.0
    spin_alpha=30, spin_beta=3,
    z_min=0, z_max=19.0, grid_size=1000
)

# Now ns comes directly from N_drawn
print(f"Generated {N_obs} events")

# --- unpack values from the generator ---
z     = np.asarray(pop["z"])
m1    = np.asarray(pop["m1"])
m2    = np.asarray(pop["m2"])
q     = np.asarray(pop["q"])
a_1  = np.asarray(pop["a_1"])
a_2  = np.asarray(pop["a_2"])
tgeo  = np.asarray(pop["geoctime"])

# sanity check
assert len(z) == N_obs == len(m1) == len(m2) == len(q) == len(a_1) == len(a_2) == len(tgeo)

# --- compute luminosity distance from z (Mpc) ---
dL = Planck18.luminosity_distance(z).value

# --- Angles uniformly distributed ---
theta_jn = np.random.uniform(0, np.pi, N_obs)
ra       = np.random.uniform(0, 2 * np.pi, N_obs)
dec      = np.random.uniform(-np.pi/2, np.pi/2, N_obs)
psi      = np.random.uniform(0, np.pi, N_obs)
phase    = np.random.uniform(0, 2 * np.pi, N_obs)
tilt_1   = np.random.uniform(0, np.pi, N_obs)
tilt_2   = np.random.uniform(0, np.pi, N_obs)

# --- build DataFrame for GWFish ---
parameters = pd.DataFrame.from_dict({
    "mass_1": m1,
    "mass_2": m2,
    "q": q,
    "redshift": z,
    "luminosity_distance": dL,
    "theta_jn": theta_jn,
    "ra": ra,
    "dec": dec,
    "psi": psi,
    "phase": phase,
    "geocent_time": tgeo,  
    "a_1": a_1,
    "a_2": a_2,
    "tilt_1": tilt_1,
    "tilt_2": tilt_2
})
parameters


# In[4]:


df = pd.DataFrame(parameters)
df.to_csv("mbhb_popIII_catalog.tsv", sep="\t", index=False)


# In[5]:


# We choose a waveform approximant suitable for BNS analysis
# In this case we are taking into account tidal polarizability effects
waveform_model = 'IMRPhenomPv2'
f_ref = 1e-4


# In[6]:


# Choose the detector onto which you want to project the signal
detector = 'LISA'

# The following function outputs the signal projected onto the chosen detector
signal, _ = gw.utilities.get_fd_signal(parameters, detector, waveform_model, f_ref) # waveform_model and f_ref are passed together
frequency = gw.detection.Detector(detector).frequencyvector[:, 0]


# In[7]:


# add the detector's sensitivity curve and plot the characteristic strain
psd_data = gw.utilities.get_detector_psd(detector)


# In[8]:


# Plot the time before the merger as a function of the frequency
_, t_of_f = gw.utilities.get_fd_signal(parameters, detector, waveform_model, f_ref)


# In[9]:


convert_from_seconds_to_hours = 3600


# ## Calculate SNR

# In[10]:


# The networks are the combinations of detectors that will be used for the analysis
# The detection_SNR is the minimum SNR for a detection:
#   --> The first entry specifies the minimum SNR for a detection in a single detector
#   --> The second entry specifies the minimum network SNR for a detection
detectors = ['LISA']
network = gw.detection.Network(detector_ids = detectors, detection_SNR = (0., 0.))
snr = gw.utilities.get_snr(parameters, network, waveform_model, f_ref)


# In[11]:


df = pd.DataFrame(snr)
df.to_csv("mbhb_popIII_catalog_snr.tsv", sep="\t", index=False)


# ## Calculate $1\sigma$ Errors
# For a more realistic analysis we can include the **duty cycle** of the detectors using `use_duty_cycle = True`

# In[13]:


# The fisher parameters are the parameters that will be used to calculate the Fisher matrix
# and on which we will calculate the errors

fisher_parameters = ['mass_1', 'mass_2', 'luminosity_distance', 'theta_jn', 'ra', 'dec',
                     'psi', 'phase', 'geocent_time', 'a_1', 'a_2', 'tilt_1', 'tilt_2']

#fisher_parameters = ['mass_1', 'mass_2', 'luminosity_distance', 'theta_jn', 'ra', 'dec',
#                     'psi', 'phase', 'geocent_time', 'a_1', 'a_2']

#fisher_parameters = ['mass_1', 'mass_2', 'luminosity_distance', 'dec','ra', #deleted theta_jn because can be the prior can be used
#                      'geocent_time', 'a_1', 'a_2'] #deleted psi, and phase because priors can be used


# In[ ]:


detected, network_snr, parameter_errors, sky_localization = gw.fishermatrix.compute_network_errors(
        network = gw.detection.Network(detector_ids = ['LISA'], detection_SNR = (0., 0.)),
        parameter_values = parameters,
        fisher_parameters=fisher_parameters, 
        waveform_model = waveform_model,
        f_ref = 1e-4,
        eps=1e-5,
        eps_mass=1e-5,
        )   
        # use_duty_cycle = False, # default is False anyway
save_matrices = True, # default is False anyway, put True if you want Fisher and covariance matrices in the output
save_matrices_path = '/home/2809904g/popIII', # default is None anyway,
                                     # otherwise specify the folder
                                     # where to save the Fisher and
                                     # corresponding covariance matrices
    
    


# In[ ]:


# Choose percentile factor of sky localization and pass from rad2 to deg2
percentile = 90.
sky_localization_90cl = sky_localization * gw.fishermatrix.sky_localization_percentile_factor(percentile)
#sky_localization_90cl


# In[ ]:


# One can create a dictionary with the parameter errors, the order is the same as the one given in fisher_parameters
parameter_errors_dict = {}
for i, parameter in enumerate(fisher_parameters):
    parameter_errors_dict['err_' + parameter] = np.squeeze(parameter_errors)[i]

#print('The parameter errors of the event are ')
parameter_errors_dict


# In[ ]:


data_folder = '/home/2809904g/popIII' 
network = gw.detection.Network(detector_ids = ['LISA'], detection_SNR = (0., 0.))
gw.fishermatrix.analyze_and_save_to_txt(network = network,
                                        parameter_values  = parameters,
                                        fisher_parameters = fisher_parameters, 
                                        sub_network_ids_list = [[0]],
                                        population_name = f'MBHB_catalog_popIII_withfisher_tilts',
                                        waveform_model = waveform_model,
                                        f_ref = 1e-4,
                                        save_path = data_folder,
                                        save_matrices = True,
                                        eps=1e-5,
                                        eps_mass=1e-5
                                        #decimal_output_format='%.6E'
                                        )


# In[ ]:




