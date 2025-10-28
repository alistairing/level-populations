### Import libraries ###

import numpy as np
import streamlit as st
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatter, LogFormatterSciNotation


colours = {1 : 'navy', 2 : 'royalblue', 3 : 'skyblue', 4 : 'darkorange', 5: 'red', 6: 'darkred', 7: 'darkgreen', 8: 'purple'}

###################################
#### DEFINE FUNCTIONS #############
###################################
# Function to create a new dictionary where each element is 1 divided by each element of the original dictionary
def get_times_from_rates(rates_dict):
    new_dict = {}
    for key, value in rates_dict.items():
        try:
            new_dict[key] = (1 / value)
        except TypeError:
            new_dict[key] = f"Cannot divide by non-numeric value: {value}"
        except ZeroDivisionError:
            new_dict[key] = "Cannot divide by zero"
    return new_dict

def convert_rates_to_scinote(rates_dict):
    new_dict = {}
    for key, value in rates_dict.items():
        try:
            new_dict[key] = f"{value:.2e}"
        except TypeError:
            new_dict[key] = f"Cannot divide by non-numeric value: {value}"
        except ZeroDivisionError:
            new_dict[key] = "Cannot divide by zero"
    return new_dict

def reset_mw_params(params):
    params_mw_off = {
          'w0xy' : w,
          'w0xz' : w, 
          'w0yz' : w,
          'w1xy' : w,
          'w1xz' : w, 
          'w1yz' : w,
         }
    return params.update(params_mw_off)
   
def calculate_excitation_rate(spot_diameter=400e-4, laser_power=0.9, wavelength=405e-9):
    '''
    Calculate the excitation rate based on the spot diameter and laser power.
    
    Args:
    - spot_diameter: Diameter of the laser spot in meters (default is 400 micrometers)
    - laser_power: Power of the laser in mW (default is 0.9 mW)
    - wavelength: Wavelength of the laser in meters (default is 405 nm)
    
    Returns:
    - Excitation rate in s^-1
    '''
    area = np.pi * (spot_diameter / 2) ** 2  # Area in square meters
    intensity = laser_power / area  # Power per unit area
    photon_flux = intensity/(3e8*6.63e-34 / wavelength)  # Photon flux in photons per square meter per second
    absorption_cross_section = 2.7e-16 # cm(^-2)
    excitation_rate = photon_flux*1e-4*absorption_cross_section

    return excitation_rate   

def odes(P, t, params):
    '''
     System of differential rate equations.

    Args:
    - P: Populations
    - t: time
    - params: dictionary of params (rate constants)

    Returns:
    - [dS0dt,  dT0xdt, dT0ydt, dT0zdt, dS1dt, dT1xdt, dT1ydt, dT1zdt]
    ''' 
    
    # populations
    S0, T0x, T0y, T0z,  S1, T1x, T1y, T1z = P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7]

    #kx, ky, kz = params['kx'], params['ky'], params['kz']
    P0x, P0y, P0z = params['P0x'], params['P0y'], params['P0z'] # ISC from T1 to S1
    Q0x, Q0y, Q0z = params['Q0x'], params['Q0y'], params['Q0z'] # ISC from S0 to T0
    w0xy, w0xz, w0yz = params['w0xy'], params['w0xz'], params['w0yz']
    w1xy, w1xz, w1yz = params['w1xy'], params['w1xz'], params['w1yz']
    k01, k10_s, k10_t = params['k01'], params['k10_s'], params['k10_t']
    
    # rate equations
    dS0dt = -(k01 + Q0x + Q0y + Q0z)*S0 + k10_s*S1 
    dT0xdt = k10_t*T1x + Q0x*S0 -(k01 + w0xy + w0xz)*T0x + w0xy*T0y + w0xz*T0z
    dT0ydt = k10_t*T1y + Q0y*S0 -(k01 + w0xy + w0yz)*T0y + w0xy*T0x + w0yz*T0z
    dT0zdt = k10_t*T1z + Q0z*S0 -(k01 + w0xz + w0yz)*T0z + w0xz*T0x + w0yz*T0y
    dS1dt = k01*S0 - k10_s*S1 + P0x*T1x + P0y*T1y + P0z*T1z 
    dT1xdt = k01*T0x  -(k10_t + w1xy + w1xz + P0x)*T1x + w1xy*T1y + w1xz*T1z
    dT1ydt = k01*T0y -(k10_t + w1xy + w1yz + P0y)*T1y + w1xy*T1x + w1yz*T1z 
    dT1zdt = k01*T0z - (k10_t + w1xz + w1yz + P0z)*T1z + w1xz*T1x + w1yz*T1y 

    return [dS0dt, dT0xdt, dT0ydt, dT0zdt, dS1dt, dT1xdt, dT1ydt, dT1zdt]

def odes_uphill(P, t, params):
    '''
     System of differential rate equations.

    Args:
    - P: Populations
    - t: time
    - params: dictionary of params (rate constants)

    Returns:
    - [dS0dt,  dT0xdt, dT0ydt, dT0zdt, dS1dt, dT1xdt, dT1ydt, dT1zdt]
    ''' 
    
    # populations
    S0, T0x, T0y, T0z,  S1, T1x, T1y, T1z = P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7]

    #kx, ky, kz = params['kx'], params['ky'], params['kz']
    P0x, P0y, P0z = params['P0x'], params['P0y'], params['P0z'] # ISC from T1 to S1
    Q0x, Q0y, Q0z = params['Q0x'], params['Q0y'], params['Q0z'] # ISC from S0 to T0
    R0x, R0y, R0z = params['R0x'], params['R0y'], params['R0z'] # ISC from T0 to S0
    w0xy, w0xz, w0yz = params['w0xy'], params['w0xz'], params['w0yz']
    w1xy, w1xz, w1yz = params['w1xy'], params['w1xz'], params['w1yz']
    k01, k10_s, k10_t = params['k01'], params['k10_s'], params['k10_t']
    
    # rate equations
    dS0dt = -(k01 + Q0x + Q0y + Q0z)*S0 + k10_s*S1 + R0x*T0x + R0y*T0y + R0z*T0z
    dT0xdt = k10_t*T1x + Q0x*S0 -(k01 + w0xy + w0xz + R0x)*T0x + w0xy*T0y + w0xz*T0z
    dT0ydt = k10_t*T1y + Q0y*S0 -(k01 + w0xy + w0yz + R0y)*T0y + w0xy*T0x + w0yz*T0z
    dT0zdt = k10_t*T1z + Q0z*S0 -(k01 + w0xz + w0yz + R0z)*T0z + w0xz*T0x + w0yz*T0y
    dS1dt = k01*S0 - k10_s*S1 + P0x*T1x + P0y*T1y + P0z*T1z 
    dT1xdt = k01*T0x  -(k10_t + w1xy + w1xz + P0x)*T1x + w1xy*T1y + w1xz*T1z
    dT1ydt = k01*T0y -(k10_t + w1xy + w1yz + P0y)*T1y + w1xy*T1x + w1yz*T1z 
    dT1zdt = k01*T0z - (k10_t + w1xz + w1yz + P0z)*T1z + w1xz*T1x + w1yz*T1y 

    return [dS0dt, dT0xdt, dT0ydt, dT0zdt, dS1dt, dT1xdt, dT1ydt, dT1zdt]

# Function to calculate contrast based on laser intensity and Rabi frequency
def calculate_contrast(rabi_frequency, laser_power=0.9, spot_diameter=400e-4, wavelength=405e-9, verbose=False, ):
    """
    Calculate the contrast based on laser intensity and Rabi frequency.
    
    Args:
    - laser_intensity: Intensity of the laser in arbitrary units.
    - rabi_frequency: Rabi frequency in Hz.
    
    Returns:
    - ground state singlet contrast (GSSC) value.
    - ground state triplet contrast (GSTC) value.
    - total ground state contrast value (GSC).
    - excited state singlet contrast (ESSC) value.
    - excited state triplet contrast (ESTC) value.
    - total excited state contrast value (ESC).
    - ratio of excited state contrast to ground state contrast.
    """
    k01 = calculate_excitation_rate(wavelength=wavelength, spot_diameter=spot_diameter, laser_power=laser_power)

    reset_mw_params(params)
    params.update({'k01' : k01})
    P = odeint(odes_uphill, P0, t, args=(params,)) # populations(t)

    params_mw = {'w0xz' : rabi_frequency + w, 'w1xz' : w}
    params.update(params_mw)
    P_mw = odeint(odes_uphill, P[-1], t, args=(params,)) # populations(t)
    
    ##  contrast cal.      ##
    S1_cntrl, S1_MW = P[:, 4][-1], P_mw[:, 4][-1]
    T1_cntrl, T1_MW = (P[:, 5]+P[:, 6]+P[:, 7])[-1], (P_mw[:, 5]+P_mw[:, 6]+P_mw[:, 7])[-1]
    contrast_singlet = GSSC = (S1_MW - S1_cntrl)/S1_cntrl
    contrast_triplet = GSTC = (T1_MW - T1_cntrl)/T1_cntrl
    contrast = GSC = (S1_MW - S1_cntrl + T1_MW - T1_cntrl)/(S1_cntrl+T1_cntrl)

    ## excited state ODMR contrast cal. ##
    params_mw = {'w1xz' : rabi_frequency + w, 'w0xz' : w}
    params.update(params_mw)
    P_mw = odeint(odes_uphill, P[-1], t, args=(params,)) # populations(t)

    S1_cntrl, S1_MW = P[:, 4][-1], P_mw[:, 4][-1]
    T1_cntrl, T1_MW = (P[:, 5]+P[:, 6]+P[:, 7])[-1], (P_mw[:, 5]+P_mw[:, 6]+P_mw[:, 7])[-1]
    contrast_singlet = ESSC = (S1_MW - S1_cntrl)/S1_cntrl
    contrast_triplet = ESTC = (T1_MW - T1_cntrl)/T1_cntrl
    contrast = ESC = (S1_MW - S1_cntrl + T1_MW - T1_cntrl)/(S1_cntrl+T1_cntrl)

    if verbose:
        print(f"\n### Ground state ODMR contrast ###\n")
        print(f"singlet contrast = {GSSC:.3e}")
        print(f"triplet contrast = {GSTC:.3e}")
        print(f"total contrast = {GSC:.3e}")
        print(f"\n### Excited state ODMR contrast ###\n")
        print(f"singlet contrast = {ESSC:.3e}")
        print(f"triplet contrast = {ESTC:.3e}")
        print(f"total contrast = {ESC:.3e}")
        print(f"\n### Ratio of Ground state contrast : Excited state contrast ###\n")
        print(f"ESC = {(ESC):.3e}")
        print(f"GSC = {(GSC):.3e}")
        print(f"Ratio = {GSC/ESC:.3e}")

    return [GSSC, GSTC, GSC, ESSC, ESTC, ESC, GSC/ESC]


###################################
### DEFINE TIMESCALES & RATES #####
###################################

T_1 = 24e-6 # Room Temp, Naitik EPR data
#T_1 = 301e-6 # 100K, Wasielewski paper 
tau_f_triplet = 9.4e-9 # from paper SI
k10_t = (1/tau_f_triplet)
tau_f_singlet = 170e-9 # from paper SI0
k10_s = (1/tau_f_singlet)
tau_S0_T0 = 95e-6 #Wasielewski paper - no spin-selectivity between S0 and T0 sublevels
k01 =  39.4*1600 #photons absorbed per second

triplet_pl_fraction = 0.98 # from SI fig S.VII.3
k_ISC = k10_t*((1/triplet_pl_fraction) - 1)
ss_factor = 100 #spin selectivity factor
k_ISC_z = k_ISC/ss_factor

P0x, P0y, P0z = k_ISC, k_ISC, k_ISC_z
Q0x = Q0y = Q0z = 1/tau_S0_T0

w = 1/T_1

params = {'P0x' : P0x, 
          'P0y' : P0y, 
          'P0z' : P0z,
          'Q0x' : Q0x, 
          'Q0y' : Q0y, 
          'Q0z' : Q0z,
          'R0x' : Q0x, 
          'R0y' : Q0y, 
          'R0z' : Q0z,  
          'k01' : k01,
          'k10_t' : k10_t,
          'k10_s' : k10_s,
          'w0xy' : w,
          'w0xz' : w, 
          'w0yz' : w,
          'w1xy' : w,
          'w1xz' : w, 
          'w1yz' : w,
         }
P0 = [0.25, 0.25, 0.25, 0.25, 0.00, 0.00, 0.00, 0.00] # S0, Tx, Ty, Tz, S1, T1x, T1y, T1z initial populations: everything starts in ground state
n_points = 6000
t = np.logspace(-10, -2, n_points)

###################################
### SET SLIDERS FOR STREAMLIT #####
###################################
#st.title("Contrast simulation for LW88 diradical")
st.sidebar.subheader("Adjust parameters to see effect on contrast")
st.sidebar.markdown("Note: all rates are in s<sup>-1</sup> unless otherwise stated.", unsafe_allow_html=True)

rabi_frequency = st.sidebar.number_input("rabi frequency (kHZ)", min_value=10.0, max_value=1000.0, value=100.0, step=1.0) * 1e3
k01 = st.sidebar.number_input(r'$k_{01}$ value', value = 40.0*1600, step=1.0, format="%0.2e")

P0x = st.sidebar.number_input(r'$P_x$ value', min_value=1e5, max_value=1e8, value=2.17e6, step=10000.0, format="%0.2e")

P0z = st.sidebar.number_input(r'$P_z$ value', min_value=1e3, max_value=1e6, value=2.17e4, step=1000.0, format="%0.2e") 

Q0x = st.sidebar.number_input(r'$Q_x$ value', min_value=1e2, max_value=1e6, value=1.05e4, step=100.0, format="%0.2e")

k10_t = st.sidebar.number_input(r'${k_{10}}^t$ value', min_value=1e7, max_value=1e9, value=1.06e8, step=1000000.0, format="%0.2e")

k10_s = st.sidebar.number_input(r'${k_{10}}^s$ value', min_value=1e6, max_value=1e8, value=5.88e6, step=100000.0, format="%0.2e")

w = st.sidebar.number_input(r"$w$ value", min_value=1e3, max_value=1e6, value=4.17e4, step=100.0, format="%0.2e")


### UPDATE PARAMS DICTIONARY WITH SLIDER VALUES ###

params.update({'P0x' : P0x, 
          'P0y' : P0x, 
          'P0z' : P0z,
          'Q0x' : Q0x, 
          'Q0y' : Q0x, 
          'Q0z' : Q0x,
          'R0x' : Q0x, 
          'R0y' : Q0x, 
          'R0z' : Q0x,  
          'k01' : k01,
          'k10_t' : k10_t,
          'k10_s' : k10_s,
          'w0xy' : w,
          'w0xz' : w, 
          'w0yz' : w,
          'w1xy' : w,
          'w1xz' : w, 
          'w1yz' : w,
         })

###################################
###### CALCULATE POPULATIONS ######
###################################
#reset_mw_params(params)
P = odeint(odes_uphill, P0, t, args=(params,)) # populations(t)


########################
##        Plot        ##
########################
fig, ax = plt.subplots(1,1, figsize=(8, 5))
line1, = ax.plot(t, P[:, 0], label='$S0$')
line2, = ax.plot(t, P[:, 1], label='$T0x$')
line3, = ax.plot(t, P[:, 2], label='$T0y$')
line4, = ax.plot(t, P[:, 3], label='$T0z$')
line5, = ax.plot(t, P[:, 4], label='$S1$')
line6, = ax.plot(t, P[:, 5], label='$T1x$')
line7, = ax.plot(t, P[:, 6], label='$T1y$')
line8, = ax.plot(t, P[:, 7], label='$T1z$')

# First legend: populations
pop_lines = [line1, line2, line3, line4, line5, line6, line7, line8]
pop_labels = [r'$S_0$', r'$T_{0,x}$', r'$T_{0,y}$', r'$T_{0,z}$', r'$S_1$', r'$T_{1,x}$', r'$T_{1,y}$', r'$T_{1,z}$']
legend1 = ax.legend(pop_lines, pop_labels, frameon=False, loc='center right', title='Populations')

# Add the first legend manually to the axes
ax.add_artist(legend1)

# Second legend: vertical lines (timescales)
vlines = [
    ax.axvline(x=1/k01, color='orange', linestyle='--', label=r'$k_{01}$', alpha=0.3),
    ax.axvline(x=1/P0x, color='b', linestyle='--', label='Px', alpha=0.3),
    ax.axvline(x=1/P0z, color='c', linestyle='--', label='Pz', alpha=0.3),
    ax.axvline(x=1/Q0x, color='g', linestyle='--', label='Qx', alpha=0.3),
    ax.axvline(x=1/k10_t, color='m', linestyle='--', label=r'k10_t', alpha=0.3),
    ax.axvline(x=1/k10_s, color='y', linestyle='--', label=r'k10_s', alpha=0.3),
    ax.axvline(x=1/w, color='k', linestyle='--', label='w', alpha=0.3),
]
vline_labels = [r'$k_{01}$', r'$P_x$', r'$P_z$', r'$Q_x$', r'${k_{10}}^t$', r'${k_{10}}^s$', 'w']
legend2 = ax.legend(vlines, vline_labels, frameon=False, loc='center left', title='Timescales')

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel('time (s)')
ax.set_xscale('log') 
ax.set_yscale('linear') 
ax.set_ylabel('Population')
#ax.legend(frameon=False)
plt.tight_layout()
fig.suptitle('Population dynamics of LW88 diradical system', fontsize=16)
fig.subplots_adjust(top=0.88)   
st.pyplot(fig)

############################
## Plot total populations ##
############################
# fig, ax = plt.subplots(1,1, figsize=(8, 5))

# ax.plot(t, P[:, 0], label='$S_0$')
# ax.plot(t, P[:, 4], label='$S_1$')
# ax.plot(t, P[:, 5]+P[:, 6]+P[:, 7], label='$T_{1tot}$',linestyle='--')
# ax.plot(t, P[:, 1]+P[:, 2]+P[:, 3], label='$T_{0tot}$',linestyle='-')

# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax.set_xlabel('time (s)')
# ax.set_xscale('log') 
# ax.set_yscale('linear') 
# ax.set_ylabel('Population')
# ax.legend(frameon=False)
# plt.tight_layout()
# fig.suptitle('Population dynamics of LW88 diradical system', fontsize=16)
# fig.subplots_adjust(top=0.88)   
# st.pyplot(fig)

###################################
###### Laser OFF populations ######
###################################

reset_mw_params(params)
laser_off_params = params.copy()
laser_off_params.update({'k01' : 0})  # Set k01 to 0 to simulate laser off
P_off = odeint(odes_uphill, P[-1], t, args=(laser_off_params,)) # populations(t)

########################
##        Plot        ##
########################
fig, ax = plt.subplots(1,1, figsize=(8, 5))
line1, = ax.plot(t, P_off[:, 0], label='$S0$')
line2, = ax.plot(t, P_off[:, 1], label='$T0x$')
line3, = ax.plot(t, P_off[:, 2], label='$T0y$')
line4, = ax.plot(t, P_off[:, 3], label='$T0z$')
line5, = ax.plot(t, P_off[:, 4], label='$S1$')
line6, = ax.plot(t, P_off[:, 5], label='$T1x$')
line7, = ax.plot(t, P_off[:, 6], label='$T1y$')
line8, = ax.plot(t, P_off[:, 7], label='$T1z$')

pop_lines = [line1, line2, line3, line4, line5, line6, line7, line8]
pop_labels = [r'$S_0$', r'$T_{0,x}$', r'$T_{0,y}$', r'$T_{0,z}$', r'$S_1$', r'$T_{1,x}$', r'$T_{1,y}$', r'$T_{1,z}$']
legend1 = ax.legend(pop_lines, pop_labels, frameon=False, loc='center right', title='Populations')

vlines = [
    ax.axvline(x=1/Q0x, color='g', linestyle='--', label= r'$Q_x$', alpha=0.3),
    ax.axvline(x=1/k10_s, color='y', linestyle='--', label= r'${k_{10}}^s$', alpha=0.3),
    ax.axvline(x=1/w, color='k', linestyle='--', label='w', alpha=0.3),
]
vline_labels = [ r'$Q_x$', r'${k_{10}}^s$', 'w']

legend2 = ax.legend(vlines, vline_labels, frameon=False, loc='center left', title='Timescales')

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel('time (s)')
ax.set_xscale('log') 
ax.set_yscale('linear') 
ax.set_ylabel('Population')
ax.add_artist(legend1)
plt.tight_layout()
fig.suptitle('Population dynamics of LW88 diradical system', fontsize=16)
fig.subplots_adjust(top=0.88)   
st.pyplot(fig)

###################################
### PREPARE DATAFRAME  ############
###################################
param_df = pd.DataFrame([convert_rates_to_scinote(params)])
st.dataframe(param_df)


###################################
### RATES IMAGE #####
###################################
st.image("fig/level-rates-diagram-uphill.svg", caption="Rate diagram for the diradical system showing the various rates and timescales involved in the model.")

