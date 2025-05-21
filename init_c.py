# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:18:03 2022

@author: Miroslav Hanzelka - mirekhanzelka@gmail.com

22-08-29
Created from older code (Hanzelka et al. 2021, 10.1029/2021JA029624)
Moved user input values to a csv file.
Moved physical constants to a separate py file.

22-08-30
Rewritten precalculation of 1D magnetic fields.

23-10-12
Refactored with named dictionaries for better readability.
"""

import os
import sys
import csv
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import init.const as cs
import model.model as md
import model.coord as co
from scipy import optimize
import copy


def initialize(input_file='input/input.csv'):
    """Initialize the simulation with parameters from the input file.
    Args:
        input_file (str): Path to input CSV file. Defaults to 'input/input.csv'.
        
    Raises:
        SystemExit: If input file doesn't exist or contains invalid data.
    """
    if not os.path.exists(input_file):
        print('INIT: Input file '+input_file+' not found. Aborting.')
        sys.exit()        
    
    # Initialize dictionaries
    dicts, dicts_units = init_dictionaries()
    tp_init, tp_trac, psd, wav, dens, mag, oth = dicts
    tp_init_units, tp_trac_units, psd_units, wav_units, dens_units, mag_units, oth_units = dicts_units

    # Read and process the input CSV file
    read_input_csv(input_file, dicts, dicts_units, tp_init)
        
    # Perform initial setup and error checking
    tp_init, wav, oth = setup_initial_parameters(tp_init, wav, oth, mag)
        
    # Normalize input values to internal units
    normalize_input_value(tp_init, mag, dicts, dicts_units)
    
    # Configure delta function PSD parameters
    psd = init_delta_psd(psd, tp_init)
    
    # Recalculate cold/hot plasma density to plasma frequency
    dens, psd = calculate_plasma_frequencies(dens, psd, mag)
    
    # Calculate time steps
    tp_trac = calculate_time_steps(tp_trac)
    
    
    # Share starting L-shell of particles with magfield and dens model
    mag['Lfield'] = tp_init['Lfield']
    dens['Lfield'] = tp_init['Lfield']
    
    # Initialize positions along field line if whole field sampling is requested
    tp_init, h_points_fieldline, lambda_points_fieldline = init_field_line_sampling(tp_init, tp_trac, wav, mag)

    # Use h_points and lambda_points from field line sampling if they were set
    if h_points_fieldline is not None:
        h_points = h_points_fieldline
        lambda_points = lambda_points_fieldline
    
    # Initialize magnetic field
    h_points, b_field, b_field_der, lambda_points = init_magnetic_field(mag, tp_init, tp_trac)
    
    # Saved, but never loaded
    mag['h_b_db'] = np.array([h_points, b_field, b_field_der])
    mag['path'] = os.path.dirname(input_file)+'/hs_bfield_bfieldder.npy'
    np.save(mag['path'], mag['h_b_db'])
    
    # Precalculate cold electron plasma frequency along field line
    dens, ompe = init_plasma_frequency(dens, mag, h_points, lambda_points, input_file)
    
    dens['h_ompe'] = np.array([h_points, ompe])
    dens['path'] = os.path.dirname(input_file)+'/hs_ompe.npy'
    np.save(dens['path'], dens['h_ompe'])
    
    # Amplitude threshold to speed up calculation outside the field
    # (later, larger time steps are used when the amplitude is weak)
    # Absolute threshold amplitude (if user-given value is negative)
    if wav.get('ampl_thresh', 0.) < 0.:
        wav['Btot_thresh'] = -wav['ampl_thresh']
    
    # Initialize wave model based on type
    if 'external' in wav.get('wav_mod', ''):
        wav = init_external_model(wav, tp_trac, input_file)
    else:
        wav = init_phase_based_model(wav, tp_trac, b_field, h_points, ompe, dens, mag, input_file)
    
    # Initialize reflective boundary conditions if needed
    tp_trac = init_reflective_boundary(tp_trac, tp_init, mag)
    
    # Save all simulation data
    save_final_state(input_file, dicts, oth)

def init_dictionaries():
    """Initialize and return dictionaries for storing input parameters."""
    tp_init, tp_init_units = {}, {} # Initial position
    tp_trac, tp_trac_units = {}, {} # Particle tracing
    psd, psd_units = {}, {} # Phase space density
    wav, wav_units = {}, {} # Wave
    dens, dens_units = {}, {} # Density
    mag, mag_units = {}, {} # Magnetic field
    oth, oth_units = {}, {} # Other
    
    dicts = [tp_init, tp_trac, psd, wav, dens, mag, oth]
    dicts_units = [tp_init_units, tp_trac_units, psd_units,
                   wav_units, dens_units, mag_units, oth_units]
    return dicts, dicts_units

def read_input_csv(input_file, dicts, dicts_units, tp_init):
    """Read and process data from the input CSV file.

    This function reads a CSV file containing simulation parameters and populates
    the provided dictionaries with the values.

    Args:
        input_file (str): Path to the input CSV file
        dicts (list): List of dictionaries to store different parameter categories:
            - tp_init: Test particle initial conditions
            - tp_trac: Particle tracing parameters
            - psd: Phase space density model parameters
            - wav: Wave model parameters
            - dens: Density model parameters
            - mag: Magnetic field model parameters
            - oth: Other parameters
        dicts_units (list): List of dictionaries to store units for each parameter category
        tp_init (dict): Dictionary containing test particle initialization parameters
        
    Returns:
        bool: True if input file was processed successfully, False if there were errors
        
    Raises:
        SystemExit: If the input file contains invalid data or incorrect format  
    """
    valid_input = True
    
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        block_count = 0
        block = ''
        block_col = 0
        for line_count, row in enumerate(csv_reader):
            if (line_count == 1) and (row[block_col] != 'BLOCK'):
                valid_input = False
                break
            if line_count > 1:
                if row[block_col] != '':
                    if row[block_col] == 'END':
                        print('INIT: Input file reading successfully finished.')
                        break
                    block = dicts[block_count]
                    block_units = dicts_units[block_count]
                    block_count += 1
                valid_input = process_row(row, block, block_units, tp_init)
                if not valid_input:
                    break

    # Raise an error if the input file included invalid characters             
    if not(valid_input):
        print('INIT: Initialization failed at line '+str(line_count+1)+'.\n'\
              'Check the input and run the code again.')
        sys.exit()
        
    return valid_input

def process_row(row, block, block_units, tp_init):
    """Process a single row from the CSV file and route it to the appropriate handler.
    Args:
        row (list): A row from the CSV file containing parameter values
        block (dict): Dictionary to store the processed parameter value
        block_units (dict): Dictionary to store the parameter's unit
        tp_init (dict): Dictionary containing test particle initialization parameters
    
    Returns:
        bool: True if processing was successful, False if:
            - Input type doesn't match the declared type
            - Input format is invalid
            - Processing fails in the handler function
    """
    val1_col, type_col = 5, 3
    if ((row[val1_col])[0]).isdigit() or ((row[val1_col])[0]) == '-':
        if (row[type_col] != 'float') and (row[type_col] != 'int'):
            return False
        return process_numeric_value(row, block, block_units, tp_init)
    elif ((row[val1_col])[0]).isalpha() or ((row[val1_col])[0]) == '_' or ((row[val1_col])[0]) == '.':
        if row[type_col] != 'str':
            return False
        return process_string_value(row, block, block_units)
    return False

def process_numeric_value(row, block, block_units, tp_init):
    """Process a numeric value from the CSV row and store it in the appropriate dictionary.
    Args:
        row (list): A row from the CSV file containing parameter values
        block (dict): Dictionary to store the processed parameter value
        block_units (dict): Dictionary to store the parameter's unit
        tp_init (dict): Dictionary containing test particle initialization parameters
            with keys:
            - espace: Energy space type ('log' or 'lin')
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    val1_col, val2_col, val_num_col, var_col, type_col = 5, 6, 7, 4, 3
    if ' ' in row[val1_col] or 'duct' in row[var_col]:
        split = (row[val1_col]).split(' ')
        block[row[var_col]] = (np.array(split)).astype(eval(row[type_col]))
    elif ((row[val_num_col]).isdigit()):
        if not skip_row(row, tp_init):
            min_val = float(row[val1_col])
            max_val = float(row[val2_col])
            n_val = int(row[val_num_col])
            if row[var_col] == 'energs0' and tp_init.get('espace', '') == 'log':
                block[row[var_col]] = bins(min_val, max_val, n_val, 'mid', 'log')
            else:
                block[row[var_col]] = bins(min_val, max_val, n_val, 'mid', 'lin')
    else:
        block[row[var_col]] = float(row[val1_col]) if row[type_col] == 'float' else int(row[val1_col])
    block_units[row[var_col]] = row[2]  # unit_col
    return True

def process_string_value(row, block, block_units):
    """Process a string value from the CSV rowand store it in the appropriate dictionary.
    Args:
        row (list): A row from the CSV file containing parameter values
        block (dict): Dictionary to store the processed parameter value
        block_units (dict): Dictionary to store the parameter's unit
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    val1_col, var_col, type_col = 5, 4, 3
    if row[val1_col] == '_':
        block[row[var_col]] = ''
    elif ' ' in row[val1_col] or 'var' in row[var_col]:
        split = (row[val1_col]).split(' ')
        block[row[var_col]] = split
    else:
        block[row[var_col]] = row[val1_col]
    block_units[row[var_col]] = row[2]  # unit_col
    return True

def skip_row(row, tp_init): 
    """Determine if a row should be skipped based on the coordinate system configuration.
    
    Args:
        row (list): A row from the CSV file containing parameter values
        tp_init (dict): Dictionary containing test particle initialization parameters
            with keys:
            - ps_coord: Phase space coordinate system ('velocity' or 'energpitch')
            - rs_coord: Spatial coordinate system ('fielddist' or 'lat')
    
    Returns:
        bool: True if the row should be skipped, False otherwise 
    """
    var_col = 4
    if ('velocity' in tp_init.get('ps_coord', '')):
        if (row[var_col] == 'energs0' or row[var_col] == 'pitchangs0'):
            return True
    elif ('energpitch' in tp_init.get('ps_coord', '')):
        if (row[var_col] == 'vparas0' or row[var_col] == 'vperps0'):
            return True
    if (tp_init.get('rs_coord', '') == 'fielddist'):
        if row[var_col] == 'lats0':
            return True
    elif (tp_init.get('rs_coord', '') == 'lat'):
        if row[var_col] == 'hs0':
            return True
    return False

def normalize_input_value(tp_init, mag, dicts, dicts_units):
    """Normalize input values to appropriate internal units for calculation
    Args:
        tp_init (dict): Dictionary containing test particle initialization parameters
            with keys:
            - ps_coord: Phase space coordinate system ('velocity' or 'energpitch')
            - eunit: Energy unit ('kev' or other)
            - qx: Particle charge
            - mx: Particle mass
            - Lfield: L-shell value
        mag (dict): Dictionary containing magnetic field parameters
            with keys:
            - B0eq: Equatorial magnetic field strength
        dicts (list): List of dictionaries containing simulation parameters
        dicts_units (list): List of dictionaries containing units for each parameter
    """
    if 'energpitch' in tp_init.get('ps_coord', ''):
        if tp_init.get('eunit', '') == 'kev':
            tp_init['energs0'] *= 1e3*np.abs(cs.qe*tp_init['qx'])/(cs.me*tp_init['mx']*cs.c0**2)
            
    # Calculate electron cyclotron frequency at equator
    # omce0 is needed for normalization
    mag['omce0'] = mag['B0eq']*cs.qe/cs.me/tp_init['Lfield']**3
    km_to_norm = mag['omce0']/cs.c0*1e3
    
    # Apply unit conversions to all dictionaries
    for valdict, unitdict in zip(dicts, dicts_units):
        for mykey in unitdict:
            # Convert seconds to normalized time using cyclotron frequency
            if unitdict[mykey] == 'sec' and mykey in valdict:
                valdict[mykey] *= mag['omce0']
            # Convert km to normalized length
            if unitdict[mykey] == 'km' and mykey in valdict:
                valdict[mykey] *= km_to_norm
            # Convert degrees to radians
            if unitdict[mykey] == 'deg' and mykey in valdict:
                valdict[mykey] *= 1/cs.radeg

def setup_initial_parameters(tp_init, wav, oth, mag):
    """Perform initial setup and validate input parameters.
    
    Args:
        tp_init (dict): Dictionary containing test particle initialization parameters
            with keys:
            - rs_coord: Spatial coordinate system ('fielddist' or 'lat')
        wav (dict): Dictionary containing wave model parameters
            with keys:
            - t_smooth: Wave field fade-in/out times
        oth (dict): Dictionary containing other parameters
            with keys:
            - out_folder: Path to output directory
        mag (dict): Dictionary containing magnetic field parameters
            with keys:
            - mag_dim: Magnetic field dimensionality (1 or 3)
    
    Returns:
        tuple: (tp_init, wav, oth) with validated and updated parameters
    """
    # Check for some common errors the user can do when editing the input file
    if tp_init.get('rs_coord', '') == 'fielddist':
        if mag.get('mag_dim', 0) > 1:
            print('INIT: 3D magnetic dipole must be initialized by initial latitude, \n'\
                  'not field-aligned distance. Change VAR rs_coord to >lat<.')
            tp_init['rs_coord'] = 'lat'
            
    # If only one value for fade-in/out of the wave-field was given
    # assume symmetric behavior at the temporal beginning and end of the packet
    if wav.get('t_smooth') is not None and np.isscalar(wav['t_smooth']):
        smooth_sym = wav['t_smooth']
        wav['t_smooth'] = np.array([smooth_sym, smooth_sym])
    
    # Create output folder if it does not exist
    if not os.path.exists(oth.get('out_folder', '')):
        os.makedirs(oth['out_folder'])
        
    return tp_init, wav, oth

def bins(min_val, max_val, n_bin, bin_type='mid', bin_scale='lin'):
    """ Create evenly spaced bin values with flexible sampling options.
        
    Args:
        min_val (float): Minimum value of the range
        max_val (float): Maximum value of the range
        n_bin (int): Number of bins to create
        bin_type (str, optional): Type of binning to use. Defaults to 'mid'.
            - 'mid': Values are placed at bin centers
            - 'edge': Values are placed at bin edges
        bin_scale (str, optional): Type of spacing to use. Defaults to 'lin'.
            - 'lin': Linear spacing using numpy.linspace
            - 'log': Logarithmic spacing using numpy.geomspace
    
    Returns:
        numpy.ndarray: Array of n_bin evenly spaced values
    """
    if bin_scale == 'lin':
        if bin_type == 'mid':
            half_bin = (max_val - min_val)/n_bin/2
            bin_vals = np.linspace(min_val + half_bin, max_val - half_bin, n_bin)
        else:
            bin_vals = np.linspace(min_val, max_val, n_bin)
    else:
        if bin_type == 'mid':
            half_bin = (max_val/min_val)**(1/n_bin/2)
            bin_vals = np.geomspace(min_val*half_bin, max_val/half_bin, n_bin)
        else:
            bin_vals = np.geomspace(min_val, max_val, n_bin)
    return bin_vals

def init_delta_psd(psd, tp_init):
    """Configure delta function PSD parameters based on energy and pitch angle bins.
    Args:
        psd (dict): Dictionary containing phase space density parameters
            with keys:
            - psd_mod: String indicating the PSD model type
            - delta_ek_ind: Index of the energy bin for delta function
            - delta_pa_ind: Index of the pitch angle bin for delta function
        tp_init (dict): Dictionary containing test particle initialization parameters
            with keys:
            - espace: Energy space type ('log' or 'lin')
            - energs0: Array of energy values
            - pitchangs0: Array of pitch angle values
    
    Returns:
        dict: Updated PSD dictionary with new keys:
            - ek_min: Minimum energy value for the delta function bin
            - ek_max: Maximum energy value for the delta function bin
            - pa_min: Minimum pitch angle value for the delta function bin
            - pa_max: Maximum pitch angle value for the delta function bin
    """
    # If the initial PSD is a delta function, get the bin edges
    if 'psd_mod' in psd and 'delta' in psd['psd_mod']:
        if tp_init.get('espace', '') == 'log':
            # For logarithmic energy spacing
            half_ek_mult = np.sqrt(tp_init['energs0'][1]/tp_init['energs0'][0])
            psd['ek_min'] = tp_init['energs0'][psd['delta_ek_ind']]/half_ek_mult
            psd['ek_max'] = tp_init['energs0'][psd['delta_ek_ind']]*half_ek_mult
        else:
            # For linear energy spacing
            half_ek_add = (tp_init['energs0'][1] - tp_init['energs0'][0])/2
            psd['ek_min'] = tp_init['energs0'][psd['delta_ek_ind']] - half_ek_add
            psd['ek_max'] = tp_init['energs0'][psd['delta_ek_ind']] + half_ek_add
        
        # Always considers the equatorial pitch angle grid
        half_pa_add = (tp_init['pitchangs0'][1] - tp_init['pitchangs0'][0])/2
        psd['pa_min'] = tp_init['pitchangs0'][psd['delta_pa_ind']] - half_pa_add
        psd['pa_max'] = tp_init['pitchangs0'][psd['delta_pa_ind']] + half_pa_add
    
    return psd

def calculate_plasma_frequencies(dens, psd, mag):
    """Convert density values to plasma frequencies in normalized units.
    Args:
        dens (dict): Dictionary containing cold plasma density parameters
            with keys:
            - ompe0: Cold plasma frequency (negative values indicate number density in m^-3)
        psd (dict): Dictionary containing hot plasma parameters
            with keys:
            - omphe0: Hot plasma frequency (negative values indicate number density in m^-3)
        mag (dict): Dictionary containing magnetic field parameters
            with keys:
            - omce0: Electron cyclotron frequency (used for normalization)
    
    Returns:
        tuple: Updated dictionaries (dens, psd) with converted plasma frequencies
    """
    if 'ompe0' in dens and dens['ompe0'] < 0.:
        dens['ompe0'] = np.sqrt(-dens['ompe0']*cs.qe**2/(cs.eps0*cs.me))/mag['omce0']
    if 'omphe0' in psd and np.any(psd['omphe0'] < 0.):
        psd['omphe0'] = np.sqrt(-psd['omphe0']*cs.qe**2/(cs.eps0*cs.me))/mag['omce0']
    
    return dens, psd

def calculate_time_steps(tp_trac):
    """Calculate number of steps and step size for non-adaptive simulation.
    Args:
        tp_trac (dict): Dictionary containing test particle tracing parameters
            with keys:
            - t_max: Maximum simulation time
            - n_step_period: Number of steps per wave period (2π)
    
    Returns:
        dict: Updated tp_trac dictionary with new keys:
            - n_step: Total number of time steps in the simulation
            - tstep: Size of each time step
    """
    tp_trac['n_step'] = int(tp_trac['t_max']*tp_trac['n_step_period']/2/np.pi)
    tp_trac['tstep'] = tp_trac['t_max']/tp_trac['n_step']
    
    return tp_trac

def init_field_line_sampling(tp_init, tp_trac, wav, mag):
    """Initialize particle positions along the field line when whole-field sampling is requested.
    
    Args:
        tp_init (dict): Dictionary containing test particle initialization parameters
            with keys:
            - if_hfull: Flag for whole-field sampling (1 = enabled)
            - rs_coord: Spatial coordinate system ('lat' or 'fielddist')
            - lats0: Initial latitudes (if rs_coord is 'lat')
            - hs0: Initial field-aligned distances (if rs_coord is 'fielddist')
            - Lfield: L-shell value
        tp_trac (dict): Dictionary containing test particle tracing parameters
            with keys:
            - loss_alt: Altitude at which particles are considered lost
        wav (dict): Dictionary containing wave model parameters
            with keys:
            - wav_mod: Wave model type (checked for 'sym' in name)
        mag (dict): Dictionary containing magnetic field parameters
            with keys:
            - mag_mod: Magnetic field model type
            - R_E: Earth radius
    
    Returns:
        tuple: (tp_init, h_points, lambda_points)
            - tp_init: Updated initialization parameters
            - h_points: Field-aligned distances along the field line
            - lambda_points: Latitudes along the field line
    """
    # If sampling along the whole fieldline was not requested, return early
    if tp_init.get('if_hfull', 0) != 1:
        return tp_init, None, None
        
    if mag.get('mag_mod', '') == 'dipole':  # Options like delta_ekpa, bima, etc. can be added
        n_lambda = 2
        lambda_points = co.init_lat(tp_init['Lfield'], n_lambda=n_lambda,
                                   alt_max=tp_trac['loss_alt'], R_E=mag['R_E'])
        
        if tp_init.get('rs_coord', '') == 'lat':
            if 'sym' in wav.get('wav_mod', ''):
                if tp_init['lats0'][-1] >= 0.:
                    lambda_points[0] = 0.
                else:
                    lambda_points[1] = 0.
            h_points = co.lat_fielddist(lambda_points, tp_init['Lfield'], mag['R_E'])
            
            n_val = len(tp_init['lats0'])
            tp_init['lats0'] = bins(lambda_points[0], lambda_points[-1], n_val, 'mid', 'lin')
            
        if tp_init.get('rs_coord', '') == 'fielddist':
            if 'sym' in wav.get('wav_mod', ''):
                if tp_init['hs0'][-1] >= 0.:
                    lambda_points[0] = 0.
                else:
                    lambda_points[1] = 0.
            h_points = co.lat_fielddist(lambda_points, tp_init['Lfield'], mag['R_E'])
            n_val = len(tp_init['hs0'])
            tp_init['hs0'] = bins(h_points[0], h_points[-1], n_val, 'mid', 'lin')
    else:
        print('''INIT: Stretching of initial position along the whole field line
              (if_hfull == 1) is not implemented for non-dipolar magnetic fields. Aborting.''')
        sys.exit()
        
    return tp_init, h_points, lambda_points

def init_external_model(wav, tp_trac, input_file):
    """Initialize external wave model by loading and processing field data files.
    
    Args:
        wav (dict): Dictionary containing wave model parameters with keys:
            - wav_mod: Wave model type (must contain 'external')
            - wav_path: Path to wave data files
            - wav_tag: Tag for wave data files
            - wav_ext: File extension for wave data
            - ampl_thresh: Amplitude threshold (must be negative for external fields)
        tp_trac (dict): Dictionary containing test particle tracking parameters
        input_file (str): Path to the input file
    
    Returns:
        dict: Updated wave dictionary with new keys:
            - bs_ext: External magnetic field data
            - es_ext: External electric field data
            - ts_ext: Time points for external field
            - hss_ext: Field-aligned distance points
            - x2ss_ext: Cross-field distance points
            - interpfun_bsx/y/z: Interpolators for magnetic field components
            - interpfun_esx/y/z: Interpolators for electric field components
            - split_input_path: Path for split input files
    
    Raises:
        SystemExit: If relative amplitude threshold is used (must be negative)
    """
    if 'external' in wav.get('wav_mod', ''):
        # Load field data from files
        wav['bs_ext'] = np.load(wav['wav_path']+'/'+wav['wav_tag']+'_bs2d.'+wav['wav_ext'])
        wav['es_ext'] = np.load(wav['wav_path']+'/'+wav['wav_tag']+'_es2d.'+wav['wav_ext'])
        wav['ts_ext'] = np.load(wav['wav_path']+'/snap_ts_conv.'+wav['wav_ext'])
        wav['ts_ext'] = wav['ts_ext'] - wav['ts_ext'][0]  # Start from zero
        wav['hss_ext'] = np.load(wav['wav_path']+'/hss.'+wav['wav_ext'])
        wav['x2ss_ext'] = np.load(wav['wav_path']+'/x2ss.'+wav['wav_ext'])
        hs_ext = wav['hss_ext'][:, 0]
        x2s_ext = wav['x2ss_ext'][0, :]
        
        # Set up interpolation
        intp_met = 'linear'
        b_err = False
        fill_val = 0.
        interpfun_keys = ('interpfun_bsx', 'interpfun_bsy', 'interpfun_bsz',
                       'interpfun_esx', 'interpfun_esy', 'interpfun_esz')
        fields_comp = (wav['bs_ext'][0], wav['bs_ext'][1], wav['bs_ext'][2],
                       wav['es_ext'][0], wav['es_ext'][1], wav['es_ext'][2])
        
        for interpfun_key, field_comp in zip(interpfun_keys, fields_comp):
            wav[interpfun_key] = RegularGridInterpolator((wav['ts_ext'],
                                                 hs_ext, x2s_ext),
                                                field_comp,
                                                method=intp_met,
                                                bounds_error=b_err,
                                                fill_value=fill_val)
            
        # Check amplitude threshold settings
        if wav.get('ampl_thresh', 0.) > 0.:
            print('''INIT: Relative amplitude threshold is not implemented for external
                  field. Please change ampl_thresh to negative (treated as absolute value). Aborting.''')
            sys.exit()
            
        wav['split_input_path'] = os.path.splitext(input_file)[0]
        
    return wav

def init_phase_based_model(wav, tp_trac, b_field, h_points, ompe, dens, mag, input_file):
    """Initialize wave model based on amplitude and phase or field integration.
    
        Args:
        wav (dict): Dictionary containing wave model parameters with keys:
            - wav_mod: Wave model type ('amplphase' or other)
            - wav_path: Path to wave data files
            - wav_tag: Tag for wave data files
            - wav_ext: File extension for wave data
            - ampl0: Amplitude scaling factor
            - ampl_thresh: Amplitude threshold
            - disp_mode: Wave dispersion mode ('whistler_1d', 'emic', etc.)
            - om0: Base frequency
            - om0_delta: Frequency spread for delta spectrum
            - n_delta: Number of delta components
            - pac_hmin: Minimum height for phase integration
            - pac_mid: Midpoint height for phase integration
        tp_trac (dict): Dictionary containing test particle tracking parameters with keys:
            - if_adapt: Flag for adaptive time stepping
            - t_dir: Time direction
            - t_ini: Initial time
            - t_max: Maximum simulation time
        b_field (numpy.ndarray): Magnetic field values along field line
        h_points (numpy.ndarray): Field-aligned distance points
        ompe (numpy.ndarray): Plasma frequency values
        dens (dict): Dictionary containing density parameters
        mag (dict): Dictionary containing magnetic field parameters
        input_file (str): Path to input file
    
    Returns:
        dict: Updated wave dictionary with new keys:
            For amplitude-phase models:
            - Btot_envs, B_envs, B_psis: Magnetic field components
            - E_envs, E_psis: Electric field components
            - ts_ext, hs_ext: Time and space points
            - kx_ext: Wavenumber values
            - rep_inds: Repetition indices
            - Btot_thresh: Amplitude threshold
            - split_input_path: Path for split files
            For phase integration models:
            - h_k: Field-aligned distance and phase information
            - path: Path to saved phase data
            - Btot_thresh: Amplitude threshold
    """
    # Handles both 'amplphase' wave models and other types
    if 'amplphase' in wav.get('wav_mod', ''):
        # Load amplitude and phase data
        Btot_envs = np.load(wav['wav_path']+'/Btot_envs_'+wav['wav_tag']+'.'+wav['wav_ext'])
        B_envs = np.load(wav['wav_path']+'/B_envs_'+wav['wav_tag']+'.'+wav['wav_ext'])
        B_psis = np.load(wav['wav_path']+'/B_psis_'+wav['wav_tag']+'.'+wav['wav_ext'])
        E_envs = np.load(wav['wav_path']+'/E_envs_'+wav['wav_tag']+'.'+wav['wav_ext'])
        E_psis = np.load(wav['wav_path']+'/E_psis_'+wav['wav_tag']+'.'+wav['wav_ext'])
        ts_ext = np.load(wav['wav_path']+'/ts_'+wav['wav_tag']+'.'+wav['wav_ext'])
        hs_ext = np.load(wav['wav_path']+'/hs_'+wav['wav_tag']+'.'+wav['wav_ext'])
        kx_ext = np.load(wav['wav_path']+'/kx_'+wav['wav_tag']+'.'+wav['wav_ext'])
        
        # Rescale amplitudes if needed
        if wav.get('ampl0', 0.) > 0.:
            Btot_max = np.nanmax(Btot_envs)
            Btot_envs *= wav['ampl0']/Btot_max
            B_envs *= wav['ampl0']/Btot_max
            E_envs *= wav['ampl0']/Btot_max

        # Calculate repetition indices
        if not(np.isclose(wav.get('rep_per', 0.), 0.)):
            if tp_trac.get('t_dir', 0) < 0:
                t_0 = tp_trac['t_ini'] + tp_trac['t_dir']*tp_trac['t_max']
                t_1 = tp_trac['t_ini']
            else:
                t_0 = tp_trac['t_ini'] 
                t_1 = tp_trac['t_ini'] + tp_trac['t_dir']*tp_trac['t_max']
            
            rep_ind_min = -int(t_1//wav['rep_per'])
            rep_ind_max = int((wav['t_wav'] - t_0)//wav['rep_per'])
            wav['rep_inds'] = np.arange(rep_ind_min, rep_ind_max + 1)
            
            # Set default amplitude threshold if none specified
            if np.isclose(wav.get('ampl_thresh', 0.), 0.):
                wav['ampl_thresh'] = 0.001
        else:
            wav['rep_inds'] = np.array([0])

        # Set up field interpolation
        intp_met = 'linear'
        intp_met_tot = 'nearest'
        b_err = False
        fill_val = 0.
        
        # Configure interpolation based on wave type
        if 'whistler' in wav.get('disp_mode', '') or 'emic' in wav.get('disp_mode', ''):
            fields_comp = np.stack((B_envs[0], B_envs[1], B_envs[2],
                                    B_psis[0],
                                    E_envs[0], E_envs[1], E_envs[2],
                                    kx_ext), axis=-1)
                
            interpfun_EBkxs = RegularGridInterpolator((ts_ext, hs_ext),
                                                    fields_comp, method=intp_met,
                                                    bounds_error=b_err, fill_value=fill_val)
            
            interpfun_Btot_envs = RegularGridInterpolator((ts_ext, hs_ext),
                                                        Btot_envs, method=intp_met_tot,
                                                        bounds_error=b_err, fill_value=fill_val)
        else:
            fields_comp = np.stack((B_envs[0], B_envs[1], B_envs[2],
                                    B_psis[0], B_psis[1], B_psis[2],
                                    E_envs[0], E_envs[1], E_envs[2],
                                    E_psis[0], E_psis[1], E_psis[2],
                                    kx_ext), axis=-1)
                
            interpfun_EBkxs = RegularGridInterpolator((ts_ext, hs_ext),
                                                    fields_comp, method=intp_met,
                                                    bounds_error=b_err, fill_value=fill_val)
            
        # Set amplitude threshold
        if wav.get('ampl_thresh', 0.) > 0.:
            Btot_max = np.nanmax(Btot_envs)
            wav['Btot_thresh'] = wav['ampl_thresh']*Btot_max
        elif np.isclose(wav.get('ampl_thresh', 0.), 0.):
            wav['Btot_thresh'] = 0.
        
        # Set up file structure for data
        wav['split_input_path'] = os.path.splitext(input_file)[0]
        
        # Handle adaptive vs non-adaptive approach
        if tp_trac.get('if_adapt', 0) == 0:
            # Split large data into chunks for non-adaptive simulation
            elem_10mb = 1310720
            grid_EBkxs = interpfun_EBkxs.grid
            values_EBkxs = interpfun_EBkxs.values
                
            wav['n_wavfile'] = np.max([values_EBkxs.size//elem_10mb, 2])
            n_t_ext_chunk = len(ts_ext)//wav['n_wavfile']
            t_chunk_inds = np.arange(n_t_ext_chunk*wav['n_wavfile'], step=n_t_ext_chunk)
            t_chunk_inds = np.append(t_chunk_inds, len(ts_ext))
            
            # Save each chunk separately
            for forind, t_ext_ind in enumerate(t_chunk_inds[:-1]):
                interpfun_EBkxs_ind = copy.deepcopy(interpfun_EBkxs)
                interpfun_EBkxs_ind.grid = (grid_EBkxs[0][np.max([0, t_ext_ind-2]):t_chunk_inds[forind+1]],
                                           grid_EBkxs[1])
                interpfun_EBkxs_ind.values = values_EBkxs[np.max([0, t_ext_ind-2]):t_chunk_inds[forind+1], :, :]
                
                np.save(wav['split_input_path']+'_EBkxs_'+str(forind)+'.npy',
                       interpfun_EBkxs_ind)
                
                # Try saving total amplitudes if they exist
                try:
                    interpfun_Btot_envs_ind = copy.deepcopy(interpfun_Btot_envs)
                    interpfun_Btot_envs_ind.grid = interpfun_EBkxs_ind.grid
                    interpfun_Btot_envs_ind.values = interpfun_Btot_envs.values[np.max([0, t_ext_ind-2]):t_chunk_inds[forind+1], :]
                
                    np.save(wav['split_input_path']+'_Btot_envs_'+str(forind)+'.npy',
                           interpfun_Btot_envs_ind)
                except:
                    pass
        else:
            # For adaptive simulation, save single file
            np.save(wav['split_input_path']+'_EBkxs_00.npy', interpfun_EBkxs)
            
            try:
                np.save(wav['split_input_path']+'_Btot_envs_00.npy', interpfun_Btot_envs)
            except:
                pass
                
    else:
        # Handle other wave models (phase integration along field line)
        # Calculate frequencies
        omce = b_field*cs.qe/cs.me/mag['omce0']
        omps = md.omps_prepare(ompe, dens.get('np_rel0', 0), dens.get('nhe_rel0', 0), dens.get('no_rel0', 0))  
        omcs = md.omcs_prepare(omce)
        
        # Handle delta spectrum or standard wave
        if 'delta' in wav.get('wav_mod', ''):
            # Special handling for delta function spectrum
            if not(wav.get('disp_mode', '') == 'whistler_1d'):
                print('INIT: Delta sum spectrum is defined only for whistler_1d. Aborting.')
                sys.exit()
            
            wav['randpsi0_delta'] = np.random.random(wav['n_delta'])*2*np.pi
            
            # Calculate wavenumbers for all components
            kh = np.zeros((wav['n_delta'], len(omce)))
            for delta_ind in range(wav['n_delta']):
                delta_om = wav['om0'] + ((delta_ind + 0.5)/wav['n_delta'] - 0.5)*wav['om0_delta']
                oms = np.full(len(omce), wav['om0'] + delta_om)
                kh[delta_ind] = md.whist_k_para(oms, ompe, omce)
        else:
            # Standard single frequency calculation
            oms = np.full(np.shape(omce), wav['om0'])
            if '1d' in wav.get('disp_mode', ''):
                if 'whistler' in wav.get('disp_mode', ''):
                    kh = md.whist_k_para(oms, ompe, omce)
                elif 'emic' in wav.get('disp_mode', ''):
                    kh = md.emic_k_para(oms, omps, omcs)
            if '2d' in wav.get('disp_mode', ''):
                ths = np.full(np.shape(omce), wav['th0'])
                mus = md.refind(ths, oms, omcs, omps)[0]
                if 'whistler' in wav.get('disp_mode', ''):
                    kh = oms*np.cos(ths)*mus[..., 0]
                elif 'emic' in wav.get('disp_mode', ''):
                    kh = oms*np.cos(ths)*mus[..., 0]
        
        # Find starting point for phase integration
        if wav.get('pac_hmin', 0) > 0.:
            h_ind_wav = np.argmin(np.abs((wav['pac_mid'] - wav['pac_hmin']) - h_points))
        else:
            h_ind_wav = np.argmin(np.abs(h_points))
            
        # Integrate phase along field line
        if 'delta' in wav.get('wav_mod', ''):
            # Multi-frequency case
            k_phase = np.zeros((wav['n_delta'], len(h_points)))
            
            for delta_ind in range(wav['n_delta']):
                k_phase_1d = np.zeros(np.shape(h_points))
                k_phase_1d[h_ind_wav+1:] = np.cumsum((h_points[h_ind_wav+1:] -
                                                h_points[h_ind_wav:-1])*kh[delta_ind, h_ind_wav+1:])
                k_phase_1d[:h_ind_wav] = np.flip(np.cumsum(np.flip((h_points[:h_ind_wav] -
                                                h_points[1:h_ind_wav+1])*kh[delta_ind, :h_ind_wav])))
                k_phase[delta_ind] = k_phase_1d
        else:
            # Single-frequency case
            k_phase = np.zeros(np.shape(h_points))
            k_phase[h_ind_wav+1:] = np.cumsum((h_points[h_ind_wav+1:] -
                                           h_points[h_ind_wav:-1])*kh[h_ind_wav+1:])
            k_phase[:h_ind_wav] = np.flip(np.cumsum(np.flip((h_points[:h_ind_wav] -
                                           h_points[1:h_ind_wav+1])*kh[:h_ind_wav])))
        
        # Save phase information
        if 'delta' in wav.get('wav_mod', ''):
            wav['h_k'] = np.array([h_points, k_phase], dtype=object)
        else:
            wav['h_k'] = np.array([h_points, k_phase])
            
        wav['path'] = os.path.dirname(input_file)+'/hs_kphase.npy'
        np.save(wav['path'], wav['h_k'])
        
        # Set amplitude threshold
        if wav.get('ampl_thresh', 0.) > 0.:
            wav['Btot_thresh'] = wav['ampl_thresh']*wav['ampl0']
            
    return wav

def init_magnetic_field(mag, tp_init, tp_trac):
    """Initialize magnetic field model and calculate field strength along field line
    
    Args:
        mag (dict): Dictionary containing magnetic field parameters with keys:
            - mag_dim: Magnetic field dimensionality (1 or 3)
            - mag_mod: Magnetic field model type ('const', 'parab', or 'dipole')
            - R_E: Earth radius
            - a_parab: Parabolic coefficient (optional, auto-calculated if 0)
        tp_init (dict): Dictionary containing test particle initialization parameters with keys:
            - Lfield: L-shell value
            - lats0: Initial latitudes (for 3D models)
        tp_trac (dict): Dictionary containing test particle tracking parameters with keys:
            - h_min: Minimum field-aligned distance
            - h_max: Maximum field-aligned distance
    
    Returns:
        tuple: (h_points, b_field, b_field_der, lambda_points)
            - h_points: Field-aligned distance points
            - b_field: Magnetic field strength values
            - b_field_der: Magnetic field derivatives
            - lambda_points: Latitude points (None for 1D models except dipole)
    """

    lambda_points = None
    
    if mag.get('mag_dim', 0) == 1:
        if mag.get('mag_mod', '') == 'const':
            n_h = 10  # Any integer larger than 1
            h_points = np.linspace(tp_trac['h_min']-1., tp_trac['h_max']+1., n_h)
            b_field, b_field_der = md.bfield(h_points, mag, der=True)
            
        elif mag.get('mag_mod', '') == 'parab':
            if np.isclose(0., mag.get('a_parab', 0.)):
                mag['a_parab'] = 9./2./(tp_init['Lfield']*mag['R_E'])**2.
                
            n_h = 20000
            h_points = np.linspace(tp_trac['h_min']-1., tp_trac['h_max']+1., n_h)
            b_field, b_field_der = md.bfield(h_points, mag, der=True)
                
        elif mag.get('mag_mod', '') == 'dipole':
            n_lambda = 20000
            lambda_points = co.init_lat(tp_init['Lfield'], n_lambda=n_lambda)
            h_points, b_field, b_field_der = md.bfield(lambda_points, mag, der=True)
    else:
        if mag.get('mag_mod', '') == 'parab' or mag.get('mag_mod', '') == 'const':
            print('INIT: Parabolic or constant magnetic field approximation does not work with 3D '+
                  'magnetic field models.')
            sys.exit()
        
        # TODO this is useless for 3D field, later calculated analytically
        n_lambda = 20000
        lambda_points = co.init_lat(tp_init['Lfield'], n_lambda=n_lambda)
        _, b_field, _ = md.bfield(lambda_points, mag, der=True)
        h_points = np.nan*b_field
        b_field_der = np.nan*b_field
        
        # Initialization by latitude; MLT fixed
        rs0 = tp_init['Lfield']*mag['R_E']*np.column_stack((np.cos(tp_init['lats0']),
                                    np.zeros(len(tp_init['lats0'])), np.sin(tp_init['lats0'])))
        tp_init['hs0'] = co.sm_cart_fielddist(rs0, tp_init['Lfield'], mag['R_E'])
    
    return h_points, b_field, b_field_der, lambda_points

def init_reflective_boundary(tp_trac, tp_init, mag):
    """Initialize reflective boundary conditions for particle bounce calculations.
    
    Args:
        tp_trac (dict): Dictionary containing test particle tracking parameters with keys:
            - bound_cond: Boundary condition type ('reflected' or other)
            - loss_alt: Loss altitude
            - h_min: Minimum field-aligned distance
            - h_max: Maximum field-aligned distance
        tp_init (dict): Dictionary containing test particle initialization parameters with keys:
            - Lfield: L-shell value
        mag (dict): Dictionary containing magnetic field parameters with keys:
            - mag_mod: Magnetic field model type
            - mag_dim: Magnetic field dimensionality
            - R_E: Earth radius
    
    Returns:
        dict: Updated tp_trac dictionary with additional keys:
            - sinalph_lc: Loss cone angle sine
            - sinalph_loss: Loss cone angle sine at boundaries
            - sinalph_points: Array of sine of pitch angles for bounce calculations
            - bounce_integrs: Integration results for bounce period calculations

    Raises:
        SystemExit: If boundary conditions are not compatible with magnetic field model
    """
    # If reflective boundary is desired, precalculate bounce periods (dipole model)
    if tp_trac.get('bound_cond', '') != 'reflected':
        return tp_trac
        
    if mag.get('mag_mod', '') != 'dipole' or mag.get('mag_dim', 0) != 1:
        print('''INIT: Reflection at boundaries implemented only for
              a field-aligned propagation in a dipole field. Aborting.''')
        sys.exit()
    
    L_loss = tp_init['Lfield']/(1. + tp_trac['loss_alt']/mag['R_E'])
    tp_trac['sinalph_lc'] = 1./(4*L_loss**6 - 3*L_loss**5)**0.25
    
    n_sinalph = 1000
    # TODO temporarily decreased number of integration points
    n_lat_toint = 10000
    h_bounds = np.array([tp_trac['h_min'], tp_trac['h_max']])
    lat_bounds = co.fielddist_lat(h_bounds, tp_init['Lfield'], mag['R_E'], n_lambda=1000000)
    lat_points_hires = co.init_lat(tp_init['Lfield'], n_lambda=1000000)
    
    # Use the approximate latitude (lat_bounds) to get the B field exactly at those points
    # Explanation: the h(MLAT) relation cannot be inverted
    B_bounds = np.sqrt(1 + 3*np.sin(lat_bounds)**2)/np.cos(lat_bounds)**6
    tp_trac['sinalph_loss'] = np.sqrt(B_bounds*tp_trac['sinalph_lc']**2)
    sinalph_points_left = np.sin(np.linspace(np.arcsin(tp_trac['sinalph_loss'][0]), np.pi/2, n_sinalph))
    sinalph_points_right = np.sin(np.linspace(np.arcsin(tp_trac['sinalph_loss'][1]), np.pi/2, n_sinalph))
    tp_trac['sinalph_points'] = np.array([sinalph_points_left, sinalph_points_right])
    
    bisec_left0 = np.min(lat_points_hires)
    bisec_left1 = lat_bounds[0]
    bisec_right0 = lat_bounds[1]
    bisec_right1 = np.max(lat_points_hires)
    
    # Mirroring latitudes for each alph must be found numerically
    integrs = [np.zeros(n_sinalph), np.zeros(n_sinalph)]
    for ind in range(n_sinalph):
        sinalphs = tp_trac['sinalph_points'][:, ind]
        # Skip calculation for alpha = 90 deg
        if np.isclose(sinalphs[0], 1., rtol=1e-10) and np.isclose(sinalphs[1], 1., rtol=1e-10):
            integrs[0][ind] = 0.
            integrs[1][ind] = 0.
            continue
        
        B_mirrors = B_bounds/sinalphs**2

        f_left = lambda x: np.sqrt(1 + 3*np.sin(x)**2)/np.cos(x)**6 - B_mirrors[0]
        f_right = lambda x: np.sqrt(1 + 3*np.sin(x)**2)/np.cos(x)**6 - B_mirrors[1]
        lat_mirror_left = optimize.bisect(f_left, bisec_left0, bisec_left1, xtol=1e-12)
        lat_mirror_right = optimize.bisect(f_right, bisec_right0, bisec_right1, xtol=1e-12)
        lat_mirrors = [lat_mirror_left, lat_mirror_right]
        
        integrs[0][ind], _ = md.time_lat0_to_lat1(lat_bounds[0], lat_mirrors[0], n_lat_toint, tp_init['Lfield'],
                                              mag['R_E'], B_bounds[0], sinalphs[0], int_type='para')
        integrs[1][ind], _ = md.time_lat0_to_lat1(lat_bounds[1], lat_mirrors[1], n_lat_toint, tp_init['Lfield'],
                                              mag['R_E'], B_bounds[1], sinalphs[1], int_type='para')
    
    tp_trac['bounce_integrs'] = integrs
    
    return tp_trac

def init_plasma_frequency(dens, mag, h_points, lambda_points, input_file):
    """Calculate plasma frequency along field line based on density model.
    
    Args:
        dens: Dictionary with density parameters
        mag: Dictionary with magnetic field parameters
        h_points: Field-aligned distance points
        lambda_points: Latitude points
        input_file: Path to input file
        
    Returns:
        tuple: Updated density dictionary and plasma frequency array

    Raises:
        SystemExit: If density model is incompatible with magnetic field model
                  or if higher-dimensional density models are attempted
    """
    # Depends on the model type and dimension
    if 'dens_dim' in dens and dens['dens_dim'] == 1:
        # h_points are already defined by magnetic field
        if (dens.get('dens_mod', '') == 'denton') and (mag.get('mag_mod', '') == 'parab' or mag.get('mag_mod', '') == 'const'):
            print('INIT: Denton density model does not work with constant or parabolic '+
                  'magnetic field model. Aborting.')
            sys.exit()
        
        if dens.get('dens_mod', '') == 'denton':
            if dens.get('dent_a', 0.) < 0.:
                dens['dent_a'] = md.denton_alpha(mag['Lfield'], dens['ompe0'], mag['omce0'])
            
            ompe = md.dens(lambda_points, dens, mag)[1]  # [0] is h_points, redundant
        else:
            ompe = md.dens(h_points, dens, mag)
    else:
        print('INIT: Density models in higher dimensions are not yet implemented.')
        sys.exit()
    
    dens['h_ompe'] = np.array([h_points, ompe])
    dens['path'] = os.path.dirname(input_file)+'/hs_ompe.npy'
    np.save(dens['path'], dens['h_ompe'])
    
    return dens, ompe

def save_final_state(input_file, dicts, oth):
    """Save the final state of the simulation to files.
    
    Args:
        input_file: Path to input file
        dicts: List of simulation dictionaries
        oth: Other parameters dictionary
    
    Raises:
        SystemExit: If unable to save final state   
    """
    # Unpack dictionaries
    tp_init, tp_trac, psd, wav, dens, mag, oth = dicts
    
    # Save input into a dictionary
    ini_dict = {
        'tp_init': tp_init,
        'tp_trac': tp_trac,
        'psd': psd,
        'wav': wav,
        'dens': dens,
        'mag': mag,
        'oth': oth
    }
    np.save(os.path.splitext(input_file)[0]+'.npy', ini_dict)

    # Save all final variables for reference
    final_vars = {}
    for name, value in globals().items():
        if (not name.startswith('__') and 
            not isinstance(value, type(sys)) and  # Filter out modules
            not callable(value) and               # Filter out functions
            not isinstance(value, type)):         # Filter out classes
            try:
                # Test if the value can be pickled
                np.save('test.npy', value, allow_pickle=True)
                final_vars[name] = value
            except:
                continue
    
    # Save all final variables
    np.save(os.path.join(oth['out_folder'], 'final_state.npy'), final_vars, allow_pickle=True)
    
    print('INIT: Gamma-version Initialization complete.')