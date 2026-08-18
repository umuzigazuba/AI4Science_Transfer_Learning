import os
import csv

import pandas as pd
import numpy as np

from extinction import fitzpatrick99
import astropy.units as u
from astropy.cosmology import WMAP9, z_at_value
from astropy.coordinates import SkyCoord, Distance
from dustmaps.sfd import SFDQuery

import redback
import george

import rubin_sim.maf as maf

from event_window import EventWindowParams, detect_event_window_for_object

### Auxilary functions

#Effective wavelength for each band - sourced from SVO Filter Profile Service
EFFECTIVE_WAVELENGTHS = {"u": np.array([3641]), "g": np.array([4704]), "r": np.array([6155]), 
                         "i": np.array([7504]), "z": np.array([8695]), "y": np.array([10056])}

#LSST frequencies
BAND_FREQUENCIES = {"lsstu": redback.utils.bands_to_frequency(["lsstu"])[0], "lsstg": redback.utils.bands_to_frequency(["lsstg"])[0],
                    "lsstr": redback.utils.bands_to_frequency(["lsstr"])[0], "lssti": redback.utils.bands_to_frequency(["lssti"])[0],
                    "lsstz": redback.utils.bands_to_frequency(["lsstz"])[0], "lssty": redback.utils.bands_to_frequency(["lssty"])[0]}


window_parameters = EventWindowParams(smooth_points = 2, detect_thr = 5.0, window_thr = 4.5, min_points_above = 2, pad_days = 0.0)

def ab_to_uJy(magAB):
    flux_Jy = 10 ** (23 - (magAB + 48.6) / 2.5)
    flux_mJy = flux_Jy * 1000000
    return flux_mJy

#Defining function to apply extinction to the flux for a respective band
def flux_extinction(flux, ebv, eff_wl):
    A_lambda = fitzpatrick99(eff_wl, ebv * 3.1) #3.1 = Standard Milky Way value
    flux_ext = flux * 10 ** (-(A_lambda) / 2.5)
    return flux_ext, A_lambda

#Defining function to de-extinct a set of flux values
def flux_de_extinction(flux, ebv, eff_wl):
    A_lambda = fitzpatrick99(eff_wl, ebv * 3.1) #3.1 = Standard Milky Way value
    return flux * 10 ** ((A_lambda) / 2.5)

# Function to match LSST visit dates to the simulated flux values
def get_match(band_visits, gp_time, gp_flux, err_list, tol = 0.5):
    matched = {}

    # Ensure consistent rounding
    band_visits = np.round(band_visits, 4)
    gp_time = np.round(gp_time, 4)

    # Deduplicate visits while preserving errors
    band_visits, unique_idx = np.unique(band_visits, return_index = True)
    err_list = np.array(err_list)[unique_idx]

    for i, visit in enumerate(band_visits):
        # Find the GP point closest in time
        idx = np.argmin(np.abs(gp_time - visit))
        if np.abs(gp_time[idx] - visit) <= tol:
            key = float(np.round(gp_time[idx], 4))
            if key not in matched:
                matched[key] = (gp_flux[idx], err_list[i])

    # Sort keys and build arrays
    times = np.array(sorted(matched.keys()))
    flux = np.array([matched[float(np.round(t, 4))][0] for t in times])
    errs = np.array([matched[float(np.round(t, 4))][1] for t in times])

    return times, flux, errs

### Gaussian process
def fit_gp(object_lc_df, redback_object, unique_bands, nr_points = 1000):

    # scale guesses in redback"s scaled space
    y_scaled = object_lc_df["Flux"].values / np.nanmax(object_lc_df["Flux"].values)
    var_y = np.nanvar(y_scaled) if np.isfinite(y_scaled).any() else 1.0

    # time lengthscales (days)
    ls_t_long  = 50.0 # broad evolution; keeps u from going flat
    ls_t_short = 10.0 # allows shape differences near peak

    # log-frequency lengthscales (dex)
    ls_f_long  = 0.50 # correlates u..z reasonably
    ls_f_short = 0.12 # lets r/i/z differ; u is only partly tied via long component

    amp_long  = 0.8 * var_y
    amp_short = 0.12 * var_y

    kernel_long = george.kernels.ConstantKernel(np.log(amp_long), ndim = 2) \
                * george.kernels.ExpSquaredKernel([ls_f_long ** 2, ls_t_long ** 2],  ndim = 2) 

    kernel_short = george.kernels.ConstantKernel(np.log(amp_short), ndim = 2) \
                * george.kernels.Matern32Kernel([ls_f_short ** 2, ls_t_short ** 2], ndim = 2) 

    kernel = kernel_long + kernel_short
    out = redback_object.fit_gp(mean_model = None, kernel = kernel, use_frequency = True)

    time_gp = np.linspace(object_lc_df["Time (MJD)"].min(), object_lc_df["Time (MJD)"].max(), nr_points)
    flux_gp = {}

    for band_name in unique_bands:

        frequencies_gp = np.ones(len(time_gp)) * BAND_FREQUENCIES[band_name]
        x_gp = np.column_stack((frequencies_gp, time_gp))
        y_gp, _ = out.gp.predict(out.scaled_y, x_gp, return_cov = True)

        flux_gp[band_name] = y_gp * out.y_scaler * 1e3

    return time_gp, flux_gp

def apply_time_dilation(time_gp, redshift):

    #Using redshift to calculate the distance to the object
    dist_object = Distance(unit = u.pc, z = redshift, cosmology = WMAP9)

    #Generating a value to add some random scatter to the simulated distance
    dist_scatter = np.random.uniform(0.9, 1.2)

    #Calculating a simulated distance value that places the object the same amount above the LSST detection limit as the original object was above the ZTF detection limit
    dist_LSST = dist_object * dist_scatter

    #Calculate the corresponding redshift for the simulated LSST distance
    lsst_redshift = z_at_value(WMAP9.luminosity_distance, dist_LSST)

    dilation_factor = (1 + lsst_redshift) / (1 + float(redshift))
    gp_timescale_dilated = time_gp * dilation_factor

    return gp_timescale_dilated, lsst_redshift

def apply_reddening(flux_gp, lsst_bands):
    lsst_sim_flux = {}
    fudge = np.random.normal(loc = 1, scale = 0.1)

    for val in lsst_bands:
        lsst_sim_flux[val] = flux_gp["lsst" + val] * fudge

    return lsst_sim_flux

def retrieve_random_lsst_observation_shedule(data_dir = "data"):

    #Using baseline from Rubin Survey Simulator
    baseline_file = f"{data_dir}/rubin_sim_data/baseline_v5.0.0_10yrs.db"
    name = os.path.basename(baseline_file).replace(".db","")
    out_dir = "data/temporary_rubin_maf"
    results_db = maf.db.ResultsDb(out_dir = out_dir)
    bundle_list = []
    metric = maf.metrics.PassMetric(cols = ["filter", "observationStartMJD", "fiveSigmaDepth", "visitExposureTime"])

    #Choosing new RA, Dec and date (in MJD) for simulated lightcurve.
    sim_date = np.random.uniform(60970, 64000)
    sim_obj_ra = np.random.uniform(0, 360); sim_obj_dec = np.random.uniform(-80, 20) # changed from np.random.uniform(-90, 40)

    #Setting up slicer for chosen date, RA and Dec
    sql = ""
    slicer = maf.slicers.UserPointsSlicer(ra = sim_obj_ra, dec = sim_obj_dec)
    bundle_list.append(maf.MetricBundle(metric, slicer, sql, run_name = name))
    bd = maf.metricBundles.make_bundles_dict_from_list(bundle_list)
    bg = maf.metricBundles.MetricBundleGroup(bd, baseline_file, out_dir = out_dir, results_db = results_db)
    bg.run_all()

    #Producing list of visits with their respective filters
    data_slice = bundle_list[0].metric_values[0]

    if type(data_slice) == np.ma.core.MaskedConstant:

        while type(data_slice) == np.ma.core.MaskedConstant:

            results_db = maf.db.ResultsDb(out_dir = out_dir)
            bundle_list = []
            metric = maf.metrics.PassMetric(cols = ["filter", "observationStartMJD", "fiveSigmaDepth", "visitExposureTime"])

            sim_obj_ra = np.random.uniform(0, 360); sim_obj_dec = np.random.uniform(-80, 20)

            sql = ""
            slicer = maf.slicers.UserPointsSlicer(ra = sim_obj_ra, dec = sim_obj_dec)
            bundle_list.append(maf.MetricBundle(metric, slicer, sql, run_name = name))
            bd = maf.metricBundles.make_bundles_dict_from_list(bundle_list)
            bg = maf.metricBundles.MetricBundleGroup(bd, baseline_file, out_dir = out_dir, results_db = results_db)
            bg.run_all()

            data_slice = bundle_list[0].metric_values[0]
            
    if isinstance(data_slice, dict):
        visit_df = pd.DataFrame([data_slice])
    elif isinstance(data_slice, (list, np.ndarray)):
        visit_df = pd.DataFrame(data_slice)
    visit_df.sort_values(by = "observationStartMJD", inplace = True)
    visit_df = visit_df[["observationStartMJD", "filter", "fiveSigmaDepth"]]

    return sim_date, sim_obj_ra, sim_obj_dec, visit_df

def generate_sample(object_id, object_lc_df, object_log_df, spectral_type, data_dir = "data", active_bands = "ugirz"): 

    ### Data pre-processing
    redshift = object_log_df["Z"].iloc[0]
    ebv = object_log_df["EBV"].iloc[0]

    # Apply de-excitation
    for band in active_bands:
        mask = (object_lc_df["Filter"] == band)
        object_lc_df.loc[mask, "Flux"] = flux_de_extinction(object_lc_df.loc[mask, "Flux"], 
                                                            ebv, EFFECTIVE_WAVELENGTHS[band])

    # Discard observations that are not in the active bands
    filt_mask = object_lc_df["Filter"].isin(list(active_bands))
    object_lc_df = object_lc_df[filt_mask]

    # Change flux unit to milliJy as redback assumes flux values are given in this unit
    object_lc_df["Flux"] *= 1e-3
    object_lc_df["Flux_err"] *= 1e-3

    # Rename filters (otherwise redback will assume the wrong frequency associated with a certain LSST band)
    for band in active_bands:
        mask = (object_lc_df["Filter"] == band)
        object_lc_df.loc[mask, "Filter"] = "lsst" + band

    unique_bands = ["lsst" + c for c in active_bands]
    lsst_bands = [band[-1:] for band in unique_bands]

    time_peak = object_lc_df.loc[object_lc_df["Flux"].idxmax(), "Time (MJD)"]
    object_lc_df["Time (MJD)"] -= time_peak

    # Save tde in redback object for GP
    redback_object = redback.transient.TDE(name = object_id, time_mjd = object_lc_df["Time (MJD)"].values, 
                                           flux_density = object_lc_df["Flux"].values, flux_density_err = object_lc_df["Flux_err"].values, 
                                           bands = object_lc_df["Filter"].values, use_phase_model = True, data_mode = "flux_density", save = False) 

    time_gp, flux_gp = fit_gp(object_lc_df, redback_object, unique_bands)

    ### Determining new redshift for simulated object
    gp_timescale_dilated, new_redshift = apply_time_dilation(time_gp, redshift)

    ### Applying reddening to the the fluxes
    lsst_sim_flux = apply_reddening(flux_gp, lsst_bands)

    ### LSST cadence
    condition_fulfilled = False

    while not condition_fulfilled:
        sim_date, sim_obj_ra, sim_obj_dec, visit_df = retrieve_random_lsst_observation_shedule(data_dir)
        visit_list, visit_err, ds_mask = {}, {}, {}

        for val in lsst_bands:
            visit_list[val] = []
            ds_mask[val] = visit_df["filter"] == val # Mask of observations in each filter ? 

        for val in lsst_bands:
            visit_list[val] = visit_df.loc[ds_mask[val], "observationStartMJD"].tolist() # For every filter MJD of observations
            visit_err[val] = visit_df.loc[ds_mask[val], "fiveSigmaDepth"].tolist() 

        peaks_gp = [np.argmax(flux_gp[key]) for key in flux_gp.keys()]
        values, counts = np.unique(peaks_gp, return_counts = True)
        largest_peak = values[np.argmax(counts)]

        gp_timescale_shifted = gp_timescale_dilated + sim_date
        dilated_peak_time = gp_timescale_shifted[largest_peak]
        sim_date_shift = dilated_peak_time - sim_date
        sim_date = sim_date + sim_date_shift

        #Using function to match visit dates and fluxes for each of the six bands
        #Also generates error values
        lsst_visit_list, lsst_visit_flux, lsst_visiter_err = {}, {}, {}

        for val in lsst_bands:
            lsst_visit_list[val], lsst_visit_flux[val], lsst_visiter_err[val] = get_match(visit_list[val], gp_timescale_shifted, lsst_sim_flux[val], visit_err[val])

        flux_err, mjd_lsst_visit_list = {}, {}

        for val in lsst_bands:
            flux_err[val] = ab_to_uJy(lsst_visiter_err[val]) * 0.2

            mjd_lsst_visit_list[val] = lsst_visit_list[val]

        ### Applying extinction and random scatter to lightcurve
        #Calculating extinction value for simulated object using same method as shown in 1.6
        coords = SkyCoord(sim_obj_ra, sim_obj_dec, unit = "deg")
        SFD = SFDQuery()
        new_ebv = SFD(coords) * 0.86

        #Applying extinction to the flux from each band
        for val in lsst_bands:
            lsst_visit_flux[val], band_A_lambda = flux_extinction(lsst_visit_flux[val], new_ebv, EFFECTIVE_WAVELENGTHS[val])

        #Applying random scatter to the lightcurve within measurement uncertainties expected from LSST signal-to-noise. Used to enhance variety in data set.
        for val in lsst_bands:
            lsst_visit_flux[val] = np.random.normal(loc = lsst_visit_flux[val], scale = flux_err[val], size = len(lsst_visit_flux[val]))

        # Check with window if there is a detectable transient
        new_time = np.concatenate(list(mjd_lsst_visit_list.values()))
        new_flux_density = np.concatenate(list(lsst_visit_flux.values()))
        new_flux_density_err = np.concatenate(list(flux_err.values()))
        new_filter = np.concatenate([[key] * len(mjd_lsst_visit_list[key]) for key in mjd_lsst_visit_list.keys()])

        if spectral_type == "AGN": 

            if (len(new_time) >= 17) & (len(new_time) <= 300):
                condition_fulfilled = True 

        else:

            data = {"Time (MJD)": new_time, "Flux": new_flux_density, "Flux_err": new_flux_density_err, "Filter": new_filter}
            lc_data = pd.DataFrame(data = data)

            #Save sample if window detects an event
            condition_fulfilled = detect_event_window_for_object(lc_data, params = window_parameters)["has_event"]

    return new_time, new_flux_density, new_flux_density_err, new_filter, new_redshift, new_ebv

def generate_samples_specific_type(spectral_type, number, output_file_name, data_dir = "data", active_bands = "ugirz"): 

    os.environ["RUBIN_SIM_DATA_DIR"] = f"{data_dir}/rubin_sim_data"

    log_df = pd.read_csv(f"{data_dir}/train_log.csv")

    if spectral_type == "SN":
        type_mask = ((log_df["SpecType"] != "AGN") & (log_df["SpecType"] != "TDE"))

    else:
        type_mask = log_df["SpecType"] == spectral_type

    type_object_id = log_df["object_id"].loc[type_mask].to_numpy()

    object_id_list = np.random.choice(type_object_id, size = number, replace = True)

    output_file = f"{data_dir}/{output_file_name}.csv"
    with open(output_file, "a", newline = "") as file:
        
        writer = csv.writer(file)
        writer.writerow(["object_id", "Time (MJD)", "Flux", "Flux_err", "Filter", "redshift", "EBV"])

        for idx, object_id in enumerate(object_id_list): 

                object_log_df = log_df.loc[log_df["object_id"] == object_id]
                split_nr = object_log_df["split"].iloc[0]
                 
                lightcurve_df = pd.read_csv(f"{data_dir}/{split_nr}/train_full_lightcurves.csv")
                lightcurve_df = lightcurve_df.dropna().reset_index(drop = True)

                object_lc_mask = (lightcurve_df["object_id"] == object_id)
                object_lc_df = lightcurve_df.loc[object_lc_mask].copy()
                object_lc_df.sort_values(by = "Time (MJD)", inplace = True) 

                new_time, new_flux_density, new_flux_density_err, new_filter, new_redshift, new_ebv = generate_sample(object_id, object_lc_df, object_log_df, spectral_type, data_dir, active_bands)
                
                # Write to output file
                for time, flux, flux_err, filter in zip(new_time, new_flux_density, new_flux_density_err, new_filter):
                    writer.writerow([object_id + f"_{idx}", time, flux, flux_err, filter, new_redshift.value, new_ebv])

