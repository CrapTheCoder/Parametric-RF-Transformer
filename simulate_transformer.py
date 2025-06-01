import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gdsfactory as gf  # Used here to load p-cell for simulation setup

import transformer_pcell  # Import the p-cell module
from openems import openems

# Unit constants
MM = 1e-3
UM = 1e-6


def setup_simulation(params: dict, sim_name: str = "xf_sim"):
    """
    Sets up the OpenEMS simulation for a transformer.

    Args:
        params: Dictionary of transformer geometry and simulation settings.
        sim_name: Base name for simulation files.

    Returns:
        openems.OpenEMS: Configured OpenEMS simulation object.
    """
    # Extract parameters from dictionary
    f_start_ghz = params['f1_ghz']
    f_stop_ghz = params['f2_ghz']
    N1 = params['N1']
    N2 = params['N2']
    w_um = params['w_um']
    s_um = params['s_um']
    r1_pri_in_um = params['r1_primary_inner_um']
    eps_r_sub = params['eps_r_substrate']
    h_sub_um = params['h_substrate_um']

    # Optional parameters with defaults
    metal_th_um = params.get('metal_thickness_um', 1.0)
    sub_tand = params.get('substrate_tand', 0.002)
    box_pad_factor = params.get('sim_box_padding_factor', 1.2)
    mesh_res_factor = params.get('mesh_resolution_factor', 4)
    end_crit = params.get('EndCriteria', 1e-4)
    nrts = params.get('NrTS', 3000000)
    fsteps = params.get('fsteps_output', 101)
    add_common_ground_via = params.get('add_common_ground_via', True)

    # 1. Generate GDS device using the p-cell
    gf.clear_cache()
    xf_device = transformer_pcell.transformer(
        N1=N1, N2=N2, w=w_um, s=s_um,
        r1_pri_in=r1_pri_in_um,
        eps_r=eps_r_sub, h_sub=h_sub_um
    )
    xf_device.plot(return_fig=True).show()  # Show layout being simulated

    # Get device dimensions for simulation box sizing
    r_pri_out_edge = xf_device.info.get('r_pri_out_edge_um', r1_pri_in_um + w_um / 2)
    r_sec_in_cl = xf_device.info.get('r1_sec_in_cl_um', r_pri_out_edge + s_um + w_um / 2)
    r_sec_out_edge = xf_device.info.get('r_sec_out_edge_um', r_sec_in_cl + w_um / 2)
    max_dev_r_um = max(r_pri_out_edge, r_sec_out_edge)
    if N1 == 0 and N2 == 0: max_dev_r_um = r1_pri_in_um + w_um  # Handle no-turn case

    # 2. Initialize OpenEMS Simulation
    em = openems.OpenEMS(
        sim_name, EndCriteria=end_crit,
        fmin=f_start_ghz * 1e9, fmax=f_stop_ghz * 1e9,
        fsteps=fsteps, NrTS=nrts
    )

    # 3. Define Materials
    copper = openems.Metal(em, 'copper')
    ground_plane_metal = openems.Metal(em, 'ground_plane_metal')
    sub_mat = openems.Dielectric(
        em, 'substrate', eps_r=eps_r_sub,
        tand=sub_tand, fc=f_stop_ghz * 1e9
    )

    # Convert dimensions to meters
    h_sub_m = h_sub_um * UM
    metal_th_m = metal_th_um * UM
    sub_xy_span = 2 * max_dev_r_um * box_pad_factor * UM
    sub_start_xy = -sub_xy_span / 2
    sub_stop_xy = sub_xy_span / 2

    # 4. Define Geometry Primitives
    # Substrate
    openems.Box(sub_mat, 1,
                start=[sub_start_xy, sub_start_xy, 0],
                stop=[sub_stop_xy, sub_stop_xy, h_sub_m])
    # Ground Plane
    gnd_plane_thickness_m = metal_th_m
    openems.Box(ground_plane_metal, priority=10,
                start=[sub_start_xy, sub_start_xy, 0 - gnd_plane_thickness_m],
                stop=[sub_stop_xy, sub_stop_xy, 0])
    print(f"Info: Added explicit ground plane from z={-gnd_plane_thickness_m:.3e} to z=0.")

    # Transformer Traces from GDS polygons
    gds_polys_by_spec = xf_device.get_polygons(by="tuple")
    metal_z0_on_sub = h_sub_m
    metal_z1_on_sub = h_sub_m + metal_th_m
    dbu_to_um = 0.001  # GDS database units to micrometers

    # GDS layers to be treated as copper
    copper_gds_layers = [(1, 0), (2, 0), transformer_pcell.COMMON_ARM_LAYER_TUPLE]

    for spec, polys_list in gds_polys_by_spec.items():
        if spec in copper_gds_layers:
            if not polys_list: continue
            for klay_poly in polys_list:  # klayout.db.Polygon or gdsfactory Polygon
                poly_um_np = None
                if hasattr(klay_poly, 'points') and isinstance(klay_poly.points, np.ndarray):
                    poly_um_np = klay_poly.points * dbu_to_um
                elif hasattr(klay_poly, 'each_point_hull'):  # For klayout.db.Polygon
                    pts_um_list = [[p.x * dbu_to_um, p.y * dbu_to_um] for p in klay_poly.each_point_hull()]
                    if pts_um_list: poly_um_np = np.array(pts_um_list)
                else:
                    continue

                if poly_um_np is None or poly_um_np.size < 6 or \
                        not (poly_um_np.ndim == 2 and poly_um_np.shape[1] == 2):
                    print(f"Warn: Invalid/empty polygon data for layer {spec}. Skipping.")
                    continue

                poly_m = poly_um_np * UM  # Convert to meters
                openems.Polygon(
                    copper, points=poly_m,
                    elevation=[metal_z0_on_sub, metal_z1_on_sub],
                    normal_direction='z', priority=10
                )

    # Common Ground Via
    if add_common_ground_via and (N1 > 0 or N2 > 0):
        via_radius_um = params.get('common_via_radius_um', w_um)
        via_radius_m = via_radius_um * UM
        openems.Cylinder(copper, priority=11,
                         start=[0, 0, metal_z0_on_sub], stop=[0, 0, 0],
                         radius=via_radius_m)
        print(f"Added common ground via at (0,0) from z={metal_z0_on_sub:.3e} to z=0, radius={via_radius_m:.3e}m.")
        em.mesh.AddLine('x', [-via_radius_m, via_radius_m])  # Mesh refinement for via
        em.mesh.AddLine('y', [-via_radius_m, via_radius_m])

    # 5. Boundary Conditions
    em.boundaries = ['PEC'] * 6  # Perfect Electric Conductor on all sides

    # 6. Mesh Definition
    min_feat_um = min(w_um, s_um) if w_um > 0 and s_um > 0 else max(w_um, s_um, 1.0)
    base_res_m = (min_feat_um / mesh_res_factor) * UM if mesh_res_factor > 0 else min_feat_um * UM

    port_mesh_x, port_mesh_y = [], []
    port_feed_len_m = 3 * base_res_m if base_res_m > 0 else 3 * UM  # Heuristic feed length

    gds_ports_map = {}  # Map GDS port names to simulation ports
    try:
        if N1 > 0 and 'o1' in xf_device.ports: gds_ports_map['P1'] = xf_device.ports['o1']
    except KeyError:
        print("Warn: GDS Port 'o1' (KeyError).")
    try:
        if N2 > 0 and 'o2' in xf_device.ports: gds_ports_map['P2'] = xf_device.ports['o2']
    except KeyError:
        print("Warn: GDS Port 'o2' (KeyError).")

    processed_ports_info = []  # Store details for creating OpenEMS ports
    for port_label, gds_port in gds_ports_map.items():
        center_m = np.array(gds_port.center) * UM
        width_m = gds_port.width * UM
        angle_deg = gds_port.orientation
        dx, dy = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
        px, py = center_m[0], center_m[1]
        p0, p1, d_char = [0, 0, 0], [0, 0, 0], 'x'  # Port start, stop, direction

        if abs(dx) > abs(dy):  # X-directed port
            d_char = 'x'
            y_cs = [py - width_m / 2, py + width_m / 2];
            port_mesh_y.extend(y_cs)
            x_cs = [px - port_feed_len_m, px] if dx > 0 else [px, px + port_feed_len_m]
            p0, p1 = [x_cs[0], y_cs[0], metal_z0_on_sub], [x_cs[1], y_cs[1], metal_z1_on_sub]
            port_mesh_x.extend(x_cs)
        else:  # Y-directed port
            d_char = 'y'
            x_cs = [px - width_m / 2, px + width_m / 2];
            port_mesh_x.extend(x_cs)
            y_cs = [py - port_feed_len_m, py] if dy > 0 else [py, py + port_feed_len_m]
            p0, p1 = [x_cs[0], y_cs[0], metal_z0_on_sub], [x_cs[1], y_cs[1], metal_z1_on_sub]
            port_mesh_y.extend(y_cs)
        processed_ports_info.append({'start': p0, 'stop': p1, 'direction': d_char})

    mesh_xy_ext = max_dev_r_um * UM
    em.mesh.AddLine('x', sorted(list(set([-mesh_xy_ext, 0, mesh_xy_ext] + port_mesh_x))))
    em.mesh.AddLine('y', sorted(list(set([-mesh_xy_ext, 0, mesh_xy_ext] + port_mesh_y))))

    air_pad_z_m = sub_xy_span / 4  # Air padding in Z
    z_lines = sorted(list({  # Key Z-coordinates for mesh lines
        0 - gnd_plane_thickness_m, 0, h_sub_m, metal_z1_on_sub,
        metal_z1_on_sub + air_pad_z_m, -air_pad_z_m
    }))
    em.mesh.AddLine('z', z_lines)
    em.resolution = [base_res_m] * 3  # Set base mesh resolution

    # 7. Define Ports
    for p_info in processed_ports_info:
        openems.Port(em, start=p_info['start'], stop=p_info['stop'],
                     direction=p_info['direction'], z=50)  # 50 Ohm port impedance

    return em


def run_and_extract(em: openems.OpenEMS, num_threads: int = 8):
    """
    Runs the OpenEMS simulation and extracts S-parameters.

    Returns:
        tuple: (frequencies_array, s_parameters_dict)
    """
    s_raw_vec = em.run_openems(options='solve', numThreads=num_threads, show_plot=False)

    if s_raw_vec is None:
        print("Err: Sim returned no S-params");
        sys.exit(1)

    freqs = getattr(em, 'frequencies', None)
    if freqs is None:  # Fallback if frequencies attribute not found
        print("Err: Freq vector not in 'em.frequencies'. Reconstructing")
        freqs = np.linspace(em.fmin, em.fmax, em.fsteps)
        if not s_raw_vec or not em.ports or len(s_raw_vec[0]) != len(freqs):
            print("Err: Freq length mismatch");
            sys.exit(1)

    s = {}  # Dictionary to store S-parameters
    n_ports = len(em.ports)

    # Extract S-parameters (simplified for 2-port, assumes S12=S21, S22 not primary focus)
    s['S11'] = s_raw_vec[0] if n_ports >= 1 and len(s_raw_vec) >= 1 else np.zeros_like(freqs, dtype=complex)
    s['S21'] = s_raw_vec[1] if n_ports >= 2 and len(s_raw_vec) >= 2 else np.zeros_like(freqs, dtype=complex)
    s['S12'] = s['S21']  # Assume reciprocal passive network
    s['S22'] = np.zeros_like(freqs, dtype=complex)  # S22 not explicitly extracted/plotted by default

    if n_ports < 2:  # Handle single port case
        s['S21'] = np.zeros_like(freqs, dtype=complex)
        s['S12'] = np.zeros_like(freqs, dtype=complex)

    # Ensure all S-parameter keys exist with correct size
    for k in ['S11', 'S21', 'S12', 'S22']:
        if k not in s or s[k].size != freqs.size:
            s[k] = np.zeros_like(freqs, dtype=complex)

    return freqs, s


def save_results_csv(freqs: np.ndarray, s_params: dict, out_dir: str, name_base: str):
    """Saves S-parameters (magnitude and phase) to a CSV file."""
    if freqs is None or freqs.size == 0 or not s_params:
        print("No valid data for CSV");
        return

    fpath = os.path.join(out_dir, f"{name_base}_sparameters.csv")
    hdrs = ["Frequency (Hz)"];
    cols = [freqs]

    for k_s in ['S11', 'S21', 'S12', 'S22']:
        s_data = s_params.get(k_s)
        if s_data is not None and s_data.size == freqs.size:
            cols.extend([np.abs(s_data), np.angle(s_data)])
            hdrs.extend([f"|{k_s}|", f"Phase({k_s}) (rad)"])
        else:
            cols.extend([np.zeros_like(freqs)] * 2)
            hdrs.extend([f"|{k_s}| (nodata)", f"Phase({k_s}) (rad) (nodata)"])

    data_mtx = np.array(cols).T
    try:
        np.savetxt(fpath, data_mtx, delimiter=',', header=",".join(hdrs), comments='')
        print(f"S-params saved: {fpath}")
    except Exception as e:
        print(f"Err saving CSV {fpath}: {e}")


def plot_results(freqs: np.ndarray, s_params: dict, out_dir: str, name_base: str):
    """Plots S11, S21, S12 in dB and saves the plots."""
    if freqs is None or freqs.size == 0 or not s_params:
        print("No valid data to plot");
        return

    f_ghz = freqs / 1e9
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f'S-Parameters: {name_base}', fontsize=15)

    # Plot Return Loss (S11, S22 if available and non-zero)
    s11, s22 = s_params.get('S11'), s_params.get('S22')
    if s11 is not None and s11.size == freqs.size:
        axes[0].plot(f_ghz, 20 * np.log10(np.abs(s11) + 1e-10), label='|S11| (dB)')
    if s22 is not None and s22.size == freqs.size and np.any(np.abs(s22) > 1e-9):  # Only plot S22 if meaningful
        axes[0].plot(f_ghz, 20 * np.log10(np.abs(s22) + 1e-10), label='|S22| (dB)', linestyle='--')
    axes[0].set_title('Return Loss', fontsize=13)
    axes[0].set_xlabel('Freq (GHz)');
    axes[0].set_ylabel('Mag (dB)')
    axes[0].set_ylim(-80, 5);
    axes[0].grid(True, linestyle=':', alpha=0.6);
    axes[0].legend()

    # Plot Transmission/Coupling (S21, S12)
    s21, s12 = s_params.get('S21'), s_params.get('S12')
    if s21 is not None and s21.size == freqs.size:
        axes[1].plot(f_ghz, 20 * np.log10(np.abs(s21) + 1e-10), label='|S21| (dB)')
    if s12 is not None and s12.size == freqs.size and \
            (np.any(np.abs(s12) > 1e-9) or np.any(np.abs(s21) > 1e-9)):  # Plot S12 if meaningful
        axes[1].plot(f_ghz, 20 * np.log10(np.abs(s12) + 1e-10), label='|S12| (dB)', linestyle='--')
    axes[1].set_title('Transmission/Coupling', fontsize=13)
    axes[1].set_xlabel('Freq (GHz)');
    axes[1].set_ylabel('Mag (dB)')
    axes[1].set_ylim(-150, 5);
    axes[1].grid(True, linestyle=':', alpha=0.6);
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])  # Adjust layout

    for ext in ['png', 'pdf']:  # Save in multiple formats
        fpath = os.path.join(out_dir, f"{name_base}_sparameters.{ext}")
        try:
            plt.savefig(fpath);
            print(f"Plot saved: {fpath}")
        except Exception as e:
            print(f"Err saving plot {fpath}: {e}")
    plt.close(fig)


if __name__ == "__main__":
    # Main script execution block
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_dir = os.path.join(script_dir, 'configs')
    gen_out_dir = os.path.join(script_dir, "outputs")

    cfg_files_to_run = []
    # Handle command-line argument for specific config or run all in 'configs' dir
    if len(sys.argv) > 1:
        cfg_fpath_arg = sys.argv[1]
        if not os.path.exists(cfg_fpath_arg): sys.exit(f"Err: Config not found: {cfg_fpath_arg}")
        cfg_files_to_run.append(cfg_fpath_arg)
    else:
        print(f"No specific config. Searching in: {cfg_dir}")
        if not os.path.isdir(cfg_dir): sys.exit(f"Err: 'configs' dir not found: {cfg_dir}")
        json_fs = [os.path.join(cfg_dir, f) for f in os.listdir(cfg_dir) if f.endswith(".json")]
        if not json_fs: sys.exit(f"No JSON configs found in {cfg_dir}")
        cfg_files_to_run.extend(json_fs)
        print(f"Found {len(cfg_files_to_run)} configs: {cfg_files_to_run}")

    # Process each identified configuration file
    for cfg_fpath in cfg_files_to_run:
        print(f"\n--- Processing: {cfg_fpath} ---")
        params = {}
        try:
            with open(cfg_fpath, 'r') as f:
                params = json.load(f)
        except Exception as e:
            print(f"Err loading JSON {cfg_fpath}: {e}");
            continue

        sim_base = os.path.splitext(os.path.basename(cfg_fpath))[0]
        spec_out_dir = os.path.join(gen_out_dir, sim_base)

        # Create output directories if they don't exist
        for p_dir_to_check in [gen_out_dir, spec_out_dir]:
            if not os.path.exists(p_dir_to_check):
                try:
                    os.makedirs(p_dir_to_check)
                except OSError as e:
                    print(f"Err creating dir {p_dir_to_check}: {e}"); continue
        print(f"Output for this run: {spec_out_dir}")

        # Generate a reference GDS file for this simulation run
        try:
            print("Generating GDS")
            gf.clear_cache()
            ref_xf_dev = transformer_pcell.transformer(
                N1=params['N1'], N2=params['N2'], w=params['w_um'], s=params['s_um'],
                r1_pri_in=params['r1_primary_inner_um'],
                eps_r=params['eps_r_substrate'], h_sub=params['h_substrate_um']
            )
            gds_fpath = os.path.join(spec_out_dir, f"{sim_base}.gds")
            ref_xf_dev.write_gds(gds_fpath)
            print(f"Reference GDS: {gds_fpath}")
        except Exception as e:
            print(f"Err GDS gen for {sim_base}: {e}");
            continue

        # Setup and run the OpenEMS simulation
        print("Setting up OpenEMS")
        ems_sim_name = f"sim_{sim_base}"
        ems_env = None
        try:
            ems_env = setup_simulation(params, sim_name=ems_sim_name)
        except Exception as e:
            print(f"Err OpenEMS setup for {sim_base}: {e}");
            continue

        if ems_env is None:
            print(f"Skipping run for {sim_base} due to setup failure");
            continue

        n_threads = params.get("numThreads", 8)
        print(f"Running simulation with {n_threads} threads")
        freqs_data, s_params_data = None, None
        try:
            freqs_data, s_params_data = run_and_extract(ems_env, num_threads=n_threads)
        except Exception as e:
            print(f"Err OpenEMS run for {sim_base}: {e}")

        # Save and plot results if simulation was successful
        if freqs_data is not None and s_params_data is not None and freqs_data.size > 0:
            print("Saving & Plotting results")
            save_results_csv(freqs_data, s_params_data, spec_out_dir, sim_base)
            plot_results(freqs_data, s_params_data, spec_out_dir, sim_base)
        else:
            print(f"Skipping results for {sim_base} (no data/errors)")
        print(f"--- Finished: {cfg_fpath} ---")

    print("\nAll configurations processed")
    print("Script finished")
