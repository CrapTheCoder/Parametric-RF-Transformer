import gdsfactory as gf
import numpy as np
from gdsfactory.path import Path

# GDS layer for the common inner connection arms of the transformer spirals.
COMMON_ARM_LAYER_TUPLE = (100, 0)


@gf.cell
def transformer(
        N1: int = 10, N2: int = 6, w: float = 5.0, s: float = 5.0,
        r1_pri_in: float = 50.0, eps_r: float = 11.68, h_sub: float = 500.0,
        inner_connect_width_factor: float = 1.0
) -> gf.Component:
    """
    Generates a gdsfactory component for a planar RF transformer.

    The transformer has two concentric spiral inductors (primary and secondary).
    Spiral turns are on GDS layers (1,0) (primary) and (2,0) (secondary).
    Inner connection arms to (0,0) are on COMMON_ARM_LAYER_TUPLE (100,0)
    to avoid GDS overlap between primary/secondary layers for these arms.

    Args:
        N1: Number of primary turns.
        N2: Number of secondary turns.
        w: Trace width (µm).
        s: Edge-to-edge spacing between turns (µm).
        r1_pri_in: Inner radius of the primary coil's first turn centerline (µm).
        eps_r: Substrate relative permittivity (informational).
        h_sub: Substrate thickness (µm) (informational).
        inner_connect_width_factor: Factor for 'common' port width at (0,0).

    Returns:
        gf.Component: The generated transformer component.
    """
    c = gf.Component(name=f"transformer_N1_{N1}_N2_{N2}_sep_arms")

    if s < 0:
        print(f"Warning: Spacing 's' is {s} um. Assumed positive edge-to-edge spacing.")

    pitch = w + s  # Centerline-to-centerline distance between adjacent turns

    def create_spiral_and_arm(parent_component: gf.Component, r_in_cl: float, turns: int,
                              coil_layer_tuple: tuple, arm_layer_tuple: tuple, trace_width: float,
                              pts_per_turn: int = 100):
        """
        Creates one spiral coil and its inner connection arm to (0,0).
        The spiral is on `coil_layer_tuple`, the arm is on `arm_layer_tuple`.

        Returns:
            tuple: (outer centerline radius, last spiral point, second to last spiral point)
        """
        r_out_cl_val = r_in_cl
        last_point_spiral = np.array([r_in_cl, 0.0])
        second_last_point_spiral = np.array([r_in_cl, 0.0])

        if turns > 0:
            # --- 1. Curved spiral part (Archimedean spiral) ---
            th_max = 2 * np.pi * turns
            n_pts_curve = max(3, int(turns * pts_per_turn))

            theta_spiral = np.linspace(0, th_max, n_pts_curve)
            r_values_spiral = r_in_cl + (pitch / (2 * np.pi)) * theta_spiral
            curved_segment_points = np.column_stack([
                r_values_spiral * np.cos(theta_spiral),
                r_values_spiral * np.sin(theta_spiral)
            ])

            if curved_segment_points.shape[0] > 0:
                spiral_comp = gf.Path(curved_segment_points).extrude(width=trace_width, layer=coil_layer_tuple)
                parent_component.add_ref(spiral_comp)

                last_point_spiral = curved_segment_points[-1]
                second_last_point_spiral = curved_segment_points[-2] if n_pts_curve > 1 else curved_segment_points[0]
                r_out_cl_val = r_in_cl + (pitch / (2 * np.pi)) * th_max

            # --- 2. Straight inner arm to (0,0) ---
            if r_in_cl > 1e-9:  # Avoid zero-length arm
                inner_arm_points = np.array([[0.0, 0.0], [r_in_cl, 0.0]])
                inner_arm_comp = gf.Path(inner_arm_points).extrude(width=trace_width, layer=arm_layer_tuple)
                parent_component.add_ref(inner_arm_comp)

        return r_out_cl_val, last_point_spiral, second_last_point_spiral

    # --- Primary Coil (N1) ---
    primary_coil_layer = (1, 0)
    r_pri_out_cl = r1_pri_in
    if N1 > 0:
        r_pri_out_cl, p1_last, p1_seclast = create_spiral_and_arm(
            c, r1_pri_in, N1, primary_coil_layer, COMMON_ARM_LAYER_TUPLE, w
        )
        # Add port 'o1' at the end of the primary spiral
        if not np.allclose(p1_last, p1_seclast):
            angle_rad = np.arctan2(p1_last[1] - p1_seclast[1], p1_last[0] - p1_seclast[0]) + np.pi / 2
            c.add_port(name="o1", center=p1_last.tolist(), width=w,
                       orientation=np.rad2deg(angle_rad), layer=primary_coil_layer)
        elif np.any(p1_last):  # Fallback for very short paths
            c.add_port(name="o1", center=p1_last.tolist(), width=w, orientation=0, layer=primary_coil_layer)
    else:
        print(f"Info: N1={N1}, primary spiral not generated.")

    # --- Secondary Coil (N2) ---
    secondary_coil_layer = (2, 0)
    # Calculate inner radius of secondary based on primary's outer radius and spacing
    if N1 > 0:
        r1_sec_in_cl = (r_pri_out_cl + w / 2) + s + (w / 2)
    else:  # If no primary, secondary starts at the baseline inner radius
        r1_sec_in_cl = r1_pri_in

    r_sec_out_cl = r1_sec_in_cl
    if N2 > 0:
        r_sec_out_cl, p2_last, p2_seclast = create_spiral_and_arm(
            c, r1_sec_in_cl, N2, secondary_coil_layer, COMMON_ARM_LAYER_TUPLE, w
        )
        # Add port 'o2' at the end of the secondary spiral
        if not np.allclose(p2_last, p2_seclast):
            angle_rad = np.arctan2(p2_last[1] - p2_seclast[1], p2_last[0] - p2_seclast[0]) + np.pi / 2
            c.add_port(name="o2", center=p2_last.tolist(), width=w,
                       orientation=np.rad2deg(angle_rad), layer=secondary_coil_layer)
        elif np.any(p2_last):  # Fallback
            c.add_port(name="o2", center=p2_last.tolist(), width=w, orientation=0, layer=secondary_coil_layer)
    else:
        print(f"Info: N2={N2}, secondary spiral not generated.")

    # --- Common Port at (0,0) ---
    common_port_width = w * inner_connect_width_factor
    c.add_port(name="common", center=(0, 0), width=common_port_width, orientation=0, layer=COMMON_ARM_LAYER_TUPLE)

    # Store key parameters and calculated dimensions in the component's info attribute
    c.info.update({
        'N1': N1, 'N2': N2, 'w_um': w, 's_um': s, 'r1_pri_in_um': r1_pri_in,
        'eps_r': eps_r, 'h_sub_um': h_sub,
        'r_pri_out_edge_um': (r_pri_out_cl + w / 2) if N1 > 0 else (r1_pri_in + w / 2),
        'r1_sec_in_cl_um': r1_sec_in_cl,
        'r_sec_out_edge_um': (r_sec_out_cl + w / 2) if N2 > 0 else (r1_sec_in_cl + w / 2)
    })
    return c


if __name__ == "__main__":
    # This block is for testing the p-cell when run directly.
    gf.clear_cache()  # Ensure fresh component generation

    # Example 1
    xf_config1 = transformer(N1=1, N2=3, w=20, s=20, r1_pri_in=30)
    xf_config1.plot(return_fig=True).show()

    # Example 2
    xf_image_like = transformer(N1=2, N2=5, w=5, s=10, r1_pri_in=15)
    xf_image_like.plot(return_fig=True).show()
