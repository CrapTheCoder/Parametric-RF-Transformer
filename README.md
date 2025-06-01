# Parametric RF Transformer P-Cell & OpenEMS Simulation

Generates planar RF spiral transformer GDS layouts using gdsfactory and simulates them with OpenEMS.

## Features

* Parametric GDS generation (`transformer_pcell.py`).
* Automated OpenEMS simulation pipeline (`simulate_transformer.py`).
* S-parameter extraction (CSV) and plotting (PNG/PDF).

## Project Structure

```
.
├── transformer_pcell.py
├── simulate_transformer.py
├── configs/                # JSON configuration files
│   └── example.json
├── outputs/                # Simulation results (GDS, CSV, plots)
└── README.md
```

## Setup

1. **Prerequisites:**
    * Python 3.11 (recommended)
    * Install OpenEMS & CSXCAD and add to system PATH.
2. **Install Python Dependencies:**
   ```bash
   pip install numpy matplotlib gdsfactory
   ```

## Usage

1. **Create JSON Configuration:**
   Place config files (e.g., `configs/my_transformer.json`) with parameters like:
   ```json
   {
       "f1_ghz": 1.0, "f2_ghz": 10.0, "N1": 2, "N2": 2,
       "w_um": 10.0, "s_um": 5.0, "r1_primary_inner_um": 30.0,
       "eps_r_substrate": 4.4, "h_substrate_um": 100.0,
       "metal_thickness_um": 2.0, "mesh_resolution_factor": 4,
       "EndCriteria": 1e-4, "NrTS": 300000, "numThreads": 8
   }
   ```
2. **Run Simulation:**
    * Specific config: `python simulate_transformer.py configs/my_transformer.json`
    * All configs in `configs/`: `python simulate_transformer.py`
3. **Results:** Output GDS, CSV, and plots are saved in `outputs/<config_name>/`.

## P-Cell Details (`transformer_pcell.py`)

* Primary coil: GDS layer (1,0).
* Secondary coil: GDS layer (2,0).
* Common inner connection arms (to center 0,0): GDS layer (100,0).
* Ports: `o1` (primary), `o2` (secondary), `common` (center).

## Simulation Details (`simulate_transformer.py`)

* Copper GDS layers: (1,0), (2,0), (100,0).
* PEC boundaries; lumped element ports at `o1`, `o2`.
* Common port grounded via cylindrical via at (0,0).

## Assumptions & Limitations

* **Single Metal Layer Spirals:** Coils and arms are on the same physical metal layer.
* **Substrate:** Uniform dielectric on a perfect ground plane.
* **Via Model:** Simple cylindrical via for common ground.
* **Meshing:** Base resolution set by `mesh_resolution_factor`. May need manual tuning for high accuracy/frequency.
* **Computational Cost:** Finer meshes, higher frequencies, and stricter criteria increase simulation time drastically.

## Potential Enhancements

* Multi-layer/interleaved spirals.
* Advanced via/port models.
* Bayseian Optimization.
