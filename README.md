# GRIME AI
### GaugeCam Remote Image Manager Educational Artificial Intelligence

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Conda](https://img.shields.io/badge/Conda-GRIMELab-green.svg)](https://anaconda.org/GRIMELab)

GRIME AI is an open-source AI framework for the retrieval, automated quality control, and exploratory analysis of ground-based time-lapse imagery. It provides programmatic access to imagery and data products from the National Ecological Observatory Network (NEON), the PhenoCam Network, and the United States Geological Survey Hydrologic Imagery Visualization and Information System (USGS HIVIS), and supports locally stored imagery. Image segmentation using Segment Anything Model 2 (SAM 2) isolates ecologically meaningful regions such as vegetation, water, and soil to support feature extraction from user-defined areas.

GRIME AI is developed at the University of Nebraska-Lincoln and is supported by NSF Award 2411065 and the U.S. Geological Survey under Grant/Cooperative Agreement No. G23AC00141-00.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Development Setup](#development-setup)
- [Usage](#usage)
- [Validation](#validation)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Features

- **NEON Download Manager** — programmatic access to NEON ground-based imagery (hosted via PhenoCam) and associated NEON data products via the NEON API, with site browsing, date range filtering, and time-of-day filtering
- **PhenoCam Download Manager** — programmatic access to PhenoCam Network imagery via the PhenoCam API, with site browsing, date range filtering, and automatic retrieval of the latest or midday site image
- **USGS HIVIS Download Manager** — programmatic access to USGS HIVIS imagery paired with curated hydrological data products including stream stage and discharge, with site browsing and date range filtering
- **Local imagery support** — load locally stored imagery directly into the GRIME AI workflow without downloading from a public repository
- **Automated image triage** using edge-based (Laplacian) and frequency-based (FFT) blur detection, and grayscale intensity-based brightness assessment
- **Qualitative exploration** via animations, composite slice visualizations, and temporal image sequences
- **Quantitative feature extraction** including Shannon entropy, RGB-based greenness indices (GCC, ExG, VARI, GLI), GLCM texture metrics, Gabor filters, Local Binary Patterns, wavelet decomposition, Fourier analysis, Laws texture energy maps, and HSV color descriptors
- **ROI Analyzer** for interactive region-of-interest definition, management, and correlation analysis across image collections
- **Deep learning-based segmentation** using SAM 2 (Segment Anything Model 2), SegFormer-LoRA, and YOLOv11 for precise, reproducible isolation of target features such as vegetation, water, and soil across large image collections
- **Model training and tuning** workflows supporting fine-tuning of deep learning architectures on user-provided annotated datasets
- **Inference engine** for applying trained or pre-trained segmentation models to single images or entire image collections
- **SAGE annotation tool** for creating, managing, and exporting ground-truth training annotations with edge trace mode, label management, and CSV import/export
- **Command-line interface** (`cli_segment.py`) supporting `--image` and `--folder` flags for batch segmentation workflows
- **Cross-platform** operation on Windows, Linux, and Linux-derived operating systems including HPC clusters and browser-based environments such as Open OnDemand (OOD)
- **FAIR-aligned** outputs in standard CSV and image formats compatible with spreadsheet software and downstream analysis pipelines

---

## Installation

### Option 1: Conda (Recommended)

The Conda package bundles all required dependencies and ensures a consistent runtime environment across supported operating systems.

```bash
conda install grimelab::grime-ai
```

### Option 2: pip

```bash
pip install grime-ai
```

### Option 3: Install from Source

```bash
git clone https://github.com/JohnStranzl/GRIME-AI.git
cd GRIME-AI
pip install -e .
```

> **Note:** Python 3.11 is required. The Conda installation method is strongly recommended as it resolves all dependencies automatically, including PyTorch and SAM 2 model dependencies. Manual installation via pip or from source may require additional configuration steps described in the [project Wiki](https://github.com/JohnStranzl/GRIME-AI/wiki).

---

## Development Setup

GRIME AI is developed using **Python 3.11**, **PyCharm**, and **Qt Creator**. These are the supported and tested development tools. Developers choosing alternative tools (e.g., VS Code) are responsible for configuring their own environment accordingly.

### Prerequisites

- Python 3.11
- PyCharm (Community or Professional)
- Qt Creator (for UI file editing)
- Conda (recommended for dependency management)

### Setting Up the Development Environment

1. **Clone the repository**

```bash
git clone https://github.com/JohnStranzl/GRIME-AI.git
cd GRIME-AI
```

2. **Create and activate a Conda environment**

```bash
conda env create -f environment.yml
conda activate grime-ai
```

3. **Open the project in PyCharm**

   - Open PyCharm and select **File > Open**, then navigate to the cloned repository root.
   - Configure the Python interpreter to use the `grime-ai` Conda environment: **File > Settings > Project > Python Interpreter**.

4. **Edit UI files with Qt Creator**

   - `.ui` files are located in the project UI directory.
   - Open Qt Creator and load the desired `.ui` file for layout editing.
   - **Important:** Always edit `.ui` files directly. Never edit the compiled `_ui.py` files generated from them, as these are overwritten during the build process.

5. **Run the application**

```bash
python GRIME_AI.py
```

### PyTorch and GPU Support

GRIME AI uses SAM 2 for image segmentation, which requires PyTorch 2.8. GPU acceleration is supported on CUDA-compatible hardware. CPU-only operation is supported but will be significantly slower for segmentation tasks.

> **Note for Pascal GPU users (sm 6.0, e.g., GTX 1080):** PyTorch 2.8 dropped sm 6.0 support. Users on Pascal architecture hardware must pin PyTorch to an earlier compatible version:
> ```bash
> conda install "pytorch<2.5" cudatoolkit -c pytorch
> ```

---

## Usage

### Graphical Interface

Launch the GRIME AI desktop application:

```bash
python GRIME_AI.py
```

The graphical interface provides access to all GRIME AI capabilities organized around the six-step data science process:

- **Retrieve Data** — Download imagery and data products from NEON, PhenoCam, or USGS HIVIS, or load locally stored imagery
- **Prepare Data** — Run automated image triage to detect and sequester blurry or poorly exposed images
- **Explore Data** — Generate composite slices, animations, and quantitative feature extractions from user-defined regions of interest

### Command-Line Segmentation Interface

For batch segmentation workflows:

```bash
# Segment a single image
python cli_segment.py --image path/to/image.jpg

# Segment all images in a folder
python cli_segment.py --folder path/to/folder
```

Import the segmentation interface programmatically:

```python
from GRIME_AI.cli_segment import run_sam2, run_segformer
```

> **Note:** The correct import path is `GRIME_AI.cli_segment`, not `grime_ai_segment`.

### SAGE Annotation Tool

GRIME AI includes the SAGE (Segmentation Annotation and Ground-truth Editor) tool for creating and managing training annotations. SAGE supports:

- Edge Trace Mode with live overlays
- Label Manager with persistent label classes and CSV import/export
- Click-to-activate workflow with zoom and pan controls

---

## Validation

Validation test scripts, supporting datasets, test plans, and test results are publicly available on HydroShare:

- [GRIME AI Feature Extraction Validation](https://www.hydroshare.org/resource/710F8AC2E0A3475DA8DF17764091C711/) — doi:10.4211/HS.710F8AC2E0A3475DA8DF17764091C711
- [GRIME AI Triage Validation](https://www.hydroshare.org) — add DOI when available

Validation covers:

- **Image triage** — 100% classification accuracy across blur, brightness, and combined synthetic test series
- **GCC computation** — Mean difference of 0.0% (SD = 0.00003) against PhenoCam benchmark calculations
- **Feature extraction** — Perfect analytical agreement across intensity, entropy, GCC, and HSV metrics on synthetic pure-color images

---

## Documentation

Full documentation, including step-by-step tutorials, feature overviews, and best practices, is available in the [project Wiki](https://github.com/JohnStranzl/GRIME-AI/wiki).

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request. Follow existing code style and include tests for new functionality where applicable.

---

## License

GRIME AI is released under the [Apache 2.0 License](LICENSE).

---

## Citation

If you use GRIME AI in your research, please cite:

> Stranzl JE, Gilmore TE, Harner MJ, Mittelstet A, Joeckel RM, Chapman KW, et al. GRIME AI harnesses deep learning to transform time-lapse imagery into environmental intelligence. *PLOS Water*. (in review)

Additional references:

- **SAM 2:** Ravi N, et al. SAM 2: Segment Anything in Images and Videos. arXiv. 2024. https://arxiv.org/abs/2408.00714
- **PhenoCam Network:** Richardson AD. PhenoCam: An evolving, open-source tool to study the temporal and spatial variability of ecosystem-scale phenology. https://doi.org/10.1016/j.agrformet.2023.109751
- **python-vegindex / PhenoCam vegetation indices:** Richardson AD, et al. Tracking vegetation phenology across diverse North American biomes using PhenoCam imagery. *Sci Data*. 2018. https://doi.org/10.1038/sdata.2018.28
- **xROI:** Seyednasrollah B, Milliman T, Richardson AD. Data extraction from digital repeat photography using xROI. *ISPRS J Photogramm Remote Sens*. 2019. https://doi.org/10.1016/j.isprsjprs.2019.04.009
- **GRIME2:** Gilmore TE, et al. A New GRIME2: Using an Octagon Calibration Target and Trail Camera to Measure Stream Water Level Over a 2-Year Period. *Water Resources Research*. 2026. https://doi.org/10.1029/2025WR042244
- **Stage and discharge from time-lapse imagery:** Chapman KW, Gilmore TE, et al. Stage and discharge prediction from documentary time-lapse imagery. *PLOS Water*. 2024. https://doi.org/10.1371/journal.pwat.0000106
- **NEON:** Thibault KM, et al. The US National Ecological Observatory Network and the Global Biodiversity Framework. *Journal of Ecology and Environment*. 2023. https://doi.org/10.5141/jee.23.076
- **USGS HIVIS:** https://www.usgs.gov/tools/hydrologic-imagery-visualization-and-information-system-hivis
- **Open OnDemand:** Hudak D, et al. Open OnDemand: A web-based client portal for HPC centers. *JOSS*. 2018. https://doi.org/10.21105/joss.00622
- **FAIR Data Principles:** Wilkinson MD, et al. The FAIR Guiding Principles for scientific data management and stewardship. *Sci Data*. 2016. https://doi.org/10.1038/sdata.2016.18
- **KOLA Data Dashboard:** https://csdportal1.unl.edu/portal/apps/dashboards/6779a481d31d4305a46d7edbbcd3b13c

---

## Acknowledgements

GRIME AI is developed at the University of Nebraska-Lincoln, Conservation and Survey Division, School of Natural Resources, in collaboration with Northern Arizona University. Development is supported by NSF Award 2411065 and the U.S. Geological Survey under Grant/Cooperative Agreement No. G23AC00141-00.

Thanks to Keegan Johnson (U.S. Geological Survey, Upper Midwest Water Science Center) for project coordination and liaison support, and to Christopher Terry, Maggie Wells, Mackenzie Smith, and Dawson Kosmicki (University of Nebraska at Kearney) for image annotation and software functionality testing.
