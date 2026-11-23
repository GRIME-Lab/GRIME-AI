# GRIME AI
### GaugeCam Remote Image Manager Educational — Artificial Intelligence

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Conda](https://img.shields.io/badge/Conda-GRIMELab-green.svg)](https://anaconda.org/GRIMELab)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22087226.svg)](https://doi.org/10.5281/zenodo.22087226)

GRIME AI is an open-source Python platform for retrieving, quality-controlling, and analyzing environmental time-lapse imagery with AI. It provides programmatic access to imagery and data products from the United States Geological Survey Hydrologic Imagery Visualization and Information System (USGS HIVIS), the PhenoCam Network, and the National Ecological Observatory Network (NEON), and works with locally stored imagery. From one application, users can triage image quality, extract quantitative features, annotate training data, fine-tune segmentation models, and run inference across entire image collections, through either a graphical interface or a command-line interface for batch processing on high-performance computing (HPC) systems.

GRIME AI is developed at the University of Nebraska-Lincoln and is supported by NSF Award 2411065 and the U.S. Geological Survey under Grant/Cooperative Agreement No. G23AC00141-00.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Validation](#validation)
- [Documentation](#documentation)
- [Contributing and Support](#contributing-and-support)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Features

### Data retrieval
- **USGS HIVIS**: site browsing and date-range retrieval of HIVIS imagery, paired with hydrological data products including stream stage and discharge
- **PhenoCam Network**: site browsing, date-range and time-of-day filtering, and retrieval of the latest or midday site image
- **NEON**: site browsing and retrieval of NEON imagery and data products, with date-range and time-of-day filtering
- **Local imagery**: load locally stored imagery directly into the GRIME AI workflow

### Image quality triage
- Automated blur detection (Laplacian and Fast Fourier Transform) and brightness assessment
- Flagged images are retained rather than deleted, with a full audit trail for reprocessing at adjusted thresholds
- Structured reports in CSV and human-readable formats

### Exploration and feature extraction
- Composite slices, GIF and video animations, and temporal image sequences
- Quantitative features including Shannon entropy, RGB-based greenness indices (GCC, ExG, VARI, GLI), GLCM texture metrics, Gabor filters, Local Binary Patterns, wavelet decomposition, Fourier analysis, Laws texture energy maps, and HSV color descriptors
- **ROI Analyzer** for interactive region-of-interest definition, management, and correlation analysis across image collections

### Annotation, training, and inference
- **SAGE** (Segmentation & Annotation for Geospatial Ecohydrology), a companion annotation tool launched from GRIME AI. SAM 2 proposes a mask from a point prompt, or the annotator draws one freehand; reviewed masks are exported in COCO 1.0 format for the GRIME AI training pipeline
- Fine-tuning of **SAM 2** (Segment Anything Model 2) and **SegFormer** on user-annotated imagery, with leave-one-season-out (LOSO) cross-validation
- Inference with trained models on single images or entire image collections
- SAM 3 foundation-model inference and YOLOv11 segmentation

### Interfaces and platforms
- Graphical interface (PyQt5) and command-line interface (CLI) for batch processing on HPC systems
- Runs on Windows and Linux, including HPC clusters and browser-based virtual desktops such as Open OnDemand
- Outputs in standard CSV and image formats compatible with spreadsheet software and downstream analysis pipelines

---

## Installation

Python 3.11 is required.

### Option 1: Conda (recommended)

The Conda package bundles all required dependencies, including PyTorch and SAM 2, and provides a consistent runtime environment across supported operating systems.

```bash
conda install grimelab::grime-ai
```

### Option 2: Docker

Container images are published to the GitHub Container Registry:

```bash
docker pull ghcr.io/grime-lab/grime-ai:stable
```

The `stable` tag tracks tagged releases; the `nightly` tag tracks the latest nightly build.

### Option 3: From source

```bash
git clone https://github.com/GRIME-Lab/GRIME-AI.git
cd GRIME-AI
pip install -e .
```

Installing from source may require additional configuration described in the [project Wiki](https://github.com/GRIME-Lab/GRIME-AI/wiki).

### GPU support

SAM 2 requires PyTorch 2.8. GPU acceleration is supported on CUDA-compatible hardware. CPU-only operation is supported but is significantly slower for segmentation.

> **Pascal GPUs (sm 6.0, e.g., GTX 1080):** PyTorch 2.8 dropped sm 6.0 support. Pin PyTorch to an earlier compatible version:
> ```bash
> conda install "pytorch<2.5" cudatoolkit -c pytorch
> ```

---

## Usage

### Graphical interface

Launch the GRIME AI desktop application:

```bash
grime-ai
```

SAGE can also be launched on its own:

```bash
sage
```

The tabs follow the analysis pipeline:

1. **Retrieve** imagery and data products from USGS HIVIS, PhenoCam, or NEON, or load local imagery
2. **Triage** image quality and flag blurry or poorly exposed images
3. **Explore** imagery with composite slices, animations, and the ROI Analyzer
4. **Extract features** from whole images or user-defined regions
5. **Annotate** training images in SAGE
6. **Train** a segmentation model and **run inference** on new imagery

### Command-line interface

The CLI provides subcommands for batch processing:

| Command | Purpose |
|---|---|
| `triage` | Flag images with brightness, blur, or rotation problems; optionally move flagged images and generate a report |
| `slice` | Extract a fixed-width horizontal band centered on a pixel row from each image |
| `coco` | Build a COCO-format annotation file from images and masks |
| `train` | Train or fine-tune a SAM 2 or SegFormer model headlessly from a site configuration file |
| `segment` | Run a trained SAM 2 or SegFormer model on one image or a folder of images |
| `roi` | Run ROI color-clustering feature extraction on image/mask pairs |

```bash
python -m GRIME_AI.main <command> [options]
```

Data retrieval and correlation analysis are available in the graphical interface only. Full options for each command are documented on the [CLI Wiki page](https://github.com/GRIME-Lab/GRIME-AI/wiki/07-Command-Line-Interface-(CLI)).

---

## Development Setup

GRIME AI is developed with **Python 3.11**, **PyCharm**, and **Qt Creator**. These are the supported and tested development tools. Developers using other tools (e.g., VS Code) are responsible for configuring their own environment.

1. **Clone the repository**

   ```bash
   git clone https://github.com/GRIME-Lab/GRIME-AI.git
   cd GRIME-AI
   ```

2. **Create a Conda environment and install GRIME AI in editable mode**

   ```bash
   conda create -n grime-ai python=3.11
   conda activate grime-ai
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Open the project in PyCharm** (**File > Open**) and set the interpreter to the Conda environment (**File > Settings > Project > Python Interpreter**).

4. **Edit UI files with Qt Creator.** Always edit the `.ui` files directly. Never edit the generated `_ui.py` files; they are overwritten during the build.

---

## Testing

[CONFIRM: where the automated tests live and the command that runs them, e.g., `pytest tests/`.]

Continuous integration runs through GitHub Actions, which also publishes Conda packages for nightly builds and tagged releases to the GRIME Lab Anaconda.org channel and Docker images to the GitHub Container Registry.

---

## Validation

Validation scripts, datasets, test plans, and results are publicly available on HydroShare:

- [GRIME AI Feature Extraction Validation](https://doi.org/10.4211/hs.710f8ac2e0a3475da8df17764091c711), doi:10.4211/hs.710f8ac2e0a3475da8df17764091c711
- [GRIME AI Image Triage Validation Using Synthetically Generated Images](https://doi.org/10.4211/hs.d0381011dbb044f3bd22aa7256ccb911), doi:10.4211/hs.d0381011dbb044f3bd22aa7256ccb911

Validation covers:

- **Image triage**: classification of blur, brightness, and combined synthetic test series
- **GCC computation**: agreement with the PhenoCam GCC algorithm on synthetic solid-color test images (mean difference < 0.00001, SD = 0.00003)
- **Feature extraction**: analytical agreement for intensity, entropy, GCC, and HSV metrics on synthetic solid-color images

---

## Documentation

Installation guides, tutorials, feature overviews, and the CLI reference are in the [project Wiki](https://github.com/GRIME-Lab/GRIME-AI/wiki). The Wiki is updated as features are added or changed.

---

## Contributing and Support

- **Report a bug or request a feature:** open an [issue](https://github.com/GRIME-Lab/GRIME-AI/issues). For bugs, include your operating system, GRIME AI version, installation method, and the steps that reproduce the problem.
- **Ask a question or get help:** open an issue labeled *question*.
- **Contribute code:** open an issue to discuss the proposed change first, then fork the repository and submit a pull request. Follow the existing code style and include tests for new functionality.

---

## License

GRIME AI is released under the [Apache 2.0 License](LICENSE).

---

## Citation

If you use GRIME AI in your research, please cite the software:

> Stranzl Jr., J. E., Gilmore, T. E., Bin Issa, R., Caprez, A., Fell, M., & Harner, M. J. (2026). *GRIME AI* (v2.1.6.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22087226

[CONFIRM: add the JOSS paper citation once published, and the PLOS Water manuscript once accepted.]

Related work:

- **SAM 2:** Ravi, N., et al. (2024). SAM 2: Segment Anything in Images and Videos. arXiv. https://arxiv.org/abs/2408.00714
- **SegFormer:** Xie, E., et al. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. *Advances in Neural Information Processing Systems*, 34, 12077–12090.
- **PhenoCam Network:** Richardson, A. D., et al. (2018). Tracking vegetation phenology across diverse North American biomes using PhenoCam imagery. *Scientific Data*, 5, 180028. https://doi.org/10.1038/sdata.2018.28
- **PhenoCam review:** Richardson, A. D. (2023). PhenoCam: An evolving, open-source tool to study the temporal and spatial variability of ecosystem-scale phenology. *Agricultural and Forest Meteorology*. https://doi.org/10.1016/j.agrformet.2023.109751
- **xROI:** Seyednasrollah, B., Milliman, T., & Richardson, A. D. (2019). Data extraction from digital repeat photography using xROI. *ISPRS Journal of Photogrammetry and Remote Sensing*, 152, 132–144. https://doi.org/10.1016/j.isprsjprs.2019.04.009
- **GRIME2:** Gilmore, T. E., et al. (2026). A New GRIME2: Using an Octagon Calibration Target and Trail Camera to Measure Stream Water Level Over a 2-Year Period. *Water Resources Research*, 62(3). https://doi.org/10.1029/2025WR042244
- **Stage and discharge from time-lapse imagery:** Chapman, K. W., et al. (2024). Stage and discharge prediction from documentary time-lapse imagery. *PLOS Water*, 3(4). https://doi.org/10.1371/journal.pwat.0000106
- **NEON:** Thibault, K. M., et al. (2023). The US National Ecological Observatory Network and the Global Biodiversity Framework. *Journal of Ecology and Environment*. https://doi.org/10.5141/jee.23.076
- **USGS HIVIS:** https://apps.usgs.gov/hivis/
- **Open OnDemand:** Hudak, D., et al. (2018). Open OnDemand: A web-based client portal for HPC centers. *Journal of Open Source Software*, 3(25), 622. https://doi.org/10.21105/joss.00622
- **FAIR Data Principles:** Wilkinson, M. D., et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship. *Scientific Data*, 3, 160018. https://doi.org/10.1038/sdata.2016.18
- **KOLA Data Dashboard:** https://csdportal1.unl.edu/portal/apps/dashboards/6779a481d31d4305a46d7edbbcd3b13c

---

## Acknowledgements

GRIME AI is developed at the University of Nebraska-Lincoln, School of Natural Resources, Conservation and Survey Division, in collaboration with Northern Arizona University.

This material is based upon work supported by the National Science Foundation under Award No. 2411065 and by the U.S. Geological Survey under Grant/Cooperative Agreement No. G23AC00141-00. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation or the U.S. Geological Survey. Mention of trade names or commercial products does not constitute their endorsement by the U.S. Geological Survey.

Thanks to Keegan Johnson (U.S. Geological Survey, Upper Midwest Water Science Center) for project coordination and liaison support, and to Christopher Terry, Jamila Bajelan, Dawson Kosmicki, Mackenzie Smith, Maggie Wells, Jahir Uddin, Pavan Guggilla, Christopher B. Coffey, Razin Bin Issa, and Juan Sebastian Camacho Puerto for annotation, testing, and validation contributions.
