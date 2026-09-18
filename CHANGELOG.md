# CHANGELOG

All notable changes to GRIME AI will be documented in this file.

## [1.1.0.0-alpha] - 2025-12-01

### Fixed
- Inconsistent loss functions - resolved BCEWithLogitsLoss on logits vs sigmoid+BCE discrepancy
- Memory leak in validation where pixels accumulated across all epochs
- Removed re-instantiation of already instantiated Predictor
- Removed redundant device definitions

### Changed
- Moved SDPA backend trial to once at startup instead of looping on every sample
- Checkpoint management now keeps top N scoring models
- Unused AMP scaler made optional (defaults to False)
- Validation inhibited during early epochs with auto-configuration based on total epochs

### Added
- COCO Viewer tab to Machine Learning Image Processing dialog for reviewing annotation files and diagnosing feature labeling issues

## [1.0.0.4-alpha] - 2025-11-03

### Added
- Radio buttons for selecting between 3 ML models when training: SAM2, LoRA, or Mask R-CNN (note: currently only SAM2 is active)
- New Phenocam_API class for interacting with Phenocam website using Phenocam API
- New Phenocam tab on main GRIME AI canvas
- Additional map colors for enhanced visualization
- HYDRA_FULL_ERROR environment variable (os.environ["HYDRA_FULL_ERROR"] = "1") for complete stack traces
- R. Issa to authorized users list for beta features
- Icon to exe - GRIME AI logo now appears in task bar instead of generic floppy disk icon
- Support for all Microsoft proprietary EXIF tags (Windows "XP" tags)
- Geopandas support for independent study (Kalispel Tribe) research
- .venv to list of folders to ignore
- Console messages to isolate DNS errors
- Windows and Linux scripts to help developers install SAM2 (install_sam2.bat and shell script)
- Scripts to display OpenStreetMap and Google Maps with push pins for USGS and Phenocam sites
- Script to snapshot user's torch environment (CUDA version, torch version, etc.)
- Script to check GRIME AI code for Linux and Windows compatibility

### Changed
- Refactored all LoRA code to integrate into GRIME AI (initial incorporation, three modeling pipelines: SAM2, Mask R-CNN, and LoRA)
- Moved core NEON API and NEON interaction code to separate subfolder
- Refactored older Phenocam API and created new Phenocam API with new functionality (intent to obsolesce older API)
- Made NEON site interaction more robust when using VPN (addressed SSL handshake failures/SSLError)
- Incorporated latest fixes/updates from Razin to Image Organizer class
- Improved cross-platform path handling (cleaned up Linux vs Windows path issues, replaced backslash with forward slash)
- Modified code to support new geomaps project subfolder containing all mapping functions/files (GIS or otherwise)
- Created separate project subfolder for all mapping functions/files
- Updated GIS/OpenStreetMap code
- Upgraded camera site map visualization design and capabilities
- Refactored NEON images download code
- Put all splashscreen graphics into resources subfolder
- Put all toolbar icons into resources subfolder
- Moved color segmentation dialog box into "dialogs" project subfolder ('color_segmentation')
- Modified splash screen code to support new project folder structure
- INNO Installer Script modifications due to project folder restructuring
- Modified INNO installer script due to changes in project folder structure
- Moved app icon used for desktop icon under resources
- Updated to use Python 3.4+ pythonic Path(__file__).resolve().parent to fetch execution folder
- Changed from old Python style super(MyDialogClass, self).__init__(parent) to modern Python 3.x style super().__init__(parent)
- Project folder restructured to be more intuitive, modular, and more portable
- Normalized paths to avoid mixture of Linux and Windows file path formats
- Updated version number to 1.0.0.4
- Latest updates to packages used by GRIME AI
- Explicitly specified hydra version_base (currently 1.3.2, only major.minor specified)
- Updated pyInstaller due to deprecation of some required packages (11/30/2025 deprecation date)
- Refactored Mainframe to put USGS code into its own package
- Separated USGS functionality into service and client to abstract USGS website interface from GUI
- Renamed build.bat to grime_ai_build_script_windows.bat (differentiate from new Linux build script)
- Installer no longer needs to copy site_config.json template (GRIME AI creates it if missing)
- JsonEditor class moved into its own file (GRIME_AI_JSON_Editor.py) for improved readability and modularity
- Initial implementation of 'sip' package, upgraded 'lightning' and 'packaging' packages
- Refactored ML code (primary functions getting too large)
- Created Windows Batch file and Linux shell script to compile GRIME AI for respective operating systems
- Cleaned up ROI Analyzer on ML Image Processing dialog with maintenance modifications
- Implemented more streamlined interface to GRIME_AI_QProgressWheel
- Refactored QProgressWheel to self-initialize with title, range, and close callback
- Moved progress bar setup logic from external _make_progress_bar() helper into QProgressWheel.__init__
- Removed code that copies images for model training (now generates JSON file with selected images)
- Image Organizer design changes
- Code formatting cleanup (excessive spaces between variable names and values)

### Fixed
- Force mask to correct dimension if dimension issue detected
- Crash caused by progress bar once NEON images download is complete
- Path issue when performing "create JSON" during model training
- Width and position of slice selection (previously stopped working due to App_QLabel correction for ROI disappearance)
- GitHub sync issue
- Missing import getpass

### Removed
- POC code from repositories
- Custom flash-attn wheel (not ready for prime time release)
- Need for chrome_driver from main.py
- Deprecated processImageMat function (commented out)
- Superfluous icons folder
- Superfluous "Start Time" console output message
- Commented code block that is obsolete
- Deprecated functions from main.py
- Classes and code not pertinent to USGS grant
- Unnecessary code obtaining site information from USGS HIVIS site (speeds up software)

### Developer Notes
- Stop tracking sam2 directory
- Prevent accidental commit of sam2 folder and subfolders
- Ignore sam2 folder on local development computers (cloned and installed from Meta's site as manual step)
- Exclude certain files from "neon" folder
- Excluded experimental folder from git tracking
- Updated to ignore everything in Installer folder except ISS script
- Ignore everything in LoRA folder except two files listed in .gitignore
- Chrome driver functions no longer needed since 2024

## [1.0.0.3-alpha] - 2025-09-21

### Added
- Capability to overlay shape files onto maps
- Nebraska counties shape file as proof-of-concept
- Map pin color variations (USGS sites: green, Phenocam sites: gold, Lincoln NE: red)
- Diagnostic outputs for NEON data fetching issues

### Changed
- Migrated from R neonUtilities to Python neonutilities
- Refactored openstreetmap_viewer.py for multiple pin colors
- Moved graphics and images to resources folder
- Modified splash screen code for new project structure
- Refactored Mainframe to separate USGS functionality

### Fixed
- pyInstaller parameters for neonutilities inclusion in frozen mode
- DNS error handling with explicit console messages

### Removed
- Dependency on R and R Studio
- Need for multiple past project support
- Temporary shape file overlay (pending refinement)

## [1.0.0.2] - 2025-09-16

### Added
- Automated site_config.json template creation on startup

### Changed
- Design improvements to Image Organizer functionality
- Separated JsonEditor class into standalone file
- Site name validation before model training

## [1.0.0.1] - 2025-09-16

### Added
- Initial Image Organizer functionality

## [1.0.0.0] - 2025-09-16

### Added
- First open source public release of GRIME AI

## [0.0.6.0-beta.19] - 2025-09-13

### Added
- Initial Image Organizer functionality

## [0.0.6.0-beta.18] - 2025-09-11

### Changed
- GIF generator progress wheel now reflects write-to-disk operation
- Image Triage and Feature Extraction CSVs display only image names (not full paths) in cells

## [0.0.6.0-beta.17] - 2025-09-09

### Added
- Training folder text field persistence across sessions
- Meaningful names to ROI Analyzer widgets

### Changed
- Optimized ML metric graphs for large datasets
- Turned off axes on mask overlay in segmented images

### Fixed
- Updated packages requiring minor code changes for MIOU plots and model training
- Extract Features code for proper GCC value output
- Fetch Images folder browsing functionality

### Removed
- GRIME_AI_BuildModelDlg.py class (replaced by ML_ImageProcessingDlg.py)
- Classes and code not pertinent to USGS grant

## [0.0.6.0-beta.16] - 2025-08-27

### Changed
- Further optimizations for graph generation during model training

## [0.0.6.0-beta.15] - 2025-08-27

### Changed
- Limited data points plotted in training graphs
- Optimized ROC and Loss graph generation code

## [0.0.6.0-beta.14] - 2025-08-26

### Added
- Five additional metric graphs (total of 7):
  1. Loss curves
  2. Accuracy curves
  3. Confusion matrix
  4. ROC Curve + AUC
  5. Precision-Recall
  6. F1 vs. Threshold
  7. Mean IoU curve

## [0.0.6.0-beta.13] - 2025-08-26

### Fixed
- Crash when loading models with different dictionary formats

## [0.0.6.0-beta.12] - 2025-08-16

### Changed
- Major refactoring of model training for deterministic results
- Fixed erratic model performance and inconsistent segmentation results

## [0.0.6.0-beta.11] - 2025-08-05

### Added
- Sync JSON Annotations tool to verify image-annotation file consistency
- Error management for NEON Field Site Table issues
- Visualization class for metric plots, graphs, and heatmaps
- Annotation file inspection tool (generates XLSX report)

### Changed
- Training data folder validation for image-annotation synchronization
- All model training files now saved to GRIME AI Models subfolder

### Removed
- Intermittent model saving during training
- Deprecated functions from main.py

## [0.0.6.0-beta.10] - 2025-07-31

### Changed
- Eliminated CVAT folder structure requirement (annotation file and images in same folder)
- Reorganized experimental ROI Analyzer tab

### Fixed
- Training function incorrectly checking segmentation options

## [0.0.6.0-beta.8] - 2025-07-22

### Added
- Filmstrip to image navigation dialog
- GCC greenness index generation

### Fixed
- Image navigation dialog spinbox infinite click loop
- USGS site connection reliability (SSL-related)

## [0.0.6.0-beta.7] - 2025-07-13

### Added
- Basic annotation functionality to ML dialog

### Fixed
- ML dialog box re-opening issue

## [0.0.6.0-beta.6] - 2025-07-12

### Added
- Model category embedding (labels and IDs) in .torch files
- COCO 1.0 annotation file generation from images and masks
- Single mask application to multiple images
- Experimental ROI Analyzer tab
- Composite Slice execution from File Utilities dialog

### Changed
- Segmentation now checks models for categories and presents selection options
- Segment Images progress wheel can terminate process on close
- Optional mask saving alongside segmented images

## [0.0.6.0-beta.4] - 2025-06-16

### Added
- Products listing on NEON tab (center placement)

### Changed
- Latest Image moved from center to right on NEON tab
- ML Segmentation and Model Training dialog reformatted/resized
- Site_config.json moved from system folder to user's GRIME AI folder

### Fixed
- Torchvision transforms.py enum issue with pyInstaller

## [0.0.6.0-beta.2] - 2025-05-31

### Added
- SAM2 machine learning integration
- Hyperparameter adjustment dialog interface

### Changed
- Site_config.json location moved to user's Documents folder

## [0.0.5.16] - 2025-05-22

### Fixed
- Replaced lazy iterator with traditional loop control for large image lists (32,000+)

## [0.0.5.15b] - 2025-04-27

### Fixed
- ROI color swatch display in GUI Feature Table
- Correct image display after fetching image list (first instead of last)

## [0.0.5.15] - 2025-04-03

### Changed
- Filter out USGS cameras with 'hideCam' attribute set to True
- Print hidden cameras list to console on startup
- Refactored site information retrieval for improved performance

### Fixed
- Feature extraction spreadsheet output formatting

## [0.0.5.14] - 2025-03-31

### Changed
- Major restructuring and bug fixes to Feature Extraction
- Color display corrections in Feature Extraction panel

## [0.0.5.13] - 2025-03-xx

### Added
- Additional greenness indices from RGB vegetation studies literature
- Dynamic greenness indices column management in GUI

### Changed
- Complete vegetation indices class rewrite
- Normalized Excessive Greenness calculations

### Fixed
- Greenness calculation normalization for lighting variations

## [0.0.5.12a] - 2025-03-12

### Fixed
- Fetch Images issue caused by CLI framework implementation

## [0.0.5.12] - 2025-03-09

### Added
- Basic CLI framework supporting Triage and Composite Slice

## [0.0.5.11f] - 2025-02-24

### Fixed
- BGR to RGB inversion issue in video creation

## [0.0.5.11e] - 2025-02-14

### Added
- CompositeSlice GUI slice dragging (left mouse button)
- CompositeSlice GUI region width adjustment (right mouse button)

### Changed
- Modified installer for toolbar icon fixes

## [0.0.5.11d] - 2025-02-03

### Changed
- Made torch import optional for users without CUDA

## [0.0.5.11c] - 2025-01-13

### Fixed
- Deadlock between Mask Editor and Color Segmentation toolboxes
- QMessageBox window focus behavior

### Added
- CLI diagnostic messages

## [0.0.5.11b] - 2025-01-13

### Changed
- Unified NEON/Phenocam download completion messages

## [0.0.5.11a] - 2025-01-10

### Added
- Image Triage button to Data Explorer dialog

### Changed
- NEON download manager shows single comprehensive report

## [0.0.5.11] - 2025-01-06

### Added
- SAM2 model segmentation integration
- SAM2 model training/tuning with configurable parameters
- Mask saving option alongside segmented images
- Composite Slice execution from File Utilities dialog

### Changed
- Major File Utilities dialog redesign
- Segment Images progress wheel can terminate process
- Create Video/GIF buttons use common folder field

## [0.0.5.10a] - 2024-11-16

### Fixed
- Download issues to non-default folders for USGS and NEON sites

## [0.0.5.10] - 2024-11-14

### Added
- Last folder retention for USGS and NEON downloads

### Fixed
- USGS and NEON site download issues

## [0.0.5.9] - 2024-09-30

### Added
- Verification check for fetched files in Composite Slice

### Fixed
- Fetch files dialog crash from incorrect UI template

## [0.0.5.8] - 2024-09-22

### Added
- COCO annotation mask reconstitution feature

## [0.0.5.7] - 2024-08-08

### Changed
- Rewrote NEON data fetching code for website changes
- Modified installer for OS environment variable handling

## [0.0.5.6] - 2024-07-05

### Added
- Delay in NEON site/product list fetching to fix connection failures

## [0.0.5.5] - 2024-06-26

### Added
- First rendition of Composite Slices functionality

## [0.0.5.4] - 2024-06-25

### Added
- Completed first Composite Slices rendition

## [0.0.5.3] - 2024-06-24

### Changed
- Enhanced Composite Slices functionality

## [0.0.5.2] - 2024-06-16

### Added
- Initial Composite Slices functionality for image slice concatenation

## [0.0.5.1] - 2024-05-30

### Added
- Options to save masks and/or copy original images with segmented output

## [0.0.5.0] - 2024-05-15

### Added
- Initial Segment Anything Model (SAM) integration

## [0.0.4.0c] - 2024-04-24

### Fixed
- NEON site list filename detection after CSV extension removal

## [0.0.4.0b] - 2024-04-19

### Added
- Reference image validation for alignment correction

## [0.0.4.0a] - 2024-04-16

### Added
- ORB feature detection polyline overlays saved to "poly" subfolder
- Rotated images saved to "warp" subfolder for diagnostics

## [0.0.4.0] - 2024-04-09

### Added
- Image rotation/shift detection relative to reference image in triage

## [0.0.3.9] - 2024-02-25

### Changed
- USGS stage data conversion from ASCII to CSV format

## [0.0.3.8g] - 2024-01-21

### Added
- Date/time stamp format handler class for NEON and USGS files

### Fixed
- Duplicate image list fetching causing delays

## [0.0.3.8f] - 2024-01-07

### Fixed
- Shannon Entropy values incorrectly reported as zero

## [0.0.3.8e] - 2023-11-13

### Fixed
- USGS DownloadManager tab refresh on startup
- NEON and USGS DownloadManager column sizing
- Multiple tab refresh issue

## [0.0.3.8d] - 2023-11-06

### Added
- Unverified HTTPS context to resolve SSL certificate verification issues

## [0.0.3.8c] - 2023-10-19

### Added
- Build Feature File button to Color Segmentation dialog

### Changed
- Moved Feature Extraction functions to appropriate classes
- Whole Image extracted features to first ROI row

### Fixed
- Color cluster display in Image Analysis ROI table

## [0.0.3.8b] - 2023-10-19

### Changed
- USGS NIMS image now resizes properly
- Renamed latestImage to NEON_latestImage for consistency

## [0.0.3.8a] - 2023-10-19

### Fixed
- Crash from missing configuration file

## [0.0.3.8] - 2023-10-13

### Added
- Filter to ignore deprecated USGS HIVIS sites

### Changed
- GUI resizing improvements for NEON and USGS site tabs

## [0.0.3.7] - 2023-09-28

### Fixed
- Crash during greenness index calculation

## [0.0.3.6] - 2023-09-24

### Fixed
- Crash when fetching files

## [0.0.3.5] - 2023-09-22

### Fixed
- USGS image availability verification

## [0.0.3.4] - 2023-09-17

### Added
- New date selection for NEON download manager
- NEON data availability verification

### Changed
- General code refactoring

### Fixed
- NEON download issue

## [0.0.3.3] - 2023-08-28

### Changed
- Refactored NEON Download Manager

### Fixed
- NEON data availability verification

## [0.0.3.2] - 2023-07-30

### Added
- USGS Download Manager
- Simultaneous stage and flow rate data download

### Changed
- Continued UI redesign implementation

## [0.0.3.1] - 2023-07-29

### Changed
- Continued UI redesign implementation

## [0.0.3.0] - 2023-02-18

### Changed
- Complete user interface redesign

## [0.0.2.3] - 2022-10-24

### Added
- Enable/disable subfolder recursion feature

## [0.0.2.2] - 2022-10-11

### Added
- Third GUI specific to Feature Extraction

### Changed
- All fonts changed to Calibri for Windows 10/11 compatibility

### Fixed
- Color cluster display in feature table
- Phenocam site reference updated

## [0.0.2.1] - 2022-09-19

### Changed
- GUI modifications for feature extraction

### Fixed
- Bugs reported during unit testing

## [0.0.2.0] - 2022-09-18

### Added
- Limited capability release for ROI information extraction

### Changed
- Image triage speed improvement through decimation
- Hue values now 0-360 degrees (per color space definition)

### Fixed
- CSV output file format for scalars

## [0.0.1.7] - 2022-06-10

### Added
- Limited capability release as NEON/Phenocam Download Manager only

## [0.0.1.6] - 2022-04-06

### Added
- HIVIS Proof-of-Concept

## [0.0.1.5] - 2022-03-06

### Changed
- Code refactoring

### Fixed
- ROI drawing issue
- ROI clustering issue (BGR to RGB conversion for KMeans)

## [0.0.1.4] - 2022-02-27

### Added
- Image quality detection (blurry, too dark, too light)

### Changed
- Regular code refactoring

## [0.0.1.2] - 2022-01-31

### Added
- PBT file folder traversal with date/time filtering

### Changed
- Refactored code
- Miscellaneous fixes

## [0.0.1.1] - 2022-01-20

### Added
- Color segmentation experiments
- Diagnostics functions

### Fixed
- Various code fixes

## [0.0.1.0] - 2021-11-05

### Changed
- Code cleanup
- Minor UI fixes

## [0.0.0.9] - 2021-10-22

### Added
- Local image download option
- Colorized push buttons for better UX

### Changed
- Reorganized Download Site Products tab widgets

## [0.0.0.8] - 2021-10-22

### Added
- Auto-save to GRIME-AI folder in user's Documents
- Separate folders for each NEON product by ID

### Changed
- Released for Platte Basin Timelapse group presentation

## [0.0.0.0] - 2021-04-xx

### Added
- Initial NEON/Phenocam prototype (Blade Vision Systems presentation)
