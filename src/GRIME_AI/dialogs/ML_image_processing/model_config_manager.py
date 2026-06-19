import shutil
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union
from datetime import datetime

# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===        class ModelConfigManager        ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
class ModelConfigManager:
    def __init__(self, filepath: str = None):
        self.filepath = filepath
        self.config: Dict[str, Any] = {}

    # ======================================================================================================================
    #  =====     =====     =====     =====     =====    CLASS LEVEL HELPERS     =====     =====     =====     =====     =====
    # ======================================================================================================================

    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """
        Flattens a nested dictionary for DataFrame conversion.
        Handles lists by enumerating indices.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ModelConfigManager._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                if all(isinstance(i, dict) for i in v):
                    # Flatten list of dicts with index
                    for idx, subdict in enumerate(v):
                        items.extend(ModelConfigManager._flatten_dict(subdict, f"{new_key}[{idx}]", sep=sep).items())
                else:
                    # Store list as-is
                    items.append((new_key, v))
            else:
                items.append((new_key, v))
        return dict(items)

    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def set_defaults(config: Dict[str, Any], overwrite: bool = True) -> Dict[str, Any]:
        """
        Sets default values for core configuration keys (training + inference).
        If overwrite=False, only fills in missing or empty values.
        """
        defaults = {
            "learningRates": [0.0001],
            "optimizer": "Adam",
            "loss_function": "IOU",
            "weight_decay": 0.01,
            "number_of_epochs": 5,
            "batch_size": 32,
            "use_lora": False,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_bias": "none",
            "lora_target_modules": ["query", "key", "value"],
            "save_model_frequency": 20,
            "validation_frequency": 20,
            "early_stopping": False,
            "patience": 3,
            "device": "gpu",
            "save_model_masks": True,
            "copy_original_model_image": True,
            "num_clusters": 3,
            "validation_overlay_samples": 5,
            "val_split": 0.2,
            "yolo_base_weights": "",
            "SAM2_CHECKPOINT": "checkpoints/sam2.1_hiera_large.pt",
            "MODEL_CFG": "configs/sam2.1/sam2.1_hiera_l.yaml"
        }
        for key, value in defaults.items():
            if overwrite or key not in config or config[key] in (None, "", []):
                config[key] = value

        # Handle nested load_model defaults
        if "load_model" in config:
            if overwrite or not config["load_model"].get("SAM2_CHECKPOINT"):
                config["load_model"]["SAM2_CHECKPOINT"] = defaults["SAM2_CHECKPOINT"]
            if overwrite or not config["load_model"].get("MODEL_CFG"):
                config["load_model"]["MODEL_CFG"] = defaults["MODEL_CFG"]

        return config


    # ======================================================================================================================
    #  =====     =====     =====     =====     =====        CLASS FUNCTIONS        =====     =====     =====     =====     =====
    # ======================================================================================================================

    @staticmethod
    def create_template() -> Dict[str, Any]:
        """
        Creates a template of the JSON file with the same structure
        but empty/default values.
        """

        # TRY TO IMPORT THE VERSION CONSTANT FROM VERSION.PY
        # IF IT DOESN'T EXIST, WE CREATE THE TEMPLATE WITHOUT A VERSION NUMBER
        # IF THE KEY DOESN'T EXIST, OR THE VERSION NUMBER DOESN'T EXIST WHEN THE
        # CONFIG FILES IS READ, THE ASSUMPTION IS THAT THE CONFIG FILE IS OF OLDER
        # VINTAGE AND MAY NOT BE COMPATIBLE WITH THE LATEST SOFTWARE.
        try:
            from GRIME_AI.version import SW_VERSION
        except ImportError:
            # FALLBACK IF VERSION.PY DOES NOT EXIST. 0.0.0.0 IS AN INVALID VERSION NUMBER
            SW_VERSION = ""

        return {
            "version": SW_VERSION,
            "siteName": "",
            "learningRates": [0.0001],
            "optimizer": "Adam",
            "loss_function": "IOU",
            "weight_decay": 0.01,
            "number_of_epochs": 20,
            "batch_size": 32,
            "use_lora": False,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_bias": "none",
            "lora_target_modules": ["query", "key", "value"],
            "save_model_frequency": 5,
            "validation_frequency": 5,
            "early_stopping": False,
            "patience": 3,
            "device": "gpu",
            "folder_path": "",
            "available_folders": [],
            "selected_folders": [],
            "save_model_masks": True,
            "copy_original_model_image": True,
            "save_probability_maps": True,
            "save_diagnostic_panels": True,
            "num_clusters": 3,
            "validation_overlay_samples": 5,
            "val_split": 0.2,
            "yolo_base_weights": "",
            "holdout_seasons": [],
            "blob_filter_radius": 0.0,
            "blob_radius_mode": "Manual",
            "segmentation_images_path": "",
            "load_model": {
                "SAM2_CHECKPOINT": "checkpoints/sam2.1_hiera_large.pt",
                "MODEL_CFG": "configs/sam2.1/sam2.1_hiera_l.yaml",
                "MODEL": "",
                "segmentation_images_path": "",
                "segmentation_image_folders": [],
                "predictions_output_path": "",
                "output_folder": "",
                "segment_seasons": [],
                "SAM2_MODEL": "",
                "SAM3_MODEL": "",
                "SEGFORMER_MODEL": "",
                "YOLO_MODEL": "",
                "SEGMENTATION_CATEGORIES": []
            },
            "train_model": {
                "TRAINING_CATEGORIES": []
            },
            "Path": [
                {
                    "siteName": "",
                    "directoryPaths": {
                        "folders": "",
                        "annotations": ""
                    }
                }
            ]
        }

    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    def save_template(self, output_path: str, apply_defaults: bool = False):
        """
        Saves the template JSON to a file. Optionally applies defaults.
        """
        template = self.create_template()
        if apply_defaults:
            template = self.set_defaults(template)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=4)

    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    def save_config(self, output_path: str):
        """
        Saves the current config to a file.
        """
        if not self.config:
            raise ValueError("No configuration loaded or created to save.")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)

    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    def load_config(self, return_type: str = "dict") -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Loads the JSON config file and returns either a dictionary or a DataFrame.
        If the file is empty or invalid, writes a fresh template to disk and returns it.

        Parameters:
            return_type (str): "dict" (default) or "dataframe"

        Returns:
            dict or pandas.DataFrame
        """
        if not self.filepath:
            raise ValueError("No filepath provided.")

        config_path = Path(self.filepath)
        template = self.create_template()

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    # Empty file → write template
                    self.config = template
                    with open(config_path, "w", encoding="utf-8") as wf:
                        json.dump(self.config, wf, indent=4)
                else:
                    self.config = json.loads(content)
        except FileNotFoundError:
            # No file → create parent dirs (if needed), write template
            if config_path.parent and not config_path.parent.exists():
                config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config = template
            with open(config_path, "w", encoding="utf-8") as wf:
                json.dump(self.config, wf, indent=4)
        except json.JSONDecodeError:
            # Invalid JSON → write template
            self.config = template
            with open(config_path, "w", encoding="utf-8") as wf:
                json.dump(self.config, wf, indent=4)

        if return_type == "dict":
            return self.config
        elif return_type == "dataframe":
            flat_config = self._flatten_dict(self.config)
            return pd.DataFrame([flat_config])
        else:
            raise ValueError("return_type must be 'dict' or 'dataframe'.")

    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    def update_config(self, values: Dict[str, Any], save: bool = True):
        """
        Update self.config with values from UI controls and optionally save to JSON file.

        Parameters:
            values (dict): Dictionary of values collected from UI.
            save (bool): If True, persist changes to the JSON file at self.filepath.
        """
        # Always start from template to preserve key ordering
        base = self.create_template()

        # Merge existing config into template
        if self.config:
            for key, val in self.config.items():
                base[key] = val

        # Overlay new values from UI
        for key, val in values.items():
            base[key] = val

        # Update manager state
        self.config = base

        # Save to file if requested
        if save:
            if not self.filepath:
                raise ValueError("No filepath provided to save updated config.")
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)

    # --------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------
    def backup_config(self) -> Dict[str, Any]:
        """
        Backup the existing config file (if present) and load its contents.
        Returns the loaded config dictionary, or {} if no file or load fails.
        """
        if not self.filepath:
            raise ValueError("No filepath provided for backup.")

        config_path = Path(self.filepath).resolve()
        site_config = {}

        if config_path.exists():
            now_str = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
            backup_file = config_path.with_name(f"{now_str}_site_config.json")
            shutil.copy(config_path, backup_file)
            print(f"Existing {config_path} backed up to {backup_file}")

            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    site_config = json.load(f)
            except Exception as e:
                print(f"Failed to load existing config, starting fresh: {e}")
                site_config = {}
        else:
            print("No existing config found; starting fresh.")

        # Update manager state
        self.config = site_config
        return site_config
