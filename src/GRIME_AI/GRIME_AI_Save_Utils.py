#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os

from configparser import ConfigParser
import configparser


# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====  class GRIME_AI_Save_Utils  =====     =====     =====     =====    =====
# ======================================================================================================================
# ======================================================================================================================
class GRIME_AI_Save_Utils:
    def __init__(self):
        self.className = "GRIME_AI_Save_Utils"

        self.configFile = self.get_settings_folder()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_users_GRIME_AI_folder(self):
        models_file_path = os.path.join(os.path.expanduser('~'), 'Documents', 'GRIME-AI')

        if not os.path.exists(models_file_path):
            os.makedirs(models_file_path)

        return models_file_path

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_phenocam_folder(self):
        phenocam_file_path = os.path.join(os.path.expanduser('~'), 'Documents', 'GRIME-AI', 'phenocam')

        if not os.path.exists(phenocam_file_path):
            os.makedirs(phenocam_file_path)

        return phenocam_file_path

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_models_folder(self):
        models_file_path = os.path.join(os.path.expanduser('~'), 'Documents', 'GRIME-AI', 'Models')

        if not os.path.exists(models_file_path):
            os.makedirs(models_file_path)

        return models_file_path


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_artifacts_folder(self):
        models_file_path = os.path.join(os.path.expanduser('~'), 'Documents', 'GRIME-AI', 'Artifacts')

        if not os.path.exists(models_file_path):
            os.makedirs(models_file_path)

        return models_file_path

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # RECIPE OVERRIDE — honor the active recipe's output folder (Recipe Manager)
    # ------------------------------------------------------------------------------------------------------------------
    def _recipe_override(self, json_key: str):
        """Return the active recipe's folder for `json_key` (as written by the
        Recipe Manager on 'Set Active'), or None if no recipe folder is set.
        Falls back silently so behavior is unchanged when no recipe is active."""
        try:
            from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor
            value = JsonEditor().getValue(json_key)
        except Exception:
            value = None
        return value.strip() if isinstance(value, str) and value.strip() else None


    def create_composite_slices_folder(self, image_file_folder: str) -> str:
        """
        Given any image_file_folder path, builds and creates:
          <artifacts_base>/<last_folder_of_image_file_folder>/compositeSlices

        Returns the full path to that compositeSlices folder.
        """
        # 1. Normalize the incoming path and grab its last segment
        normalized = os.path.normpath(image_file_folder)
        #JES last_folder = Path(normalized).name
        #JES if not last_folder:
        #JES     raise ValueError(f"Could not determine last folder from '{image_file_folder}'")

        # 2. Fetch your artifacts base
        # PROVISIONAL
        #JES artifacts_base = self.get_artifacts_folder()
        #JES if not artifacts_base:
        #JES     raise RuntimeError("get_artifacts_folder() returned an empty path")

        # 3. Build the complete compositeSlices path
        #JES target = Path(artifacts_base) / last_folder / "CompositeSlices"
        # Prefer the active recipe's Composite Slices folder; otherwise fall
        # back to the default under Documents/GRIME-AI (not the images folder).
        target = self._recipe_override("Composite_Slices_Folder") \
            or os.path.join(self.get_users_GRIME_AI_folder(), "composite_slices")

        # 4. Create directories if missing (equivalent to mkdir -p)
        os.makedirs(target, exist_ok=True)

        return str(target)


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_video_folder(self, video_file_folder: str) -> str:
        """
        Given any image_file_folder path, builds and creates:
          <artifacts_base>/<last_folder_of_image_file_folder>/Videos

        Returns the full path to the videos folder.
        """
        # 1. Normalize the incoming path and grab its last segment
        normalized = os.path.normpath(video_file_folder)
        #JES last_folder = Path(normalized).name
        #JES if not last_folder:
        #JES     raise ValueError(f"Could not determine last folder from '{video_file_folder}'")

        # 2. Fetch your artifacts base
        #JES artifacts_base = self.get_artifacts_folder()
        #JES if not artifacts_base:
        #JES     raise RuntimeError("get_artifacts_folder() returned an empty path")

        # 3. Build the complete compositeSlices path
        #JES target = Path(artifacts_base) / last_folder / "Videos"
        # Prefer the active recipe's Videos folder; otherwise fall back to the
        # default under Documents/GRIME-AI (not the images folder).
        target = self._recipe_override("Videos_Folder") \
            or os.path.join(self.get_users_GRIME_AI_folder(), "videos")

        # 4. Create directories if missing (equivalent to mkdir -p)
        #JES target.mkdir(parents=True, exist_ok=True)
        os.makedirs(target, exist_ok=True)

        return str(target)


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_gif_folder(self, gif_file_folder: str) -> str:
        """
        Given any image_file_folder path, builds and creates:
          <artifacts_base>/<last_folder_of_image_file_folder>/gif

        Returns the full path to the gif folder.
        """
        # 1. Normalize the incoming path and grab its last segment
        normalized = os.path.normpath(gif_file_folder)
        #JES last_folder = Path(normalized).name
        #JES if not last_folder:
        #JES     raise ValueError(f"Could not determine last folder from '{gif_file_folder}'")

        # 2. Fetch your artifacts base
        #JES artifacts_base = self.get_artifacts_folder()
        #JES if not artifacts_base:
        #JES     raise RuntimeError("get_artifacts_folder() returned an empty path")

        # 3. Build the complete compositeSlices path
        #JES target = Path(artifacts_base) / last_folder / "gif"
        # Prefer the active recipe's GIFs folder; otherwise fall back to the
        # default under Documents/GRIME-AI (not the images folder).
        target = self._recipe_override("GIFs_Folder") \
            or os.path.join(self.get_users_GRIME_AI_folder(), "gifs")

        # 4. Create directories if missing (equivalent to mkdir -p)
        #JES target.mkdir(parents=True, exist_ok=True)
        os.makedirs(target, exist_ok=True)

        return str(target)


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_segmented_images_folder(self, segmented_images_file_folder: str) -> str:
        """
        Given any image_file_folder path, builds and creates:
          <artifacts_base>/<last_folder_of_image_file_folder>/Segmented_Images

        Returns the full path to the segmented images folder.
        """
        # 1. Normalize the incoming path and grab its last segment
        normalized = os.path.normpath(segmented_images_file_folder)
        #JES last_folder = Path(normalized).name
        #JES if not last_folder:
        #JES     raise ValueError(f"Could not determine last folder from '{segmented_images_file_folder}'")

        # 2. Fetch your artifacts base
        #JES artifacts_base = self.get_artifacts_folder()
        #JES if not artifacts_base:
        #JES     raise RuntimeError("get_artifacts_folder() returned an empty path")

        # 3. Build the complete compositeSlices path
        #JES target = Path(artifacts_base) / last_folder / "Segmented_Images"
        target = os.path.join(normalized, "Segmented_Images")

        # 4. Create directories if missing (equivalent to mkdir -p)
        #JES target.mkdir(parents=True, exist_ok=True)
        os.makedirs(target, exist_ok=True)

        return str(target)


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_settings_folder(self):
        configFilePath = os.path.join(os.path.expanduser('~'), 'Documents', 'GRIME-AI', 'Settings')

        if not os.path.exists(configFilePath):
            os.makedirs(configFilePath)

        return configFilePath


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def read_config_file(self):
        configFilePath = self.get_settings_folder()
        configFile = os.path.join(configFilePath, self.json_filename)

        config_object = configparser.ConfigParser()
        with open(configFile, "r") as file_object:
            config_object.read_file(file_object)


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def saveSettings(self):
        config = ConfigParser()

        configFilePath = self.get_settings_folder()
        configFile = os.path.join(configFilePath, 'GRIME-AI.cfg')

        with open(configFile, 'w+') as f:
            config.read(configFile)

        szSection = 'ROI'
        config.add_section(szSection)
        config.set(szSection, 'Sky', 'TBD')
        config.set(szSection, 'Grass1', 'TBD')
        config.set(szSection, 'Grass2', 'TBD')
        config.set(szSection, 'Trees1', 'TBD')
        config.set(szSection, 'Trees2', 'TBD')

        szSection = 'FilePaths'
        config.add_section(szSection)

        config.write(f)
        f.close()

    # ------------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------------
    '''
    def save_images_folder_path(self, images_folder_path):
        configFilePath = self.get_settings_folder()

        config = ConfigParser()

        configFile = os.path.join(configFilePath, 'GRIME-AI.cfg')
        f = open(configFile, 'w+')

        config.read(configFile)

        szSection = 'Local_Images_Folder'
        config.add_section(szSection)
        config.set(szSection, 'ImagesFilePath', images_folder_path)

        config.write(f)
        f.close()
    '''

    # ------------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------------
    '''
    def USGS_SaveFolderPath(self, USGS_FolderPath):
        configFilePath = self.get_settings_folder()

        config = ConfigParser()

        configFile = os.path.join(configFilePath, 'GRIME-AI.cfg')
        f = open(configFile, 'w+')

        config.read(configFile)

        szSection = 'USGS'
        config.add_section(szSection)
        config.set(szSection, 'SaveFilePath', USGS_FolderPath)

        config.write(f)
        f.close()
    '''


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    '''
    def USGS_getSaveFolderPath(self):
    
        try:
            configFilePath = self.get_settings_folder()

            config = ConfigParser()

            configFile = os.path.join(configFilePath, 'GRIME-AI.cfg')
            f = open(configFile, 'r')

            config.read(configFile)

            szSection = 'USGS'
            USGS_FolderPath = config.get(szSection, 'SaveFilePath')

            f.close()
        except Exception:
            USGS_FolderPath = ""

        return USGS_FolderPath
    '''


    # ------------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------------
    '''
    def NEON_getSaveFolderPath(self):

        try:
            configFilePath = self.get_settings_folder()

            config = ConfigParser()

            configFile = os.path.join(configFilePath, 'GRIME-AI.cfg')
            f = open(configFile, 'r')

            config.read(configFile)

            szSection = 'NEON'
            NEON_FolderPath = config.get(szSection, 'SaveFilePath')

            f.close()
        except Exception:
            NEON_FolderPath = ""

        return NEON_FolderPath
    '''
