#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

"""
app_identity.py
---------------
The application name is declared once, here. Every user-facing folder, file name
and title is derived from it. Other modules import these values instead of
hard-coding the name.

Derived values reproduce the names already on users' disks, so existing settings,
models and recipes are found unchanged.
"""

from pathlib import Path

# ----------------------------------------------------------------------------------------------------------------------
# THE ONLY PLACE THE APPLICATION NAME IS DECLARED
# ----------------------------------------------------------------------------------------------------------------------
APP_NAME = "GRIME-AI"

# ----------------------------------------------------------------------------------------------------------------------
# DERIVED NAME FORMS
# ----------------------------------------------------------------------------------------------------------------------
APP_ID = APP_NAME.replace("-", "_")              # GRIME_AI   (file-name / identifier form)
APP_DISPLAY_NAME = APP_NAME.replace("-", " ")    # GRIME AI   (titles and messages)

# ----------------------------------------------------------------------------------------------------------------------
# USER FOLDERS  (<home>/Documents/<APP_NAME>/...)
# ----------------------------------------------------------------------------------------------------------------------
USER_ROOT = Path.home() / "Documents" / APP_NAME
SETTINGS_DIR = USER_ROOT / "Settings"
MODELS_DIR = USER_ROOT / "Models"
ARTIFACTS_DIR = USER_ROOT / "Artifacts"
DOWNLOADS_DIR = USER_ROOT / "Downloads"
SCRATCHPAD_DIR = USER_ROOT / "Scratchpad"
PLUGINS_DIR = USER_ROOT / "plugins"

# ----------------------------------------------------------------------------------------------------------------------
# USER FILES  (in SETTINGS_DIR)
# ----------------------------------------------------------------------------------------------------------------------
APP_CONFIG_FILENAME = f"{APP_NAME}.json"         # GRIME-AI.json
APP_CFG_FILENAME = f"{APP_NAME}.cfg"             # GRIME-AI.cfg
APP_RECIPES_FILENAME = f"{APP_ID}_Recipes.json"  # GRIME_AI_Recipes.json

# ----------------------------------------------------------------------------------------------------------------------
# PRODUCT-SPECIFIC VALUES THAT CANNOT BE DERIVED FROM THE NAME
# ----------------------------------------------------------------------------------------------------------------------
APP_LOGO_FILENAME = "GRIME-AI Logo with Tagline.png"   # in resources/splash_screens
APP_REPO_URL = "https://github.com/JohnStranzl/GRIME-AI"
