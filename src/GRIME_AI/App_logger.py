#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import sys

# ---------- logging helpers ----------
def debug(msg: str): print(f"[DEBUG] {msg}")
def info(msg: str): print(f"[INFO]  {msg}")
def warn(msg: str): print(f"[WARN]  {msg}")
def err(msg: str): print(f"[ERROR] {msg}", file=sys.stderr)