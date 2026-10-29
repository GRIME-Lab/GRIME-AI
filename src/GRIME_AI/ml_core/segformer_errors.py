#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# segformer_errors.py


class SegFormerError(Exception):
    """
    Raised when SegFormer training or inference cannot continue.

    The message states what is wrong in plain terms. Callers catch it, close
    any progress window, show the message to the user, and stop the run.
    """
    pass
