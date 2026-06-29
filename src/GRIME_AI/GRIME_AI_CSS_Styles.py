#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author(s):  John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Sep. 15, 2025
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

BUTTON_CSS_STEEL_BLUE = """
QPushButton {
  background-color: steelblue;
  color: white;
  border: 1px solid #3b6a93;
  padding: 6px 14px;
  border-radius: 6px;
}
QPushButton:hover { background-color: #5a93c2; }
QPushButton:disabled {
  background-color: gray;
  color: black;
}"""

BUTTON_CSS_DARK_RED = """
QPushButton {
  background-color: darkred;
  color: white;
  border: 1px solid #3b6a93;
  padding: 6px 14px;
  border-radius: 6px;
}
QPushButton:hover { background-color: #5a93c2; }
QPushButton:disabled {
  background-color: gray;
  color: black;
}"""

BUTTON_CSS_GRAY = """
QPushButton {
  background-color: gray;
  color: black;
  border: 1px solid #666666;
  padding: 6px 14px;
  border-radius: 6px;
}
QPushButton:hover { background-color: #808080; }
QPushButton:disabled {
  background-color: gray;
  color: black;
}
"""
BUTTON_CSS_ORANGE = """
QPushButton {
  background-color: darkorange;
  color: white;
  border: 1px solid #b8600a;
  padding: 6px 14px;
  border-radius: 6px;
}
QPushButton:hover { background-color: #e8872a; }
QPushButton:disabled {
  background-color: gray;
  color: black;
}"""

BUTTON_CSS_YELLOW = """
QPushButton {
  background-color: #f0c000;
  color: black;
  border: 1px solid #b89000;
  padding: 6px 14px;
  border-radius: 6px;
}
QPushButton:hover { background-color: #f5d030; }
QPushButton:disabled {
  background-color: gray;
  color: black;
}"""

BUTTON_CSS_RED_OUTLINE = """
QPushButton {
  background-color: transparent;
  color: darkred;
  border: 2px solid darkred;
  padding: 6px 14px;
  border-radius: 6px;
}
QPushButton:hover { background-color: rgba(139, 0, 0, 0.08); }
QPushButton:disabled {
  background-color: transparent;
  color: gray;
  border: 2px solid gray;
}"""

BUTTON_CSS_YELLOW_OUTLINE = """
QPushButton {
  background-color: transparent;
  color: #b89000;
  border: 2px solid #f0c000;
  padding: 6px 14px;
  border-radius: 6px;
}
QPushButton:hover { background-color: rgba(240, 192, 0, 0.08); }
QPushButton:disabled {
  background-color: transparent;
  color: gray;
  border: 2px solid gray;
}"""
