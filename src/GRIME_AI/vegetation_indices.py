#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import math
from typing import Final

from GRIME_AI.App_Utils import App_Utils


'''
https://www.plantsciencejournal.com/apdf/jpsp-aid1124.pdf

Green Chromatic Coordinate (GCC): GCC = G / R + G + B
This measures the proportion of green reflectance relative to the total RGB reflectance.

Excess Green Index (ExG): ExG = 2G - (R + B)
This enhances the green component while minimizing the red and blue components.

Green Leaf Index (GLI): GLI = (2G - R - B) / (2G + R + B)
This index is used to assess vegetation health and growth.

Normalized Green-Red Difference Index (NGRDI): NGRDI = (G - R) / (G + R)
This compares the green and red reflectance to highlight vegetation.

Triangular Greenness Index (TGI): TGI = -0.5 * (190*(R - G) - 120*(R - B)
This index uses a triangular approach to quantify greenness.
'''

# **********************************************************************************************************************
#
# **********************************************************************************************************************
class GreennessIndex:
    def __init__(self, name, value=-999.999):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"GreennessIndex(name={self.name!r}, value={self.value!r})"

    def get_name(self):
        return self.name

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

# **********************************************************************************************************************
#
# **********************************************************************************************************************
class Vegetation_Indices:
    # CONSTANTS
    BCC: Final[str] = "BCC"
    BGI: Final[str] = "BGI"
    BI: Final[str] = "BI"
    BRVI: Final[str] = "BRVI"
    CIVE: Final[str] = "CIVE"
    ExB: Final[str] = "ExB"
    ExG: Final[str] = "ExG"
    ExGR: Final[str] = "ExGR"
    ExR: Final[str] = "ExR"
    GCC: Final[str] = "GCC"
    GLI: Final[str] = "GLI"
    GR: Final[str] = "GR"
    GRVI: Final[str] = "GRVI"
    HI: Final[str] = "HI"
    HUE: Final[str] = "HUE"
    IKAW: Final[str] = "IKAW"
    IOR: Final[str] = "IOR"
    IPCA: Final[str] = "IPCA"
    MGRVI: Final[str] = "MGRVI"
    MPRI: Final[str] = "MPRI"
    MVARI: Final[str] = "MVARI"
    NDI: Final[str] = "NDI"
    NDVI: Final[str] = "NDVI"
    NGBDI: Final[str] = "NGBDI"
    NGRDI: Final[str] = "NGRDI"
    RCC: Final[str] = "RCC"
    RGBVI: Final[str] = "RGBVI"
    RGI: Final[str] = "RGI"
    PRI: Final[str] = "PRI"
    SAVI: Final[str] = "SAVI"
    SCI: Final[str] = "SCI"
    SI: Final[str] = "SI"
    TGI: Final[str] = "TGI"
    VARI: Final[str] = "VARI"
    VDVI: Final[str] = "VDVI"
    VEG: Final[str] = "VEG"
    VIgreen: Final[str] = "VIgreen"
    vNDVI: Final[str] = "vNDVI"
    WI: Final[str] = "WI"

    def __init__(self):
        self.className = "Vegetation_Indices"


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compute_GCC(self, red_sum, green_sum, blue_sum):
        '''
        GCC= (Gdn)/(Rdn + Gdn + Bdn)

        where 'dn' refers to the digital number (value) of the pixel for the specific
        channel. e.g., Rdn corresponds to the digital number or, pixel value, for the red channel.
        (Gillespie et al., 1987)
        '''

        try:
            denominator = red_sum + green_sum + blue_sum
            if denominator > 0.0:
                GCC = green_sum / (red_sum + green_sum + blue_sum)
            else:
                GCC = 0.0
        except Exception as e:
            print(f"Error computing GCC: {e}")
            GCC = -999

        return GCC


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compute_GLI(self, red_sum, green_sum, blue_sum):

        try:
            GLI = ((2.0 * green_sum) - red_sum - blue_sum) / ((2.0 * green_sum) + red_sum + blue_sum)
        except ValueError:
            GLI = -999

        return GLI


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compute_NGDRI(self, red_sum, green_sum, blue_sum):
        '''
        Normalized Green-Red Difference Index (NGRDI): NGRDI = (G - R) / (G + R)
        '''
        try:
            NGRDI = (green_sum - red_sum) / (green_sum + red_sum)
        except ValueError:
            NGRDI = -999

        return(NGRDI)


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compute_NDVI(self, red_sum, green_sum, blue_sum):
        '''
        True NDVI:
        NDVI is calculated using the formula: (NIR - Red) / (NIR + Red), where NIR is the near-infrared band and Red is
        the red band.

        False NDVI (using RGB):
        The idea is to use a similar formula but with RGB values, treating the green band as a proxy for NIR. A common
        formula is: (Green - Red) / (Green + Red - Blue).
        '''

        try:
            NDVI = (green_sum - red_sum) / (green_sum + red_sum + blue_sum)
        except ValueError:
            NDVI = -999

        return(NDVI)


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compute_RGI(self, red_sum, green_sum, blue_sum):

        '''
        The Relative Greenness Index (RGI) is typically used in remote sensing and vegetation studies to assess
        the "greenness" of vegetation in a given area. The specific formula for the RGI may vary depending on the
        application, but a common version involves the following equation:
                𝑅𝐺𝐼 = 𝐺 / (𝑅 + 𝐵)

        Where:
            G: Reflectance in the green spectral band
            R: Reflectance in the red spectral band
            B: Reflectance in the blue spectral band

            This index measures how relatively green an area is by comparing the green reflectance to the total reflectance across red, blue, and green bands. It's often used as a simple indicator of vegetation health or coverage.
        '''

        try:
            RGI = green_sum / (red_sum + green_sum + blue_sum)
        except ValueError:
            RGI = -999

        return RGI

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compute_Blue_Chromatic_Coordinate_Index(self, red_sum, green_sum, blue_sum):
        try:
            BCC = blue_sum / (red_sum + green_sum + blue_sum)
        except ValueError:
            BCC = -999.0
        return BCC

    def compute_Simple_Blue_Green_Ratio(self, blue_sum, green_sum):
        try:
            BGI = blue_sum / green_sum
        except ValueError:
            BGI = -999.0
        return BGI

    def compute_Brightness_Index(self, red_sum, green_sum, blue_sum):
        try:
            BI = ((red_sum**2 + green_sum**2 + blue_sum**2) / 3)**0.5
        except ValueError:
            BI = -999.0
        return BI

    def compute_Blue_Red_Vegetation_Index(self, blue_sum, red_sum):
        try:
            BRVI = (blue_sum - red_sum) / (blue_sum + red_sum)
        except ValueError:
            BRVI = -999.0
        return BRVI

    def compute_Colour_Index_of_Vegetation(self, red_sum, green_sum, blue_sum):
        try:
            CIVE = 0.441 * (red_sum / (red_sum + green_sum + blue_sum)) - \
                   0.881 * (green_sum / (red_sum + green_sum + blue_sum)) + \
                   0.385 * (blue_sum / (red_sum + green_sum + blue_sum)) + 18.78745
        except ValueError:
            CIVE = -999.0
        return CIVE

    def compute_Excess_Blue(self, blue_sum, green_sum):
        try:
            ExB = 1.4 * blue_sum - green_sum
        except ValueError:
            ExB = -999.0
        return ExB

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compute_ExG(self, red_sum, green_sum, blue_sum):
        try:
            r = red_sum / (red_sum + green_sum + blue_sum)
            g = green_sum / (red_sum + green_sum + blue_sum)
            b = blue_sum / (red_sum + green_sum + blue_sum)

            ExG = (2.0 * g) - (r + b)
        except ValueError:
            ExG = -999

        return ExG


    def compute_Excess_Green(self, green_sum, red_sum, blue_sum):
        try:

            '''
            This normalization ensures that lighting variations are minimized by calculating the proportion of each
            channel relative to the total intensity.
            '''
            r = red_sum / (red_sum + green_sum + blue_sum)
            g = green_sum / (red_sum + green_sum + blue_sum)
            b = blue_sum / (red_sum + green_sum + blue_sum)

            ExG = (2.0 * g) - (r + b)
        except ValueError:
            ExG = -999

        return ExG

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compute_Excess_Green_Excess_Red(self, red_sum, green_sum, blue_sum):
        '''
        ExGR = ExG - ExR, with ExG = 2g - r - b and ExR = 1.4r - g, where r, g, b are
        chromatic coordinates (channel / (R + G + B)). (Meyer and Neto, 2008)
        '''
        try:
            total = red_sum + green_sum + blue_sum
            r = red_sum / total
            g = green_sum / total
            b = blue_sum / total
            ExGR = ((2.0 * g) - r - b) - ((1.4 * r) - g)
        except ValueError:
            ExGR = -999.0
        return ExGR

    def compute_Excess_Red(self, red_sum, green_sum):
        try:
            ExR = 1.4 * red_sum - green_sum
        except ValueError:
            ExR = -999.0
        return ExR

    def compute_Green_Percentage_Index(self, red_sum, green_sum, blue_sum):
        try:
            GCC = green_sum / (red_sum + green_sum + blue_sum)
        except ValueError:
            GCC = -999.0
        return GCC

    def compute_Green_Leaf_Index(self, red_sum, green_sum, blue_sum):
        try:
            GLI = ((2.0 * green_sum) - red_sum - blue_sum) / ((2.0 * green_sum) + red_sum + blue_sum)
        except ValueError:
            GLI = -999.0
        return GLI

    def compute_Simple_Red_Green_Ratio(self, green_sum, red_sum):
        try:
            GR = green_sum / red_sum
        except ValueError:
            GR = -999.0
        return GR

    def compute_Green_Red_Vegetation_Index(self, green_sum, red_sum):
        try:
            GRVI = (green_sum - red_sum) / (green_sum + red_sum)
        except ValueError:
            GRVI = -999.0
        return GRVI

    def compute_Primary_Colours_Hue_Index(self, red_sum, green_sum, blue_sum):
        try:
            HI = (2 * red_sum - green_sum - blue_sum) / (green_sum - blue_sum)
        except ValueError:
            HI = -999.0
        return HI

    def compute_Overall_Hue_Index(self, red_sum, green_sum, blue_sum):
        try:
            HUE = math.atan(2 * (blue_sum - green_sum - red_sum) / (3.5 * (green_sum - red_sum)))
        except ValueError:
            HUE = -999.0
        return HUE

    def compute_Kawashima_Index(self, red_sum, blue_sum):
        try:
            IKAW = (red_sum - blue_sum) / (red_sum + blue_sum)
        except ValueError:
            IKAW = -999.0
        return IKAW

    def compute_Iron_Oxide_Ratio(self, red_sum, blue_sum):
        try:
            IOR = red_sum / blue_sum
        except ValueError:
            IOR = -999.0
        return IOR

    def compute_Principal_Component_Analysis_Index(self, red_sum, green_sum, blue_sum):
        try:
            IPCA = 0.994 * abs(red_sum - blue_sum) + \
                   0.961 * abs(green_sum - blue_sum) + \
                   0.914 * abs(green_sum - red_sum)
        except ValueError:
            IPCA = -999.0
        return IPCA

    def compute_Modified_Green_Red_Vegetation_Index(self, green_sum, red_sum):
        try:
            MGRVI = (green_sum**2 - red_sum**2) / (green_sum**2 + red_sum**2)
        except ValueError:
            MGRVI = -999.0
        return MGRVI

    def compute_Modified_Photochemical_Reflectance_Index(self, green_sum, red_sum):
        try:
            MPRI = (green_sum - red_sum) / (green_sum + red_sum)
        except ValueError:
            MPRI = -999.0
        return MPRI

    def compute_Modified_Visible_Atmospherically_Resistant_Vegetation_Index(self, green_sum, red_sum, blue_sum):
        try:
            MVARI = (green_sum - blue_sum) / (green_sum + red_sum - blue_sum)
        except ValueError:
            MVARI = -999.0
        return MVARI

    def compute_Normalized_Difference_Index(self, green_sum, red_sum):
        try:
            NDI = 128 * (((green_sum - red_sum) / (green_sum + red_sum)) + 1)
        except ValueError:
            NDI = -999.0
        return NDI

    def compute_Normalized_Green_Blue_Difference_Index(self, green_sum, blue_sum):
        try:
            NGBDI = (green_sum - blue_sum) / (green_sum + blue_sum)
        except ValueError:
            NGBDI = -999.0
        return NGBDI

    def compute_Normalized_Green_Red_Difference_Index(self, green_sum, red_sum):
        try:
            NGRDI = (green_sum - red_sum) / (green_sum + red_sum)
        except ValueError:
            NGRDI = -999.0
        return NGRDI

    def compute_Red_Chromatic_Coordinate_Index(self, red_sum, green_sum, blue_sum):
        try:
            RCC = red_sum / (red_sum + green_sum + blue_sum)
        except ValueError:
            RCC = -999.0
        return RCC

    def compute_Red_Green_Blue_Vegetation_Index(self, red_sum, green_sum, blue_sum):
        try:
            RGBVI = (green_sum**2 - (blue_sum * red_sum)) / (green_sum**2 + (blue_sum * red_sum))
        except ValueError:
            RGBVI = -999.0
        return RGBVI

    def compute_Photochemical_Reflectance_Index(self, red_sum, green_sum):
        try:
            PRI = red_sum / green_sum
        except ValueError:
            PRI = -999.0
        return PRI

    def compute_Soil_Adjusted_Vegetation_Index(self, green_sum, red_sum):
        try:
            SAVI = 1.5 * (green_sum - red_sum) / (green_sum + red_sum + 0.5)
        except ValueError:
            SAVI = -999.0
        return SAVI

    def compute_Soil_Colour_Index(self, red_sum, green_sum):
        try:
            SCI = (red_sum - green_sum) / (red_sum + green_sum)
        except ValueError:
            SCI = -999.0
        return SCI

    def compute_Spectral_Slope_Saturation_Index(self, red_sum, blue_sum):
        try:
            SI = (red_sum - blue_sum) / (red_sum + blue_sum)
        except ValueError:
            SI = -999.0
        return SI

    def compute_Triangular_Greenness_Index(self, green_sum, red_sum, blue_sum):
        try:
            TGI = green_sum - 0.39 * red_sum - 0.61 * blue_sum
        except ValueError:
            TGI = -999.0
        return TGI

    def compute_Visible_Atmospherically_Resistant_Vegetation_Index(self, green_sum, red_sum, blue_sum):
        try:
            VARI = (green_sum - red_sum) / (green_sum + red_sum - blue_sum)
        except ValueError:
            VARI = -999.0
        return VARI

    def compute_Visible_Band_Difference_Vegetation_Index(self, red_sum, green_sum, blue_sum):
        try:
            VDVI = (2 * green_sum - red_sum - blue_sum) / (2 * green_sum + red_sum + blue_sum)
        except ValueError:
            VDVI = -999.0
        return VDVI

    def compute_Vegetative_Index(self, red_sum, green_sum, blue_sum):
        try:
            VEG = green_sum / (red_sum**0.667 * blue_sum**0.334)
        except ValueError:
            VEG = -999.0
        return VEG

    def compute_Vegetation_Index_Green(self, green_sum, red_sum):
        try:
            VIgreen = (green_sum - red_sum) / (green_sum + red_sum)
        except ValueError:
            VIgreen = -999.0
        return VIgreen

    def compute_Visible_NDVI(self, red_sum, green_sum, blue_sum):
        try:
            vNDVI = 0.5268 * ((red_sum / (red_sum + green_sum + blue_sum)) -
                             0.1294 * (green_sum / (red_sum + green_sum + blue_sum))**0.3389 *
                             (blue_sum / (red_sum + green_sum + blue_sum)) - 0.3118)
        except ValueError:
            vNDVI = -999.0

        return vNDVI

    def compute_Woebbecke_Index(self, green_sum, blue_sum, red_sum):
        try:
            WI = (green_sum - blue_sum) / (red_sum - green_sum)
        except ValueError:
            WI = -999.0
        return WI

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_greenness(self, greenness, image):
        # Reset so a failed computation never returns the previous image's value.
        greenness.set_value(-999.0)
        try:
            myApp_Utils = App_Utils()
            red, green, blue = myApp_Utils.separateChannels(image)
            red_sum, green_sum, blue_sum = myApp_Utils.sumChannels(red, green, blue)
            # Plain floats: division by zero raises instead of silently producing inf/nan.
            red_sum, green_sum, blue_sum = float(red_sum), float(green_sum), float(blue_sum)

            if greenness.get_name() == self.NDVI:
                greenness.set_value(self.compute_NDVI(red_sum, green_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.ExG:
                greenness.set_value(self.compute_Excess_Green(green_sum, red_sum, blue_sum))
                #greenness.set_value(self.compute_ExG(red_sum, green_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.BCC:
                greenness.set_value(self.compute_Blue_Chromatic_Coordinate_Index(red_sum, green_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.BGI:
                greenness.set_value(self.compute_Simple_Blue_Green_Ratio(blue_sum, green_sum))
                return greenness

            if greenness.get_name() == self.BI:
                greenness.set_value(self.compute_Brightness_Index(red_sum, green_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.BRVI:
                greenness.set_value(self.compute_Blue_Red_Vegetation_Index(blue_sum, red_sum))
                return greenness

            if greenness.get_name() == self.CIVE:
                greenness.set_value(self.compute_Colour_Index_of_Vegetation(red_sum, green_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.ExB:
                greenness.set_value(self.compute_Excess_Blue(blue_sum, green_sum))
                return greenness

            if greenness.get_name() == self.ExGR:
                greenness.set_value(self.compute_Excess_Green_Excess_Red(red_sum, green_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.ExR:
                greenness.set_value(self.compute_Excess_Red(red_sum, green_sum))
                return greenness

            if greenness.get_name() == self.GCC:
                greenness.set_value(self.compute_GCC(red_sum, green_sum, blue_sum))
                return greenness

            #if greenness.get_name() == self.GCC:
            #    greenness.set_value(self.compute_Green_Percentage_Index(red_sum, green_sum, blue_sum))
            #    return greenness

            if greenness.get_name() == self.GLI:
                greenness.set_value(self.compute_GLI(red_sum, green_sum, blue_sum))
                return greenness

            #if greenness.get_name() == self.GLI:
            #    greenness.set_value(self.compute_Green_Leaf_Index(red_sum, green_sum, blue_sum))
            #    return greenness

            if greenness.get_name() == self.GR:
                greenness.set_value(self.compute_Simple_Red_Green_Ratio(green_sum, red_sum))
                return greenness

            if greenness.get_name() == self.GRVI:
                greenness.set_value(self.compute_Green_Red_Vegetation_Index(green_sum, red_sum))
                return greenness

            if greenness.get_name() == self.HI:
                greenness.set_value(self.compute_Primary_Colours_Hue_Index(red_sum, green_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.HUE:
                greenness.set_value(self.compute_Overall_Hue_Index(red_sum, green_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.IKAW:
                greenness.set_value(self.compute_Kawashima_Index(red_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.IOR:
                greenness.set_value(self.compute_Iron_Oxide_Ratio(red_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.IPCA:
                greenness.set_value(self.compute_Principal_Component_Analysis_Index(red_sum, green_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.MGRVI:
                greenness.set_value(self.compute_Modified_Green_Red_Vegetation_Index(green_sum, red_sum))
                return greenness

            if greenness.get_name() == self.MPRI:
                greenness.set_value(self.compute_Modified_Photochemical_Reflectance_Index(green_sum, red_sum))
                return greenness

            if greenness.get_name() == self.MVARI:
                greenness.set_value(self.compute_Modified_Visible_Atmospherically_Resistant_Vegetation_Index(green_sum, red_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.NDI:
                greenness.set_value(self.compute_Normalized_Difference_Index(green_sum, red_sum))
                return greenness

            if greenness.get_name() == self.NGBDI:
                greenness.set_value(self.compute_Normalized_Green_Blue_Difference_Index(green_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.NGRDI:
                greenness.set_value(self.compute_Normalized_Green_Red_Difference_Index(green_sum, red_sum))
                return greenness

            if greenness.get_name() == self.RCC:
                greenness.set_value(self.compute_Red_Chromatic_Coordinate_Index(red_sum, green_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.RGBVI:
                greenness.set_value(self.compute_Red_Green_Blue_Vegetation_Index(red_sum, green_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.RGI:
                greenness.set_value(self.compute_RGI(red_sum, green_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.PRI:
                greenness.set_value(self.compute_Photochemical_Reflectance_Index(red_sum, green_sum))
                return greenness

            if greenness.get_name() == self.SAVI:
                greenness.set_value(self.compute_Soil_Adjusted_Vegetation_Index(green_sum, red_sum))
                return greenness

            if greenness.get_name() == self.SCI:
                greenness.set_value(self.compute_Soil_Colour_Index(red_sum, green_sum))
                return greenness

            if greenness.get_name() == self.SI:
                greenness.set_value(self.compute_Spectral_Slope_Saturation_Index(red_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.TGI:
                greenness.set_value(self.compute_Triangular_Greenness_Index(green_sum, red_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.VARI:
                greenness.set_value(self.compute_Visible_Atmospherically_Resistant_Vegetation_Index(green_sum, red_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.VDVI:
                greenness.set_value(self.compute_Visible_Band_Difference_Vegetation_Index(red_sum, green_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.VEG:
                greenness.set_value(self.compute_Vegetative_Index(red_sum, green_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.VIgreen:
                greenness.set_value(self.compute_Vegetation_Index_Green(green_sum, red_sum))
                return greenness

            if greenness.get_name() == self.vNDVI:
                greenness.set_value(self.compute_Visible_NDVI(red_sum, green_sum, blue_sum))
                return greenness

            if greenness.get_name() == self.WI:
                greenness.set_value(self.compute_Woebbecke_Index(green_sum, blue_sum, red_sum))
                return greenness
        except (ValueError, TypeError, ArithmeticError):
            # ArithmeticError covers ZeroDivisionError (e.g. an all-black ROI).
            greenness.set_value(-999.0)
            return greenness

        # Unknown index name.
        return greenness




