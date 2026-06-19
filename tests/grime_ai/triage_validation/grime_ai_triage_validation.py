#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRIME AI Image Triage Validation Script

Generates controlled test images with known blur and brightness modifications
organized into per-test-case folders matching the GRIME AI Triage Validation
Test Plan. Each folder contains the images for that specific test case with
the correct parameters documented in a README.txt file.

Load each test folder directly into GRIME AI using the parameters specified
in the README.txt for that folder.

Optionally runs a local implementation of the triage algorithm on each test
folder using that test's specific parameters for quick reference comparison.
GRIME AI is always the authoritative test.

The local triage CSV output matches the GRIME AI report format:
  Focus Value, Focus Attrib, Intensity Value, Intensity Attrib., Filename, Moved

Usage:
    # Generate test folders (output to "Triage Test Images" next to script):
    python grime_ai_triage_validation.py generate

    # Generate to a specific folder:
    python grime_ai_triage_validation.py generate --output C:/my/folder

    # Generate and run local triage on each test folder with correct parameters:
    python grime_ai_triage_validation.py generate --run-triage

    # Run local triage on a single folder with specific parameters:
    python grime_ai_triage_validation.py triage
        --input "C:/Triage Test Images/Test_1_Blur_Threshold_Boundary"
        --output C:/results
        --blur-threshold 17.50 --brightness-min 0 --brightness-max 255

Author: John E. Stranzl Jr., PhD Candidate, University of Nebraska-Lincoln
License: Apache License, Version 2.0
"""

import os
import shutil
import cv2
import numpy as np
import csv
import argparse
from datetime import datetime


# ======================================================================================================================
# DEFAULT PARAMETERS
# ======================================================================================================================

DEFAULT_BLUR_THRESHOLD  = 17.50
DEFAULT_SHIFT_SIZE      = 60
DEFAULT_BRIGHTNESS_MIN  = 65.0
DEFAULT_BRIGHTNESS_MAX  = 180.0
DEFAULT_WIDTH           = 1920
DEFAULT_HEIGHT          = 1080


# ======================================================================================================================
# TEST CASE DEFINITIONS
# Each entry defines the folder name, description, which image series to use,
# and the triage parameters to apply for that test.
# ======================================================================================================================

TEST_CASES = [
    {
        "folder":      "Test_1_Blur_Threshold_Boundary",
        "title":       "Test 1 — Blur Threshold Boundary (Default Parameters)",
        "description": "Validates that the blur threshold correctly separates Nominal from Blurry images.\n"
                       "Brightness bounds are disabled (0/255) to isolate blur classification.",
        "series":      ["blur_series"],
        "blur_threshold":  DEFAULT_BLUR_THRESHOLD,
        "shift_size":      DEFAULT_SHIFT_SIZE,
        "brightness_min":  0.0,
        "brightness_max":  255.0,
        "expected": [
            ("blur_000pct_kernel1.jpg",   "Nominal", "Nominal", "N"),
            ("blur_010pct_kernel3.jpg",   "Nominal", "Nominal", "N"),
            ("blur_020pct_kernel7.jpg",   "Nominal", "Nominal", "N"),
            ("blur_030pct_kernel13.jpg",  "Nominal", "Nominal", "N"),
            ("blur_040pct_kernel21.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_050pct_kernel31.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_060pct_kernel43.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_070pct_kernel57.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_080pct_kernel73.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_090pct_kernel91.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_100pct_kernel111.jpg", "Blurry",  "Nominal", "Y"),
        ],
    },
    {
        "folder":      "Test_2_Brightness_Lower_Bound",
        "title":       "Test 2 — Brightness Lower Bound",
        "description": "Validates that images below the minimum brightness threshold are classified Too Dark.\n"
                       "Blur threshold is disabled (0) and brightness max is disabled (255) to isolate the lower bound.",
        "series":      ["brightness_series"],
        "blur_threshold":  0.0,
        "shift_size":      DEFAULT_SHIFT_SIZE,
        "brightness_min":  DEFAULT_BRIGHTNESS_MIN,
        "brightness_max":  255.0,
        "expected": [
            ("brightness_000.jpg", "Nominal", "Too Dark", "Y"),
            ("brightness_025.jpg", "Nominal", "Too Dark", "Y"),
            ("brightness_050.jpg", "Nominal", "Too Dark", "Y"),
            ("brightness_075.jpg", "Nominal", "Nominal",  "N"),
            ("brightness_100.jpg", "Nominal", "Nominal",  "N"),
            ("brightness_125.jpg", "Nominal", "Nominal",  "N"),
            ("brightness_150.jpg", "Nominal", "Nominal",  "N"),
            ("brightness_175.jpg", "Nominal", "Nominal",  "N"),
            ("brightness_200.jpg", "Nominal", "Nominal",  "N"),
            ("brightness_225.jpg", "Nominal", "Nominal",  "N"),
            ("brightness_250.jpg", "Nominal", "Nominal",  "N"),
        ],
    },
    {
        "folder":      "Test_3_Brightness_Upper_Bound",
        "title":       "Test 3 — Brightness Upper Bound",
        "description": "Validates that images above the maximum brightness threshold are classified Too Light.\n"
                       "Blur threshold is disabled (0) and brightness min is disabled (0) to isolate the upper bound.",
        "series":      ["brightness_series"],
        "blur_threshold":  0.0,
        "shift_size":      DEFAULT_SHIFT_SIZE,
        "brightness_min":  0.0,
        "brightness_max":  DEFAULT_BRIGHTNESS_MAX,
        "expected": [
            ("brightness_000.jpg", "Nominal", "Nominal",   "N"),
            ("brightness_025.jpg", "Nominal", "Nominal",   "N"),
            ("brightness_050.jpg", "Nominal", "Nominal",   "N"),
            ("brightness_075.jpg", "Nominal", "Nominal",   "N"),
            ("brightness_100.jpg", "Nominal", "Nominal",   "N"),
            ("brightness_125.jpg", "Nominal", "Nominal",   "N"),
            ("brightness_150.jpg", "Nominal", "Nominal",   "N"),
            ("brightness_175.jpg", "Nominal", "Nominal",   "N"),
            ("brightness_200.jpg", "Nominal", "Too Light", "Y"),
            ("brightness_225.jpg", "Nominal", "Too Light", "Y"),
            ("brightness_250.jpg", "Nominal", "Too Light", "Y"),
        ],
    },
    {
        "folder":      "Test_4_Combined_Conditions",
        "title":       "Test 4 — Combined Conditions (Default Parameters)",
        "description": "Validates that the triage module correctly flags images failing either criterion independently.\n"
                       "The control image (combined_sharp_nominal) must not be flagged or moved.",
        "series":      ["combined_series"],
        "blur_threshold":  DEFAULT_BLUR_THRESHOLD,
        "shift_size":      DEFAULT_SHIFT_SIZE,
        "brightness_min":  DEFAULT_BRIGHTNESS_MIN,
        "brightness_max":  DEFAULT_BRIGHTNESS_MAX,
        "expected": [
            ("combined_blurry_bright.jpg", "Blurry",  "Too Light", "Y"),
            ("combined_blurry_dark.jpg",   "Blurry",  "Too Dark",  "Y"),
            ("combined_blurry_nominal.jpg","Blurry",  "Nominal",   "Y"),
            ("combined_sharp_bright.jpg",  "Nominal", "Too Light", "Y"),
            ("combined_sharp_dark.jpg",    "Nominal", "Too Dark",  "Y"),
            ("combined_sharp_nominal.jpg", "Nominal", "Nominal",   "N"),
        ],
    },
    {
        "folder":      "Test_5_Blur_Threshold_Tighter",
        "title":       "Test 5 — Blur Threshold Sensitivity: Tighter Threshold",
        "description": "Validates that raising the blur threshold causes earlier onset of Blurry classification,\n"
                       "flagging more images than Test 1. Blur threshold raised to 25.0.",
        "series":      ["blur_series"],
        "blur_threshold":  25.0,
        "shift_size":      DEFAULT_SHIFT_SIZE,
        "brightness_min":  0.0,
        "brightness_max":  255.0,
        "expected": [
            ("blur_000pct_kernel1.jpg",   "Nominal", "Nominal", "N"),
            ("blur_010pct_kernel3.jpg",   "Nominal", "Nominal", "N"),
            ("blur_020pct_kernel7.jpg",   "Nominal", "Nominal", "N"),
            ("blur_030pct_kernel13.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_040pct_kernel21.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_050pct_kernel31.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_060pct_kernel43.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_070pct_kernel57.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_080pct_kernel73.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_090pct_kernel91.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_100pct_kernel111.jpg", "Blurry",  "Nominal", "Y"),
        ],
    },
    {
        "folder":      "Test_6_Blur_Threshold_Looser",
        "title":       "Test 6 — Blur Threshold Sensitivity: Looser Threshold",
        "description": "Validates that lowering the blur threshold causes later onset of Blurry classification,\n"
                       "flagging fewer images than Test 1. Blur threshold lowered to 5.0.",
        "series":      ["blur_series"],
        "blur_threshold":  5.0,
        "shift_size":      DEFAULT_SHIFT_SIZE,
        "brightness_min":  0.0,
        "brightness_max":  255.0,
        "expected": [
            ("blur_000pct_kernel1.jpg",   "Nominal", "Nominal", "N"),
            ("blur_010pct_kernel3.jpg",   "Nominal", "Nominal", "N"),
            ("blur_020pct_kernel7.jpg",   "Nominal", "Nominal", "N"),
            ("blur_030pct_kernel13.jpg",  "Nominal", "Nominal", "N"),
            ("blur_040pct_kernel21.jpg",  "Nominal", "Nominal", "N"),
            ("blur_050pct_kernel31.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_060pct_kernel43.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_070pct_kernel57.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_080pct_kernel73.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_090pct_kernel91.jpg",  "Blurry",  "Nominal", "Y"),
            ("blur_100pct_kernel111.jpg", "Blurry",  "Nominal", "Y"),
        ],
    },
]


# ======================================================================================================================
# IMAGE GENERATION
# ======================================================================================================================

def generate_base_image(width, height, intensity=127):
    """
    Generate a base grayscale image with texture so the FFT blur metric
    has high-frequency signal to work with. Without texture, all images
    would score as blurry regardless of applied blur level.
    """
    image = np.full((height, width), intensity, dtype=np.uint8)

    stripe_spacing = 20
    bright_stripe = min(intensity + 80, 255)
    dark_stripe   = max(intensity - 80, 0)

    for i in range(0, height, stripe_spacing):
        image[i, :] = bright_stripe
    for j in range(0, width, stripe_spacing):
        image[:, j] = dark_stripe

    for i in range(0, height, 5):
        for j in range(0, width, 5):
            if (i + j) % 10 == 0:
                image[i, j] = bright_stripe

    return image


def generate_source_images(width, height, staging_dir):
    """
    Generate all source images into staging subfolders.
    Returns a dict mapping series name to folder path.
    """
    base_image = generate_base_image(width, height)
    series_dirs = {}

    # --- Blur series ---
    blur_dir = os.path.join(staging_dir, "blur_series")
    os.makedirs(blur_dir, exist_ok=True)
    kernel_map = {0:1, 10:3, 20:7, 30:13, 40:21, 50:31, 60:43, 70:57, 80:73, 90:91, 100:111}
    print("  Blur series:")
    for level, ksize in kernel_map.items():
        blurred = cv2.GaussianBlur(base_image, (ksize, ksize), 0) if ksize > 1 else base_image.copy()
        filename = f"blur_{level:03d}pct_kernel{ksize}.jpg"
        cv2.imwrite(os.path.join(blur_dir, filename), blurred)
        print(f"    {filename}")
    series_dirs["blur_series"] = blur_dir

    # --- Brightness series ---
    brightness_dir = os.path.join(staging_dir, "brightness_series")
    os.makedirs(brightness_dir, exist_ok=True)
    print("  Brightness series:")
    for level in range(0, 256, 25):
        image = generate_base_image(width, height, intensity=level)
        filename = f"brightness_{level:03d}.jpg"
        cv2.imwrite(os.path.join(brightness_dir, filename), image)
        print(f"    {filename}")
    series_dirs["brightness_series"] = brightness_dir

    # --- Combined series ---
    combined_dir = os.path.join(staging_dir, "combined_series")
    os.makedirs(combined_dir, exist_ok=True)
    combinations = [
        {"blur_kernel": 61, "intensity": 30,  "label": "blurry_dark",   "note": "Blurry and too dark"},
        {"blur_kernel": 61, "intensity": 200, "label": "blurry_bright",  "note": "Blurry and too bright"},
        {"blur_kernel": 1,  "intensity": 30,  "label": "sharp_dark",    "note": "Sharp but too dark"},
        {"blur_kernel": 1,  "intensity": 200, "label": "sharp_bright",   "note": "Sharp but too bright"},
        {"blur_kernel": 1,  "intensity": 127, "label": "sharp_nominal",  "note": "Sharp and nominal — control image"},
        {"blur_kernel": 61, "intensity": 127, "label": "blurry_nominal", "note": "Blurry but nominal brightness"},
    ]
    print("  Combined series:")
    for combo in combinations:
        image = generate_base_image(width, height, intensity=combo["intensity"])
        ksize = combo["blur_kernel"]
        if ksize > 1:
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        filename = f"combined_{combo['label']}.jpg"
        cv2.imwrite(os.path.join(combined_dir, filename), image)
        print(f"    {filename}  [{combo['note']}]")
    series_dirs["combined_series"] = combined_dir

    return series_dirs


def write_readme(test_case, folder_path):
    """Write a README.txt into each test folder describing the test and parameters."""
    lines = [
        test_case["title"],
        "=" * len(test_case["title"]),
        "",
        test_case["description"],
        "",
        "GRIME AI Triage Parameters for this test:",
        f"  Blur Threshold:  {test_case['blur_threshold']}",
        f"  Shift Size:      {test_case['shift_size']}",
        f"  Brightness Min:  {test_case['brightness_min']}",
        f"  Brightness Max:  {test_case['brightness_max']}",
        "",
        "Options:",
        "  Move Images:     Enabled",
        "  Generate Report: Enabled",
        "",
        "Expected Results:",
        f"  {'Filename':<35} {'Focus':<10} {'Intensity':<12} {'Moved'}",
        f"  {'-'*35} {'-'*10} {'-'*12} {'-'*5}",
    ]
    for filename, focus, intensity, moved in test_case["expected"]:
        lines.append(f"  {filename:<35} {focus:<10} {intensity:<12} {moved}")

    readme_path = os.path.join(folder_path, "README.txt")
    with open(readme_path, 'w') as f:
        f.write("\n".join(lines) + "\n")


# ======================================================================================================================
# LOCAL TRIAGE ALGORITHM
# Replicates GRIME AI triage logic for quick reference comparison.
# NOTE: GRIME AI is the authoritative triage implementation.
#
# CSV output matches GRIME AI report format:
#   Focus Value, Focus Attrib, Intensity Value, Intensity Attrib., Filename, Moved
# ======================================================================================================================

def compute_fft_mean(gray_image, shift_size):
    """Replicate GRIME AI FFT blur metric."""
    h, w = gray_image.shape
    small = cv2.resize(gray_image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    cX, cY = sw // 2, sh // 2

    fft      = np.fft.fft2(small)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - shift_size:cY + shift_size, cX - shift_size:cX + shift_size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon    = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    return float(np.mean(magnitude))


def compute_intensity(gray_image):
    """Replicate GRIME AI brightness metric."""
    blurred = cv2.GaussianBlur(gray_image, (0, 0), 1)
    return float(cv2.mean(blurred)[0])


def classify_image(fft_mean, intensity, blur_threshold, brightness_min, brightness_max):
    """Apply GRIME AI decision logic."""
    focus_attr = "Blurry" if fft_mean <= blur_threshold else "Nominal"

    if intensity < brightness_min:
        intensity_attr = "Too Dark"
    elif intensity > brightness_max:
        intensity_attr = "Too Light"
    else:
        intensity_attr = "Nominal"

    moved = (focus_attr == "Blurry") or (intensity_attr != "Nominal")
    return focus_attr, intensity_attr, moved


def run_triage(input_dir, output_dir, blur_threshold, shift_size, brightness_min, brightness_max,
               label=None):
    """
    Run local triage algorithm on all images in input_dir.
    Writes a CSV matching the GRIME AI report format to output_dir.
    NOTE: GRIME AI is authoritative — this is for reference comparison only.
    """
    extensions = ('.jpg', '.jpeg', '.png')
    results    = []

    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in extensions and not f.startswith("README"):
                image_files.append(os.path.join(root, f))

    if not image_files:
        print(f"  No images found in {input_dir}")
        return

    title = label or os.path.basename(os.path.normpath(input_dir))
    print(f"\n  {title}")
    print(f"    Blur threshold: {blur_threshold}  |  Shift size: {shift_size}  |  "
          f"Brightness min: {brightness_min}  |  Brightness max: {brightness_max}\n")

    for filepath in image_files:
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"    WARNING: Could not load {filepath}")
            continue

        fft_mean   = compute_fft_mean(image, shift_size)
        intensity  = compute_intensity(image)
        focus_attr, intensity_attr, moved = classify_image(
            fft_mean, intensity, blur_threshold, brightness_min, brightness_max
        )

        results.append({
            "Focus Value":      round(fft_mean, 2),
            "Focus Attrib":     focus_attr,
            "Intensity Value":  round(intensity, 2),
            "Intensity Attrib": intensity_attr,
            "Filename":         os.path.basename(filepath),
            "Moved":            "Y" if moved else "N",
        })

        status = "MOVED" if moved else "OK   "
        print(f"    [{status}] {os.path.basename(filepath)}: "
              f"FFT={fft_mean:.2f} ({focus_attr}), "
              f"Intensity={intensity:.2f} ({intensity_attr}), "
              f"Moved={'Y' if moved else 'N'}")

    os.makedirs(output_dir, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.basename(os.path.normpath(input_dir))
    csv_path    = os.path.join(output_dir, f"local_triage_{folder_name}_{timestamp}.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    moved_count = sum(1 for r in results if r["Moved"] == "Y")
    print(f"\n    {moved_count}/{len(results)} images moved.")
    print(f"    Results: {csv_path}")
    print(f"    NOTE: Run these same images through GRIME AI for authoritative results.")


# ======================================================================================================================
# CLI COMMANDS
# ======================================================================================================================

def cmd_generate(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output is None:
        args.output = os.path.join(script_dir, "data", "triage_test_images")

    print(f"Generating validation images in: {args.output}")
    print(f"  Image size: {args.base_width} x {args.base_height}\n")

    # Generate source images into a staging subfolder
    staging_dir = os.path.join(args.output, "_source_images")
    os.makedirs(staging_dir, exist_ok=True)
    print("Generating source images...")
    series_dirs = generate_source_images(args.base_width, args.base_height, staging_dir)

    # Create one folder per test case
    print("\nCreating test case folders...")
    for tc in TEST_CASES:
        test_dir = os.path.join(args.output, tc["folder"])
        os.makedirs(test_dir, exist_ok=True)

        # Copy images from each required series into the test folder
        for series in tc["series"]:
            src_dir = series_dirs[series]
            for filename in os.listdir(src_dir):
                if os.path.splitext(filename)[1].lower() in ('.jpg', '.jpeg', '.png'):
                    shutil.copy2(os.path.join(src_dir, filename), os.path.join(test_dir, filename))

        # Write README with parameters and expected results
        write_readme(tc, test_dir)
        print(f"  {tc['folder']}  ({len(os.listdir(test_dir)) - 1} images)")

    total_tests = len(TEST_CASES)
    print(f"\nGeneration complete. {total_tests} test folders created in: {args.output}")
    print("Load each folder into GRIME AI using the parameters in its README.txt.")

    if args.run_triage:
        print("\n--- Running local triage on each test folder ---")
        triage_results_dir = os.path.join(script_dir, "results")
        for tc in TEST_CASES:
            test_dir = os.path.join(args.output, tc["folder"])
            run_triage(
                input_dir=test_dir,
                output_dir=triage_results_dir,
                blur_threshold=tc["blur_threshold"],
                shift_size=tc["shift_size"],
                brightness_min=tc["brightness_min"],
                brightness_max=tc["brightness_max"],
                label=tc["title"],
            )


def cmd_triage(args):
    print(f"Running local triage on: {args.input}")
    run_triage(
        input_dir=args.input,
        output_dir=args.output,
        blur_threshold=args.blur_threshold,
        shift_size=args.shift_size,
        brightness_min=args.brightness_min,
        brightness_max=args.brightness_max,
    )


def main():
    parser = argparse.ArgumentParser(
        description="GRIME AI Image Triage Validation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate all test folders (output to "Triage Test Images" next to script):
    python grime_ai_triage_validation.py generate

  Generate to a specific folder:
    python grime_ai_triage_validation.py generate --output C:/my/folder

  Generate and run local triage on each test folder with correct parameters:
    python grime_ai_triage_validation.py generate --run-triage

  Run local triage on a single test folder with specific parameters:
    python grime_ai_triage_validation.py triage
        --input "C:/Triage Test Images/Test_1_Blur_Threshold_Boundary"
        --output C:/results
        --blur-threshold 17.50 --brightness-min 0 --brightness-max 255

Note: GRIME AI is the authoritative triage implementation.
      Local triage is provided for reference comparison only.
      Local CSV output matches GRIME AI report format for direct comparison.
        """
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared triage parameters
    triage_params = argparse.ArgumentParser(add_help=False)
    triage_params.add_argument("--blur-threshold", type=float, default=DEFAULT_BLUR_THRESHOLD,
                               help=f"FFT mean threshold (default: {DEFAULT_BLUR_THRESHOLD})")
    triage_params.add_argument("--shift-size", type=int, default=DEFAULT_SHIFT_SIZE,
                               help=f"Half-width of FFT center mask in pixels (default: {DEFAULT_SHIFT_SIZE})")
    triage_params.add_argument("--brightness-min", type=float, default=DEFAULT_BRIGHTNESS_MIN,
                               help=f"Minimum mean grayscale intensity (default: {DEFAULT_BRIGHTNESS_MIN})")
    triage_params.add_argument("--brightness-max", type=float, default=DEFAULT_BRIGHTNESS_MAX,
                               help=f"Maximum mean grayscale intensity (default: {DEFAULT_BRIGHTNESS_MAX})")

    # Generate subcommand
    gen_parser = subparsers.add_parser("generate", help="Generate test case folders")
    gen_parser.add_argument("--output", default=None,
                            help="Output folder (default: 'Triage Test Images' next to script)")
    gen_parser.add_argument("--base-width", type=int, default=DEFAULT_WIDTH,
                            help=f"Image width in pixels (default: {DEFAULT_WIDTH})")
    gen_parser.add_argument("--base-height", type=int, default=DEFAULT_HEIGHT,
                            help=f"Image height in pixels (default: {DEFAULT_HEIGHT})")
    gen_parser.add_argument("--run-triage", action="store_true",
                            help="Also run local triage on each test folder with its specific parameters")
    gen_parser.set_defaults(func=cmd_generate)

    # Triage subcommand
    triage_parser = subparsers.add_parser("triage", parents=[triage_params],
                                          help="Run local triage on an existing image folder")
    triage_parser.add_argument("--input", required=True,
                               help="Folder containing images to triage")
    triage_parser.add_argument("--output", required=True,
                               help="Folder for triage results CSV")
    triage_parser.set_defaults(func=cmd_triage)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
