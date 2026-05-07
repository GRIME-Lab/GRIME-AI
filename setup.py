import ast
from datetime import date
from pathlib import Path
from setuptools import setup

def get_version():
    variables = {}
    text = (Path(__file__).parent / "src" / "GRIME_AI"/ "version.py").read_text()
    for line in text.splitlines():
        if "=" in line:
            var, _, value = line.partition("=")
            variables[var.strip()] = ast.literal_eval(value.strip())

    base_version = variables["SW_VERSION"]
    build_type = variables["RELEASE"]
    build_date = variables["BUILD_DATE"].replace("-","")

    if build_type == "nightly":
        return f"{base_version}.dev{build_date}"
    return base_version

setup(
    version=get_version(),
)
