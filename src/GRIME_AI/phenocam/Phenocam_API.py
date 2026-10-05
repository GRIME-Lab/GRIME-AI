import os
import requests
import pandas as pd
from appcore.http_client import TimeoutSession
from datetime import datetime

class Phenocam_API:
    """
    A lightweight client for interacting with the PhenoCam API
    and returning results as pandas DataFrames.
    """

    BASE_URL = "https://phenocam.nau.edu/api/"

    def __init__(self):
        # Persistent session that ALWAYS applies a timeout + bounded retries.
        # A plain requests.Session() has no timeout: a silent server blocks
        # every call here forever ("read timeout=None"). TimeoutSession makes
        # each self.session.get() fail fast instead.
        self.session = TimeoutSession()

        # Define available endpoints
        self.endpoints = {
            "cameras": "cameras/",
            "roilists": "roilists/",
            "dailycounts": "dailycounts/",
            "middayimages": "middayimages/"
        }

    def _fetch(self, endpoint: str, all_records: bool = False) -> pd.DataFrame:
        """
        Internal method to fetch JSON from an endpoint and convert to DataFrame.
        If all_records=True, use the 'count' field to request all rows in one call.
        """
        url = self.BASE_URL + endpoint
        response = self.session.get(url)
        response.raise_for_status()
        json_data = response.json()

        # If paginated and we want all records
        if all_records and isinstance(json_data, dict) and "count" in json_data:
            total = json_data["count"]
            url = f"{url}?format=json&limit={total}"
            response = self.session.get(url)
            response.raise_for_status()
            json_data = response.json()

        # Extract results list if present
        if isinstance(json_data, dict) and "results" in json_data:
            records = json_data["results"]
        else:
            records = json_data

        # Normalize into a flat DataFrame
        return pd.json_normalize(records)

    def get_cameras(self, all_records: bool = True) -> pd.DataFrame:
        """Fetch camera metadata. By default, fetch all records in one call."""
        return self._fetch(self.endpoints["cameras"], all_records=all_records)

    def get_roilists(self, all_records: bool = True) -> pd.DataFrame:
        """Fetch ROI lists. By default, fetch all records in one call."""
        return self._fetch(self.endpoints["roilists"], all_records=all_records)

    def get_dailycounts(self) -> pd.DataFrame:
        """Fetch daily image counts."""
        return self._fetch(self.endpoints["dailycounts"], all_records=True)

    def get_middayimages(self) -> pd.DataFrame:
        """Fetch midday image metadata."""
        return self._fetch(self.endpoints["middayimages"], all_records=True)

    def get_all(self) -> dict:
        """
        Fetch all endpoints and return as a dictionary of DataFrames.
        """
        return {name: self._fetch(endpoint, all_records=True) for name, endpoint in self.endpoints.items()}

    def get_camera_count(self) -> int:
        """
        Fetch only the first page of the cameras endpoint and return the total count.
        This avoids downloading the entire dataset.
        """
        url = self.BASE_URL + self.endpoints["cameras"]
        response = self.session.get(url)
        response.raise_for_status()
        json_data = response.json()
        return json_data.get("count", 0)

    def get_endpoint_url(self, name: str) -> str:
        """
        Return the full URL for a given endpoint key (e.g., 'cameras').
        Raises KeyError if the endpoint name is invalid.
        """
        if name not in self.endpoints:
            raise KeyError(f"Invalid endpoint name: {name}. "
                           f"Valid options are: {list(self.endpoints.keys())}")
        return self.BASE_URL + self.endpoints[name]

    def list_endpoints(self) -> list:
        """
        Return a list of all endpoint keys defined in the class.
        """
        return list(self.endpoints.keys())

    def get_cameras_with_counts(self) -> pd.DataFrame:
        """
        Fetch camera metadata and add a per-site camera count column.
        """
        cameras_df = self.get_cameras(all_records=True)
        if "sitename" not in cameras_df.columns:
            raise ValueError("Expected 'sitename' column not found in cameras data.")

        # Count cameras per site
        counts = cameras_df.groupby("sitename").size().reset_index(name="camera_count")

        # Merge back into the cameras dataframe
        cameras_with_counts = cameras_df.merge(counts, on="sitename", how="left")
        return cameras_with_counts

    def export_dailycounts(self, output_path: str = "daily_image_counts.xlsx"):
        """
        Fetch daily image counts for each site and camera, and export to Excel.
        """
        df = self.get_dailycounts()

        if df is None or df.empty:
            print("No dailycounts data returned.")
            return

        # Normalize column names
        df.columns = [c.lower() for c in df.columns]

        # Reorder if expected columns exist
        required = {"sitename", "camera", "date", "count"}
        if required.issubset(df.columns):
            cols = ["sitename", "camera", "date", "count"] + [c for c in df.columns if c not in required]
            df = df[cols]

        df.to_excel(output_path, index=False)
        print(f"Exported daily image counts → {output_path} ({len(df)} rows)")

    def export_all_to_excel(self, output_dir: str = "."):
        """
        Fetch all endpoints and export each to an Excel (.xlsx) file.
        Also calls export_dailycounts to generate the daily image counts file.
        Displays start time, end time, and total duration.
        """
        os.makedirs(output_dir, exist_ok=True)

        start_time = datetime.now()
        print(f"Export started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        dataframes = self.get_all()
        for name, df in dataframes.items():
            file_path = os.path.join(output_dir, f"{name}.xlsx")
            try:
                df.to_excel(file_path, index=False)
                print(f"Exported {name} → {file_path} ({len(df)} rows)")
            except Exception as e:
                print(f"Failed to export {name}: {e}")

        end_time = datetime.now()
        duration = end_time - start_time

        print(f"Export finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration}")

        ###JES THIS TENDS TO BE DANGEROUSLY SLOW DUE TO ALL THE REQUIRED INTERACTIONS WITH THE SERVER
        if 0:
            # Call the specialized dailycounts export
            dailycounts_path = os.path.join(output_dir, "daily_image_counts.xlsx")
            self.export_dailycounts(output_path=dailycounts_path)
