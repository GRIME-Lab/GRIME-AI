import os
import json

from pathlib import Path

from appcore.Save_Utils import Save_Utils
from .app_identity import APP_CONFIG_FILENAME


# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====      class JsonEditor      =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
class JsonEditor():
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, filename=None):
        super().__init__()

        if filename is not None:
            self.json_filename = filename
        else:
            self.json_filename = APP_CONFIG_FILENAME

    '''
    def save_to_json(self):
        data = {f'entry_{i}': edit_line.text() for i, edit_line in enumerate(self.edit_lines)}
        with open('data.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print('Data saved to data.json')
    '''

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def update_json_entry(self, entry_key, new_value):
        try:
            json_file = os.path.join(Save_Utils().get_settings_folder(), self.json_filename)
            json_file = os.path.normpath(json_file)

            with open(json_file, 'r') as file:
                data = json.load(file)

            if entry_key in data:
                data[entry_key] = new_value
                with open(json_file, 'w') as file:
                    json.dump(data, file, indent=4)
                print(f'Entry {entry_key} updated to {new_value}')
            else:
                print(f'Adding missing entry {entry_key} to {self.json_filename}')
                self.add_key_value_to_json(entry_key, new_value)
        except FileNotFoundError:
            print('{self.json_filename} file not found')
        except Exception:
            print(f'Adding missing entry {entry_key} to {self.json_filename}')
            self.add_key_value_to_json(entry_key, new_value)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def add_key_value_to_json(self, key, value):
        json_file = os.path.join(Save_Utils().get_settings_folder(), self.json_filename)

        try:
            # Load the existing data from the JSON file
            with open(json_file, 'r') as file:
                data = json.load(file)
        except Exception:
            # If the file doesn't exist, create an empty dictionary
            data = {}

        # Check if the key is already in the data
        if key not in data:
            # Add the key-value pair to the data
            data[key] = value

            # Write the updated data back to the JSON file
            with open(json_file, 'w') as file:
                json.dump(data, file, indent=4)
            print(f"Added key '{key}' with value '{value}' to the JSON file.")
        else:
            print(f"Key '{key}' already exists in the JSON file.")

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def getValue(self, key):
        json_file = os.path.join(Save_Utils().get_settings_folder(), self.json_filename)

        try:
            # Load the existing data from the JSON file
            with open(json_file, 'r') as file:
                data = json.load(file)

            return data[key]
        except Exception:
            # If the file doesn't exist, create an empty dictionary
            data = {}

            return None

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def load_json_file(self, json_filename):

        json_filename = Path(json_filename)

        # Load current settings if file exists
        if json_filename.is_file():
            with json_filename.open("r", encoding="utf-8") as f:
                settings = json.load(f)
        else:
            settings = {}

        return settings
