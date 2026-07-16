# SAGE/settings_manager.py
import json
import shutil
from pathlib import Path


def default_settings_dir() -> Path:
    """Per-user settings directory. Kept alongside the app's other state
    under ~/Documents/GRIME-AI/ so nothing lands in the execution folder."""
    return Path.home() / "Documents" / "GRIME-AI" / "Settings"


class SettingsManager:
    """Manage application settings persistence."""

    def __init__(self, settings_file="sage_settings.json"):
        p = Path(settings_file)

        # A bare filename (no directory) must NOT resolve against the CWD —
        # that put sage_settings.json in the execution folder. Route it to the
        # per-user settings dir. An explicit relative/absolute path is honored
        # as-is (useful for tests or custom deployments).
        if p.is_absolute() or p.parent != Path("."):
            self.settings_file = p
        else:
            self.settings_file = default_settings_dir() / p.name

        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy(p.name)
        self.settings = self._load_settings()

    def _migrate_legacy(self, filename):
        """One-time move of a settings file left in the CWD by older builds.
        Only runs when the new location has no file yet, so we never clobber
        current settings."""
        legacy = Path.cwd() / filename
        try:
            if (legacy.exists()
                    and legacy.resolve() != self.settings_file.resolve()
                    and not self.settings_file.exists()):
                shutil.move(str(legacy), str(self.settings_file))
        except Exception as e:
            print(f"Settings migration skipped: {e}")

    def _load_settings(self):
        """Load settings from JSON file"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading settings: {e}")
                return {}
        return {}

    def _save_settings(self):
        """Save settings to JSON file"""
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def get(self, key, default=None):
        """Get a setting value"""
        return self.settings.get(key, default)

    def set(self, key, value):
        """Set a setting value and save"""
        self.settings[key] = value
        self._save_settings()

    def get_folder_path(self):
        """Get the last used folder path"""
        return self.get("folder_path", "")

    def set_folder_path(self, path):
        """Set the folder path"""
        self.set("folder_path", path)
