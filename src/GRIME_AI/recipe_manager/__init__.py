"""GRIME AI Recipe Manager subpackage.

Exposes the public API so callers can simply:

    from GRIME_AI.recipe_manager import RecipeManagerDialog, RecipeStore, Recipe
"""

from .recipe_manager import RecipeManagerDialog, RecipeStore, Recipe

__all__ = ["RecipeManagerDialog", "RecipeStore", "Recipe"]
