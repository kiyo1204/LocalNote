"""
History manager for LocalNote chat sessions.
Handles JSON-based chat history persistence.
"""

import os, json


class HistoryManager:
    """Manages chat history saved as JSON files."""

    def __init__(self, base_output_dir):
        self.base_output_dir = base_output_dir

    def load(self, db_name):
        """Load chat history from JSON file for a given database."""
        json_path = os.path.join(self.base_output_dir, db_name, "history.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self, db_name, history_data):
        """Save chat history to JSON file for a given database."""
        out_dir = os.path.join(self.base_output_dir, db_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "history.json")
        with open(out_path, mode="w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=4, ensure_ascii=False)

    def clear(self, db_name):
        """Delete the history JSON file for a given database."""
        json_path = os.path.join(self.base_output_dir, db_name, "history.json")
        if os.path.exists(json_path):
            os.remove(json_path)
