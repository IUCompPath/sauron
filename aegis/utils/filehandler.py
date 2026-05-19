import os
from typing import Any, Dict, List, Optional

import pandas as pd

# File extensions for slide images
EXTENSIONS: List[str] = [".svs", ".mrxs", ".tiff", ".tif", ".TIFF", ".ndpi"]


class PatientFolder:
    def __init__(
        self, patients_dir: str, labels: Optional[pd.DataFrame] = None
    ) -> None:
        self.patients_dir: str = patients_dir
        self.labels: Optional[pd.DataFrame] = labels
        self.data: pd.DataFrame = pd.DataFrame(columns=["pid", "slide_id", "label"])

    def analyze(self) -> None:
        try:
            new_data: List[Dict[str, Any]] = []
            for folder_name in os.listdir(self.patients_dir):
                folder_path: str = os.path.join(self.patients_dir, folder_name)
                if os.path.isdir(folder_path):
                    label = self._get_label(folder_name)
                    self._process_folder(folder_name, folder_path, label, new_data)
            if new_data:
                self.data = pd.concat(
                    [self.data, pd.DataFrame(new_data)], ignore_index=True
                )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Error analyzing patient folders: Directory '{self.patients_dir}' not found. {e}"
            )
        except PermissionError as e:
            raise RuntimeError(
                f"Error analyzing patient folders: Permission denied when accessing '{self.patients_dir}'. {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Unexpected error analyzing patient folders: {e}")

    def _get_label(self, folder_name: str) -> Any:
        if self.labels is not None and folder_name in self.labels.index:
            return self.labels.loc[folder_name]
        elif folder_name.isdigit():
            return int(folder_name)
        else:
            raise ValueError(
                f"Label for folder '{folder_name}' not found. Ensure that the folder name is either a digit or is present in the provided labels."
            )

    def _process_folder(
        self,
        folder_name: str,
        folder_path: str,
        label: Any,
        new_data: List[Dict[str, Any]],
    ) -> None:
        for file_name in os.listdir(folder_path):
            if any(file_name.endswith(ext) for ext in EXTENSIONS):
                slide_id: str = file_name
                new_data.append(
                    {
                        "pid": folder_name,
                        "slide_id": slide_id,
                        "label": label,
                    }
                )

    def __len__(self) -> int:
        return len(self.data)

    def get_data(self) -> pd.DataFrame:
        return self.data
