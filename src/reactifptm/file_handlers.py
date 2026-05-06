import json
from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np


class FileTypes(Enum):
    """Enum of supported PAE file formats."""

    NPZ = "npz"
    NPY = "npy"
    PKL = "pkl"
    JSON = "json"

    @classmethod
    def values(cls):
        return [value.value for value in cls.__members__.values()]


class FileBase:
    """Base class providing shared path handling for the file loaders."""

    def __init__(self, pathway: Union[str, Path]):
        self.pathway = Path(pathway)
        self.suffix = self.pathway.suffix[1:]

    def __str__(self):
        return str(self.pathway)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.pathway})"


class NpzFile(FileBase):
    def __init__(self, npz_file: Union[str, Path]):
        """
        Object to handle npz files

        Args:
            npz_file (Union[str, Path]): Path to the npz file

        Attributes:
            npz_file (Path): Path to the npz file
            data (dict): Dictionary containing the data from the npz file

        """
        super().__init__(npz_file)
        self.npz_file = Path(npz_file)
        self.data = self.load_npz_file()

    def load_npz_file(self) -> dict:
        return dict(np.load(self.npz_file, allow_pickle=True))


class NpyFile(FileBase):
    def __init__(self, npy_file: Union[str, Path]):
        """
        Object to handle npy files

        Args:
            npy_file (Union[str, Path]): Path to the npy file

        Attributes:
            npy_file (Path): Path to the npy file
            data (np.ndarray): Numpy array containing the data from the np
        """

        super().__init__(npy_file)
        self.npy_file = Path(npy_file)
        self.data = self.load_npy_file()

    def load_npy_file(self) -> np.ndarray:
        return np.load(self.npy_file, allow_pickle=True)


class PklFile(FileBase):
    def __init__(self, pkl_file: Union[str, Path]):
        """
        Object to handle pkl files

        Args:
            pkl_file (Union[str, Path]): Path to the pkl file

        Attributes:
            pkl_file (Path): Path to the pkl file
            data (np.ndarray): Numpy array containing the data from the np
        """

        super().__init__(pkl_file)
        self.npy_file = Path(pkl_file)
        self.data = self.load_pkl_file()

    def load_pkl_file(self) -> np.ndarray:
        return np.load(self.npy_file, allow_pickle=True)


class JsonFile(FileBase):
    def __init__(self, json_file: Union[str, Path]):
        """
        Object to handle json files

        Args:
            json_file (Union[str, Path]): Path to the json file

        Attributes:
            json_file (Path): Path to the json file
            data (dict): Dictionary containing the data from the json file

        """
        super().__init__(json_file)
        self.data = self.load_json_file()

    def load_json_file(self):
        # load the json file
        with open(self.pathway) as f:
            data = json.load(f)

        return data
