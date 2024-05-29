import os

import numpy as np
import xarray as xr
from xarray import DataArray, Dataset


class XarrayApi:
    def __init__(
        self,
        nx: int,
        ny: int,
        total_time: float,
        report_time: float,
        number_of_inj_wells: int,
        number_of_prd_wells: int,
        nz=None,
        last_time=False,
    ):
        """Constructor of the XarrayApi class



        :param nx: the number of blocks in x direction
        :type nx: int
        :param ny: the number of blocks in y direction
        :type ny: int
        :param total_time: the total simulation time e.g. 10x365 days
        :type total_time: int
        :param report_time: the time step reported in the time_data_report e.g. 365 days
        :type report_time: int
        :param number_of_inj_wells: the number of injection wells
        :type number_of_inj_wells: int
        :param number_of_prd_wells: the number production wells
        :type number_of_prd_wells: int
        :param nz: the number of blocks in z direction, if it is a 2D model, nz is None
        :type nz: int or None
        :return: An instance of XarrayApi
        :rtype:
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.report_time = report_time
        self.total_time = total_time
        self.number_of_inj_wells = number_of_inj_wells
        self.number_of_prd_wells = number_of_prd_wells
        self.last_time = last_time

        data_vars, coords = self.create_dataset(
            int(total_time / report_time) + 1, last_time
        )
        # add more field names for dataset
        self.set_data_vars_for_wells(data_vars)

        self.xarray_dataset = Dataset(data_vars=data_vars, coords=coords)

    def create_dataset(self, number_of_time_steps, last_time=False):
        common_dims = ["Y", "X", "Z"] if self.nz is not None else ["Y", "X"]
        time_dim = "last_time" if last_time else "time"
        number_of_time_steps = 2 if last_time else number_of_time_steps

        data_vars = {
            "Perm": (
                common_dims,
                (
                    np.zeros((self.ny, self.nx, self.nz))
                    if self.nz is not None
                    else np.zeros((self.ny, self.nx))
                ),
            ),
            "Poro": (
                common_dims,
                (
                    np.zeros((self.ny, self.nx, self.nz))
                    if self.nz is not None
                    else np.zeros((self.ny, self.nx))
                ),
            ),
            "Pressure": (
                [*common_dims, time_dim],
                (
                    np.zeros((self.ny, self.nx, self.nz, number_of_time_steps))
                    if self.nz is not None
                    else np.zeros((self.ny, self.nx, number_of_time_steps))
                ),
            ),
            "Temperature": (
                [*common_dims, time_dim],
                (
                    np.zeros((self.ny, self.nx, self.nz, number_of_time_steps))
                    if self.nz is not None
                    else np.zeros((self.ny, self.nx, number_of_time_steps))
                ),
            ),
        }

        coords = {
            "time": range(number_of_time_steps),
            "last_time": range(2),
            "X": range(self.nx),
            "Y": range(self.ny),
        }

        if self.nz is not None:
            coords["Z"] = range(self.nz)

        return data_vars, coords

    def set_data_vars_for_wells(self, data_vars):
        """Set the more field names relevant to the wells in dataset

        :param data_vars: the dictionary which contains all fields in dataset
        :type data_vars: dict
        :return:
        :rtype:
        """
        # According to the number of wells to adjust the output
        # for the well information from time data report
        # injection wells
        for inj_well in range(self.number_of_inj_wells):
            data_vars[f"Injection Well {inj_well + 1} Temperature"] = (
                ["time"],
                np.zeros(int(self.total_time / self.report_time) + 1),
            )
            data_vars[f"Injection Well {inj_well + 1} Pressure"] = (
                ["time"],
                np.zeros(int(self.total_time / self.report_time) + 1),
            )
            data_vars[f"Injection Well {inj_well + 1} Energy"] = (
                ["time"],
                np.zeros(int(self.total_time / self.report_time) + 1),
            )
        # production wells
        for prd_well in range(self.number_of_prd_wells):
            data_vars[f"Production Well {prd_well + 1} Temperature"] = (
                ["time"],
                np.zeros(int(self.total_time / self.report_time) + 1),
            )
            data_vars[f"Production Well {prd_well + 1} Pressure"] = (
                ["time"],
                np.zeros(int(self.total_time / self.report_time) + 1),
            )
            data_vars[f"Production Well {prd_well + 1} Energy"] = (
                ["time"],
                np.zeros(int(self.total_time / self.report_time) + 1),
            )

    def set_xarray_static(self, property_name, property_values):
        """Set the values for static properties, e.g. permeability and porosity,
        currently can only be 'Perm' and 'Poro'

        :param property_name: the name of the static properties
        :type property_name: str
        :param property_values: numpy array of the property values
        :type property_values: np.ndarray
        :return:
        :rtype:
        """
        if property_name not in ["Perm", "Poro"]:
            raise ValueError(f"{property_name} is not in the xarray dataset...")

        self.xarray_dataset[property_name][:] = (
            DataArray(
                property_values,
                dims=("Y", "X"),
                coords={
                    "X": range(self.nx),
                    "Y": range(self.ny),
                },
            ).astype(np.float32)
            if self.nz is None
            else DataArray(
                property_values,
                dims=("Y", "X", "Z"),
                coords={"X": range(self.nx), "Y": range(self.ny), "Z": range(self.nz)},
            ).astype(np.float32)
        )

    def set_dynamic_xarray(self, wells, time_data_report):
        """Extract the information from time data report

        :param wells: the well information from DARTS reservoir class
        :type wells: list
        :param time_data_report: the time data report from the DARTS simulation
        :type time_data_report: pd.dataframe
        :return:
        :rtype:
        """
        if len(wells) != self.number_of_prd_wells + self.number_of_inj_wells:
            raise ValueError(
                "The number of production and injection wells is not consistent "
                "with the wells in DARTS model..."
            )
        for _, w in enumerate(wells):
            if w.name.lower().startswith("i"):
                self.xarray_dataset[f"Injection Well {w.name[-1]} Temperature"][
                    1:
                ] = DataArray(
                    np.array(time_data_report[f"{w.name} : temperature (K)"]),
                    dims="time",
                    coords={
                        "time": range(1, int(self.total_time / self.report_time) + 1)
                    },
                ).astype(
                    np.float32
                )
                self.xarray_dataset[f"Injection Well {w.name[-1]} Pressure"][
                    1:
                ] = DataArray(
                    np.array(time_data_report[f"{w.name} : BHP (bar)"]),
                    dims="time",
                    coords={
                        "time": range(1, int(self.total_time / self.report_time) + 1)
                    },
                ).astype(
                    np.float32
                )
                self.xarray_dataset[f"Injection Well {w.name[-1]} Energy"][
                    1:
                ] = DataArray(
                    np.array(time_data_report[f"{w.name} : energy (kJ/day)"]),
                    dims="time",
                    coords={
                        "time": range(1, int(self.total_time / self.report_time) + 1)
                    },
                ).astype(
                    np.float32
                )
            elif w.name.lower().startswith("p"):
                self.xarray_dataset[f"Production Well {w.name[-1]} Temperature"][
                    1:
                ] = DataArray(
                    np.array(time_data_report[f"{w.name} : temperature (K)"]),
                    dims="time",
                    coords={
                        "time": range(1, int(self.total_time / self.report_time) + 1)
                    },
                ).astype(
                    np.float32
                )
                self.xarray_dataset[f"Production Well {w.name[-1]} Pressure"][
                    1:
                ] = DataArray(
                    np.array(time_data_report[f"{w.name} : BHP (bar)"]),
                    dims="time",
                    coords={
                        "time": range(1, int(self.total_time / self.report_time) + 1)
                    },
                ).astype(
                    np.float32
                )
                self.xarray_dataset[f"Production Well {w.name[-1]} Energy"][
                    1:
                ] = DataArray(
                    np.array(time_data_report[f"{w.name} : energy (kJ/day)"]),
                    dims="time",
                    coords={
                        "time": range(1, int(self.total_time / self.report_time) + 1)
                    },
                ).astype(
                    np.float32
                )
            else:
                raise ValueError(
                    f"{w.name} doesn't start with 'i' or 'p', "
                    f"please check the well names..."
                )

    def set_dynamic_grid_xarray(
        self, time_index, dynamic_property_name, dynamic_property_values
    ):
        """Set the values relevant to each grid in different time steps

        :param time_index: the index of the report time
        :type time_index: int
        :param dynamic_property_name: the name of the property changing in time
        :type dynamic_property_name: str
        :param dynamic_property_values: the values of the property in one time step
        :type dynamic_property_values: np.ndarray
        :return:
        :rtype:
        """
        if dynamic_property_name not in ["Pressure", "Temperature"]:
            raise ValueError(f"{dynamic_property_name} is not in the xarray dataset...")

        if self.nz is None:
            coords = {"X": range(self.nx), "Y": range(self.ny)}
            dims = ("Y", "X")
        else:
            coords = {"X": range(self.nx), "Y": range(self.ny), "Z": range(self.nz)}
            dims = ("Y", "X", "Z")

        data_array = DataArray(
            dynamic_property_values,
            dims=dims,
            coords=coords,
        ).astype(np.float32)

        if self.nz is not None:
            # output 3D dynamic grid information
            self.xarray_dataset[dynamic_property_name][:, :, :, time_index] = data_array
        else:
            # output 2D dynamic grid information
            self.xarray_dataset[dynamic_property_name][:, :, time_index] = data_array

    def write_xarray(self, file_name, output_dir):
        """Write the xarray to .nc file

        :param file_name: the name of the nc file
        :type file_name: str
        :param output_dir: the directory where the nc file is saved
        :type output_dir: str
        :return:
        :rtype:
        """

        print(f"Saving dataset to the file {file_name}.nc")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.xarray_dataset.to_netcdf(path=os.path.join(output_dir, f"{file_name}.nc"))
        print(f"Finished saving {file_name}...")

        return 0

    @staticmethod
    def get_xarray(file_name):
        """

        :param file_name: the name of the nc file
        :type file_name: str
        :return: xarray dataset
        :rtype: xarray.Dataset
        """

        xarray_data = xr.open_dataset(file_name)

        return xarray_data
