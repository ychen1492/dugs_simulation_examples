import numpy as np
from darts.discretizer import value_vector
from darts.engines import property_evaluator_iface, sim_params
from darts.models.darts_model import DartsModel
from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer
from darts.physics.properties.iapws.iapws_property import iapws_water_density_evaluator, temperature_region1_evaluator
from darts.reservoirs.cpg_reservoir import CPG_Reservoir
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.tools.gen_cpg_grid import gen_cpg_grid
from iapws import SeaWater
from iapws.iapws97 import _Bound_Ph, _Region1, _Region4, _TSat_P
from numba import njit

from dugs_simulation_examples.xarray_api import XarrayApi


@njit
def compute_flux(
    n_cells,
    n_dim,
    adj_mat_offset,
    adj_mat_cols,
    adj_mat,
    centroids,
    conns_n,
    conns_c,
    conns_area,
):
    mat_flux = {}
    for cell_id in range(n_cells):
        a = np.zeros((adj_mat_offset[cell_id + 1] - adj_mat_offset[cell_id], n_dim))
        cell_centroid = centroids[cell_id]
        ids = np.argsort(
            adj_mat_cols[adj_mat_offset[cell_id] : adj_mat_offset[cell_id + 1]]
        )
        for i, k in enumerate(
            range(adj_mat_offset[cell_id], adj_mat_offset[cell_id + 1])
        ):
            conn_id = adj_mat[k]
            n = conns_n[conn_id]
            face_centroid = conns_c[conn_id]
            sign = np.sign((face_centroid - cell_centroid).dot(n))
            a[ids[i]] = sign * conns_area[conn_id] * n
        mat_flux[cell_id] = np.linalg.inv(a.T.dot(a)).dot(a.T)
    return mat_flux


class Model(DartsModel):
    def __init__(
        self,
        set_nx,
        set_ny,
        set_nz,
        perms,
        poro,
        set_dx,
        set_dy,
        set_dz,
        overburden,
        n_points=32,
    ):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()
        self.report_time = 365
        self.total_time = 100 * 365
        self.nx = set_nx
        self.ny = set_ny
        self.nz = set_nz
        self.dx = set_dx
        self.dy = set_dy
        self.dz = set_dz
        self.perm = perms
        self.poro = poro
        self.overburden = overburden
        self.set_reservoir()
        # self.set_cpg_reservoir()
        # self.prepare_velocity_reconstruction()
        self.set_physics(n_points=n_points)
        self.set_sim_params(
            first_ts=1e-5,
            mult_ts=8,
            max_ts=365,
            runtime=self.total_time,
            tol_newton=1e-3,
            tol_linear=1e-4,
            it_newton=20,
            it_linear=40,
            newton_type=sim_params.newton_global_chop,
            newton_params=value_vector([1]),
        )

        self.timer.node["initialization"].stop()

        # T_init = 350.
        # state_init = value_vector([200., 0.])
        # enth_init = self.physics.property_containers[0].enthalpy_ev['total'](T_init).evaluate(state_init)
        # self.initial_values = {self.physics.vars[0]: state_init[0],
        #                        self.physics.vars[1]: enth_init
        #                        }

    def prepare_velocity_reconstruction(self):
        n_dim = 3
        self.mat_flux = {}
        adj_mat_offset = np.array(
            self.reservoir.discr_mesh.adj_matrix_offset, dtype=np.int32
        )
        adj_mat_cols = np.array(
            self.reservoir.discr_mesh.adj_matrix_cols, dtype=np.int32
        )
        adj_mat = np.array(self.reservoir.discr_mesh.adj_matrix, dtype=np.int32)
        centroids = np.array([c.values for c in self.reservoir.discr_mesh.centroids])
        # Prepare structured arrays for connections
        n_conns = len(self.reservoir.discr_mesh.conns)
        conns_n = np.zeros((n_conns, n_dim))
        conns_c = np.zeros((n_conns, n_dim))
        conns_area = np.zeros(n_conns)

        for i, conn in enumerate(self.reservoir.discr_mesh.conns):
            conns_n[i] = np.array(conn.n.values)
            conns_c[i] = np.array(conn.c.values)
            conns_area[i] = conn.area
        self.mat_flux = compute_flux(
            self.reservoir.discr_mesh.n_cells,
            n_dim,
            adj_mat_offset,
            adj_mat_cols,
            adj_mat,
            centroids,
            conns_n,
            conns_c,
            conns_area,
        )

    def reconstruct_velocities(self, fluxes):
        vels = np.zeros((self.reservoir.discr_mesh.n_cells, 3))
        conn_id = 0
        max_conns = 10
        for cell_m in range(self.reservoir.discr_mesh.n_cells):
            rhs = np.zeros(max_conns)
            face_id = 0
            while self.reservoir.mesh.block_m[conn_id] == cell_m:
                cell_p = self.reservoir.mesh.block_p[conn_id]
                # skip well connections
                if not (
                    cell_p >= self.reservoir.mesh.n_res_blocks
                    and cell_p < self.reservoir.mesh.n_blocks
                ):
                    rhs[face_id] = fluxes[conn_id]
                    # face = self.unstr_discr.faces[cell_m][face_id]
                    # assert(self.discr_mesh.adj_matrix_cols[self.discr_mesh.adj_matrix_offset[cell_m] + face_id] == cell_p)
                    # assert(face.cell_id2 == cell_p or face.face_id2 + self.mesh.n_blocks == cell_p)
                    face_id += 1
                conn_id += 1

            assert face_id == self.mat_flux[cell_m].shape[1]
            vels[cell_m] = self.mat_flux[cell_m].dot(rhs[:face_id])
        return vels

    def set_cpg_reservoir(self):
        dz_list = [15, 30, 60, 120]
        underburden = self.overburden
        self.nz += self.overburden + underburden
        overburden_prop = np.ones(self.nx * self.ny * self.overburden) * 1e-5
        underburden_prop = np.ones(self.nx * self.ny * underburden) * 1e-5
        perm = np.concatenate([overburden_prop, self.perm, underburden_prop])
        poro = np.concatenate([overburden_prop, self.poro, underburden_prop])
        output = gen_cpg_grid(
            self.nx,
            self.ny,
            self.nz,
            self.dx,
            self.dy,
            self.dz,
            permx=perm,
            permy=perm,
            permz=perm,
            poro=poro,
            start_z=1800,
            burden_dz=dz_list,
        )
        coord = output["COORD"]
        zcorn = output["ZCORN"]
        arrays = {
            "SPECGRID": np.array([self.nx, self.ny, self.nz]),
            "PERMX": perm,
            "PERMY": perm,
            "PERMZ": 0.1 * perm,
            "PORO": poro,
            "COORD": coord,
            "ZCORN": zcorn,
        }
        arrays["ACTNUM"] = np.ones(arrays["SPECGRID"].prod(), dtype=np.int32)

        self.reservoir = CPG_Reservoir(self.timer, arrays)
        self.reservoir.discretize()
        # add larger volumes
        self.reservoir.set_boundary_volume(
            xz_minus=1e8, xz_plus=1e8, yz_minus=1e8, yz_plus=1e8
        )
        self.reservoir.apply_volume_depth()

        # get wrapper around local array (length equal to active blocks number)
        cond_mesh = np.array(self.reservoir.mesh.rock_cond, copy=False)
        hcap_mesh = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        cond_mesh[np.array(self.reservoir.mesh.poro) <= 0.1] = 2.2 * 86.4
        cond_mesh[np.array(self.reservoir.mesh.poro) > 0.1] = 3 * 86.4
        hcap_mesh[np.array(self.reservoir.mesh.poro) <= 0.1] = 2300
        hcap_mesh[np.array(self.reservoir.mesh.poro) > 0.1] = 2450

    def set_initial_conditions(
        self, initial_values: dict = None, gradient: dict = None
    ):
        self.physics.set_nonuniform_initial_conditions(
            self.reservoir.mesh, pressure_grad=107, temperature_grad=30
        )

    def set_reservoir(self):
        overburden_dz = np.zeros(self.nx * self.ny * self.overburden)
        underburden_dz = np.zeros(self.nx * self.ny * self.overburden)
        dz_list = [15, 30, 60, 120]
        # dz_list = []
        if self.overburden == 0:
            set_dz_new = self.dz
        else:
            for k in range(self.overburden):
                start = k * self.nx * self.ny
                end = (k + 1) * self.nx * self.ny
                if k == 0:
                    overburden_dz[start:end] = dz_list[3]
                    underburden_dz[start:end] = dz_list[0]
                elif k == 1:
                    overburden_dz[start:end] = dz_list[2]
                    underburden_dz[start:end] = dz_list[1]
                elif k == 2:
                    overburden_dz[start:end] = dz_list[1]
                    underburden_dz[start:end] = dz_list[2]
                else:
                    overburden_dz[start:end] = dz_list[0]
                    underburden_dz[start:end] = dz_list[3]

            # add the overburden layers' dz
            set_dz_new = np.concatenate(
                [
                    overburden_dz,
                    np.ones(self.nx * self.ny * self.nz) * self.dz,
                    underburden_dz,
                ]
            )

        underburden = self.overburden
        self.nz += self.overburden + underburden
        overburden_prop = np.ones(self.nx * self.ny * self.overburden) * 0.001
        underburden_prop = np.ones(self.nx * self.ny * underburden) * 0.001
        perm = np.concatenate([overburden_prop, self.perm, underburden_prop])
        poro = np.concatenate([overburden_prop, self.poro, underburden_prop])

        # gen_cpg_grid(self.nx, self.ny, self.nz-(self.overburden + underburden), self.dx, self.dy, self.dz, permx=perm, permy=perm, permz=perm, poro=poro,burden_dz=dz_list, gridname='str_reservoir.grdecl')

        self.xrdata = XarrayApi(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            total_time=self.total_time,
            report_time=self.report_time,
            number_of_inj_wells=1,
            number_of_prd_wells=1,
            last_time=False,
        )
        self.xrdata.set_xarray_static(
            "Perm", np.rot90(perm.reshape((self.nx, self.ny, self.nz), order="F"))
        )
        self.xrdata.set_xarray_static(
            "Poro", np.rot90(poro.reshape((self.nx, self.ny, self.nz), order="F"))
        )
        # rock heat capacity and rock thermal conduction
        hcap = np.ones(self.nx * self.ny * self.nz)
        rcond = np.ones(self.nx * self.ny * self.nz)
        hcap[poro <= 1e-1] = 2300  # volumetric heat capacity: kJ/m3/K
        hcap[poro > 1e-1] = 2450

        rcond[poro <= 1e-1] = 2.2 * 86.4  # kJ/m/day/K
        rcond[poro > 1e-1] = 3 * 86.4

        # depth = np.empty(self.nx * self.ny * self.nz)
        # top = 1630
        # unique_thickness = sorted(np.unique(set_dz_new))
        # if len(unique_thickness) > 1:
        #     for k in range(self.nz):
        #         if k < 4:
        #             depth[k * self.nx * self.ny:(k + 1) * self.nx * self.ny] = top + sum(
        #                 unique_thickness[-i] for i in range(1, k + 2, 1)) / 2
        #         elif 4 <= k <= 19:
        #             depth[k * self.nx * self.ny:(k + 1) * self.nx * self.ny] = top + 225 + (2 * (k - 3) - 1) * \
        #                                                                        unique_thickness[0] * 0.5
        #         elif 19 < k <= 22:
        #             depth[k * self.nx * self.ny:(k + 1) * self.nx * self.ny] = top + 225 + 100 + sum(
        #                 unique_thickness[-i] for i in range(1, 5, 1)) / 2
        # else:
        #     for k in range(self.nz):
        #         depth[k * self.nx * self.ny:(k + 1) * self.nx * self.ny] = top + (2 * k + 1) * unique_thickness[0] * 0.5

        # add more layers above or below the reservoir
        self.reservoir = StructReservoir(
            self.timer,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            dx=self.dx,
            dy=self.dy,
            dz=set_dz_new,
            permx=perm,
            permy=perm,
            permz=0.1 * perm,
            poro=poro,
            depth=None,
        )

        # add larger volumes
        # self.reservoir.boundary_volumes['xy_minus'] = 1e18
        # self.reservoir.boundary_volumes['xy_plus'] = 1e18
        self.reservoir.boundary_volumes["yz_minus"] = 1e8
        self.reservoir.boundary_volumes["yz_plus"] = 1e8
        self.reservoir.boundary_volumes["xz_minus"] = 1e8
        self.reservoir.boundary_volumes["xz_plus"] = 1e8

        return

    def set_wells(self):
        # add well's locations
        injection_well_x = int(2400 / self.dx)
        production_well_x = injection_well_x + int(1300 / self.dx)

        # add perforations to the payzone
        start_index = self.overburden + 1
        end_index = self.nz - self.overburden + 1
        # add well
        self.reservoir.add_well("INJ1")
        for k in range(start_index, end_index):
            self.reservoir.add_perforation(
                "INJ1",
                cell_index=(injection_well_x, int(self.ny / 2), k),
                well_radius=0.16,
                verbose=True,
            )

        self.reservoir.add_well("PRD1")
        for k in range(start_index, end_index):
            self.reservoir.add_perforation(
                "PRD1",
                cell_index=(production_well_x, int(self.ny / 2), k),
                well_radius=0.16,
                verbose=True,
            )

        return

    def set_physics(self, n_points):
        # create pre-defined physics for geothermal
        property_container = NewPropertyContainer()
        # components = ["H2O", "NaCl"]
        # property_container.density_ev = dict([('water', Spivey2004(components=components, molarity=250/58.44,ions=['Na','Cl'])),
        #                                       ('steam', iapws_steam_density_evaluator())])
        # property_container.viscosity_ev = dict([('water', MaoDuan2009(components=components, molarity=250/58.44)),
        #                                         ('steam', iapws_steam_viscosity_evaluator())])
        # Define property evaluators with constant density and viscosity
        # property_container.density_ev = dict([('water', DensityBasic(compr=1e-5, dens0=54.53)),
        #                                      ('steam', DensityBasic(compr=1e-5, dens0=0)),
        #                                     ])
        # property_container.viscosity_ev = dict([('water', ConstFunc(0.38)),
        #                                         ('steam', ConstFunc(0.1))])
        self.physics = Geothermal(
            self.timer,
            n_points=n_points,
            min_p=100,
            max_p=500,
            min_e=1000,
            max_e=25000,
            cache=False,
            mass_rate=True,
        )
        self.physics.add_property_region(property_container)

        return

    # T=300K, P=200bars, the enthalpy is 2358 [kJ/kmol]
    def set_well_controls(self):
        self.physics.define_well_controls()
        for i, w in enumerate(self.reservoir.wells):
            if w.name.lower().startswith("i"):
                # w.control = self.physics.new_rate_water_inj(7500, 300)
                w.control = self.physics.new_mass_rate_water_inj(417000, 2358)
                # w.constraint = self.physics.new_bhp_water_inj(200, self.inj_temperature)
            else:
                # w.control = self.physics.new_rate_water_prod(7500)
                w.control = self.physics.new_mass_rate_water_prod(417000)


class NewPropertyContainer(PropertyContainer):
    def evaluate(self, state):
        self.temperature = self.temperature_ev.evaluate(state)

        for j, phase in enumerate(["water", "steam"]):
            self.saturation[j] = self.saturation_ev[phase].evaluate(state)
            self.conduction[j] = self.conduction_ev[phase].evaluate(state)
            # self.density[j] = self.density_ev[phase].evaluate(state[0], self.temperature)
            self.density[j] = self.density_ev[phase].evaluate(state)
            self.viscosity[j] = self.viscosity_ev[phase].evaluate(state)
            # self.viscosity[j] = self.viscosity_ev[phase].evaluate(state[0], self.temperature)
            self.relperm[j] = self.relperm_ev[phase].evaluate(state)
            self.enthalpy[j] = self.enthalpy_ev[phase].evaluate(state)
        # self.viscosity[0] = self.viscosity_ev['water'].evaluate(state[0], self.temperature)
        # self.viscosity[1] = self.viscosity_ev['steam'].evaluate(state)
        #
        # self.density[0] = self.density_ev['water'].evaluate(state[0], self.temperature)/18.015
        # self.density[1] = self.density_ev['steam'].evaluate(state)
        return


class seawater_enthalpy_evaluator(property_evaluator_iface):
    def __init__(self, brine_mass_fraction):
        super().__init__()
        self.brine_mass_fraction = brine_mass_fraction

    def evaluate(self, pressure, temperature):
        seawater = SeaWater(T=temperature, P=pressure / 10, S=self.brine_mass_fraction)
        seawater_enth = seawater.h

        return seawater_enth * 18.015


class seawater_density_evaluator(property_evaluator_iface):
    def __init__(self, brine_mass_fraction):
        super().__init__()
        self.brine_mass_fraction = brine_mass_fraction

    def evaluate(self, pressure, temperature):
        seawater = SeaWater(T=temperature, P=pressure / 10, S=self.brine_mass_fraction)
        seawater_enth = seawater.rho

        return seawater_enth / 18.015


class brine_density_evaluator(property_evaluator_iface):
    def __init__(self, brine_mass_fraction):
        super().__init__()
        self.brine_mass_fraction = brine_mass_fraction

    def evaluate(self, state):
        P, h = state[0] * 0.1, state[1] / 18.015
        Pmin = 0.000611212677444
        if P < Pmin:
            P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin:
            h = hmin

        region = _Bound_Ph(P, h)
        if region == 1:
            temperature = temperature_region1_evaluator()
            T = temperature.evaluate(state)
            water_density = (
                iapws_water_density_evaluator().evaluate(state) * 18.015 / 1000
            )  # convert to g/cm3
        elif region == 4:
            T = _TSat_P(P)
            if T <= 623.15:
                water_density = 1 / _Region4(P, 0)["v"] / 1000  # convert to g/cm3
            else:
                raise NotImplementedError(
                    "Variables out of bound: p="
                    + str(P)
                    + "h="
                    + str(h)
                    + " region="
                    + str(region)
                )
        elif region == 2:
            water_density = 0
        else:
            raise NotImplementedError(
                "Variables out of bound: p="
                + str(P)
                + "h="
                + str(h)
                + " region="
                + str(region)
            )
        T -= 273.15
        brine_density = water_density + self.brine_mass_fraction * (
            0.668
            + 0.44 * self.brine_mass_fraction
            + 1e-6
            * (
                300 * P
                - 2400 * P * self.brine_mass_fraction
                + T
                * (
                    80
                    + 3 * T
                    - 3300 * self.brine_mass_fraction
                    - 13 * P
                    + 47 * P * self.brine_mass_fraction
                )
            )
        )

        return brine_density * 1000 / 18.015


class brine_viscosity_evaluator(property_evaluator_iface):
    def __init__(self, brine_mass_fraction):
        super().__init__()
        self.brine_mass_fraction = brine_mass_fraction

    def evaluate(self, state):
        P, h = state[0] * 0.1, state[1] / 18.015
        Pmin = 0.000611212677444
        if P < Pmin:
            P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin:
            h = hmin

        region = _Bound_Ph(P, h)
        if region == 1:
            temperature = temperature_region1_evaluator()
            T = temperature.evaluate(state)
        T -= 273.15
        brine_viscosity = (
            0.1
            + 0.333 * self.brine_mass_fraction
            + (1.65 + 91.9 * self.brine_mass_fraction**3)
            * np.exp(
                -(
                    0.42 * ((np.power(self.brine_mass_fraction, 0.8) - 0.17) ** 2)
                    + 0.045
                )
                * np.power(T, 0.8)
            )
        )

        return brine_viscosity
