#include <mpi.h>
#include "hdf5.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

int id, p;

// index of the first interior point of the current process
int i_start, j_start;
// number of interior grid points of the current process in x/y dir
int nx_local, ny_local;
// row/col index of the current process in the process grid
int rank_row, rank_col;
// mpi processes in x/y dir
int px, py;

const int Nx = 102;
const int Ny = 52;

const double Lx = 0.1, Ly = 0.05;
const double rho = 1000, nu = 1e-6;
const double P_max = 0.5;
const double t_end = 50.0;
const double dt_min = 1.e-3;
const double courant = 0.01;
const double dt_out = 0.5;

double** P, **P_old;
double** u, **u_old;
double** v, **v_old;
double** PPrhs;

double* P_1d, *P_old_1d;
double* u_1d, *u_old_1d;
double* v_1d, *v_old_1d;
double* PPrhs_1d;

double dx, dy, dt, t;

//* */ Obstacle handling **************
const bool ADD_OBSTACLE = true;
const int OBS_I_MIN = Nx * 0.50;
const int OBS_I_MAX = Nx * 0.55;
const int OBS_J_MIN = Ny * 0.3;
const int OBS_J_MAX = Ny * 0.4;

inline bool is_obstacle(int i_global, int j_global) {
    return i_global >= OBS_I_MIN && i_global <= OBS_I_MAX &&
           j_global >= OBS_J_MIN && j_global <= OBS_J_MAX;
}
// *******************

//* Periodic top/bottom toggle ********
const bool PERIODIC_BC = false;
//*************

static void h5_check(herr_t status, const char* msg)
{
    if (status < 0) {
        std::cerr << "Rank " << id << " HDF5 error: " << msg << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void write_double_attr_collective(hid_t loc, const char* name, double val)
{
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t at = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT, H5P_DEFAULT);
    h5_check(at, name);
    h5_check(H5Awrite(at, H5T_NATIVE_DOUBLE, &val), name);
    H5Aclose(at);
    H5Sclose(sp);
}

static void write_int_attr_collective(hid_t loc, const char* name, int val)
{
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t at = H5Acreate2(loc, name, H5T_NATIVE_INT, sp, H5P_DEFAULT, H5P_DEFAULT);
    h5_check(at, name);
    h5_check(H5Awrite(at, H5T_NATIVE_INT, &val), name);
    H5Aclose(at);
    H5Sclose(sp);
}

void grids_to_file(int outit)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    h5_check(fapl, "H5Pcreate(H5P_FILE_ACCESS)");
    h5_check(H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL), "H5Pset_fapl_mpio");

    hid_t file_id;
    if (outit == 0) {
        file_id = H5Fcreate("fields.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        h5_check(file_id, "H5Fcreate(fields.h5)");
    } else {
        file_id = H5Fopen("fields.h5", H5F_ACC_RDWR, fapl);
        h5_check(file_id, "H5Fopen(fields.h5)");
    }
    H5Pclose(fapl);

    // Create geometry once, collectively, on first output
    if (outit == 0) {
        hid_t geo_grp = H5Gcreate2(file_id, "/geometry",
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        h5_check(geo_grp, "H5Gcreate2(/geometry)");

        std::vector<double> xcoords(Nx), ycoords(Ny);
        for (int i = 0; i < Nx; i++) xcoords[i] = i * dx;
        for (int j = 0; j < Ny; j++) ycoords[j] = j * dy;

        hsize_t dimx[1] = {static_cast<hsize_t>(Nx)};
        hsize_t dimy[1] = {static_cast<hsize_t>(Ny)};
        hid_t sp_x = H5Screate_simple(1, dimx, NULL);
        hid_t sp_y = H5Screate_simple(1, dimy, NULL);

        hid_t ds_x = H5Dcreate2(geo_grp, "x_coords", H5T_NATIVE_DOUBLE,
                                sp_x, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t ds_y = H5Dcreate2(geo_grp, "y_coords", H5T_NATIVE_DOUBLE,
                                sp_y, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        h5_check(ds_x, "H5Dcreate2(x_coords)");
        h5_check(ds_y, "H5Dcreate2(y_coords)");

        // Dataset writes may be collective or independent; collective keeps ranks aligned
        hid_t dxpl_meta = H5Pcreate(H5P_DATASET_XFER);
        h5_check(dxpl_meta, "H5Pcreate(H5P_DATASET_XFER)");
        h5_check(H5Pset_dxpl_mpio(dxpl_meta, H5FD_MPIO_COLLECTIVE), "H5Pset_dxpl_mpio geometry");

        h5_check(H5Dwrite(ds_x, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, dxpl_meta, xcoords.data()),
                 "H5Dwrite(x_coords)");
        h5_check(H5Dwrite(ds_y, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, dxpl_meta, ycoords.data()),
                 "H5Dwrite(y_coords)");

        H5Pclose(dxpl_meta);

        write_double_attr_collective(geo_grp, "Lx", Lx);
        write_double_attr_collective(geo_grp, "Ly", Ly);
        write_double_attr_collective(geo_grp, "rho", rho);
        write_double_attr_collective(geo_grp, "nu", nu);
        write_int_attr_collective(geo_grp, "Nx", Nx);
        write_int_attr_collective(geo_grp, "Ny", Ny);

        H5Dclose(ds_x);
        H5Dclose(ds_y);
        H5Sclose(sp_x);
        H5Sclose(sp_y);
        H5Gclose(geo_grp);
    }

    char grpname[32];
    std::snprintf(grpname, sizeof(grpname), "/step_%04d", outit);

    hid_t step_grp = H5Gcreate2(file_id, grpname,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5_check(step_grp, "H5Gcreate2(step)");

    write_double_attr_collective(step_grp, "time", t);
    write_double_attr_collective(step_grp, "dt", dt);
    write_int_attr_collective(step_grp, "output_index", outit);

    auto write_field = [&](const char* name, double** field) {
        hsize_t gdims[2] = {
            static_cast<hsize_t>(Ny - 2),
            static_cast<hsize_t>(Nx - 2)
        };
        hid_t filespace = H5Screate_simple(2, gdims, NULL);

        hid_t dset = H5Dcreate2(step_grp, name, H5T_NATIVE_DOUBLE,
                                filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        h5_check(dset, name);

        hsize_t offset[2] = {
            static_cast<hsize_t>(j_start - 1),
            static_cast<hsize_t>(i_start - 1)
        };
        hsize_t count[2] = {
            static_cast<hsize_t>(ny_local),
            static_cast<hsize_t>(nx_local)
        };

        h5_check(H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL),
                "H5Sselect_hyperslab");

        std::vector<double> buf(static_cast<size_t>(nx_local) * static_cast<size_t>(ny_local));
        for (int j = 1; j <= ny_local; ++j) {
            for (int i = 1; i <= nx_local; ++i) {
                buf[(j - 1) * nx_local + (i - 1)] = field[i][j];
            }
        }

        hsize_t memdims[2] = {
            static_cast<hsize_t>(ny_local),
            static_cast<hsize_t>(nx_local)
        };
        hid_t memspace = H5Screate_simple(2, memdims, NULL);

        hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
        h5_check(dxpl, "H5Pcreate(H5P_DATASET_XFER)");
        h5_check(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE), "H5Pset_dxpl_mpio field");

        h5_check(H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl, buf.data()), name);

        H5Pclose(dxpl);
        H5Sclose(memspace);
        H5Sclose(filespace);
        H5Dclose(dset);
    };

    write_field("P", P);
    write_field("u", u);
    write_field("v", v);

    H5Gclose(step_grp);
    H5Fclose(file_id);
}

void setup_continuous_array(double**& array_2d, double*& array_1d, int m, int n)
{
	array_1d = new double[m * n];
	array_2d = new double* [m];

	for (int i = 0; i < m; i++)
		array_2d[i] = &array_1d[i * n];
}

void free_continuous_array(double**& array_2d, double*& array_1d)
{
	delete[] array_1d;
	delete[] array_2d;
}

void find_tiles(int p, int Nx_inner, int Ny_inner, int &px, int &py) {
    double best_cost = 1e18;
    for (int candidate_px = 1; candidate_px <= p; candidate_px++) {
        if (p % candidate_px != 0) continue;  // must divide evenly
        int candidate_py = p / candidate_px;
        double cost = (double)Nx_inner / candidate_px
                    + (double)Ny_inner / candidate_py;
        if (cost < best_cost) {
            best_cost = cost;
            px = candidate_px;
            py = candidate_py;
        }
    }
}

void setup_decomposition(void) 
{
	find_tiles(p, Nx-2, Ny-2, px, py);

	rank_col = id % px;
	rank_row = id / px;

	int Nx_inner = Nx - 2;
	int base_x = Nx_inner / px;
	int rem_x = Nx_inner % px;
	nx_local = base_x + (rank_col < rem_x ? 1 : 0);

	int Ny_inner = Ny - 2;
	int base_y = Ny_inner / py;
	int rem_y = Ny_inner % py;
	ny_local = base_y + (rank_row < rem_y ? 1 : 0);

	// + 1 to account for ghost nodes
	// there are rank_ processes before current one, each base_ long/tall
	// + the number of extra elements in previous processes due to uneven split
	i_start = 1 + rank_col*base_x + min(rem_x, rank_col);
	j_start = 1 + rank_row*base_y + min(rem_y, rank_row);
}

MPI_Datatype create_row_datatype(double** data, int j_fixed, double* buf_base)
{
    vector<int> block_lengths(nx_local, 1);
    vector<MPI_Datatype> types(nx_local, MPI_DOUBLE);
    vector<MPI_Aint> offsets(nx_local);

    MPI_Aint base_addr;
    MPI_Get_address(buf_base, &base_addr);

    for (int i = 1; i <= nx_local; i++) {
        MPI_Aint addr;
        MPI_Get_address(&data[i][j_fixed], &addr);
        offsets[i - 1] = addr - base_addr;
    }

    MPI_Datatype face_type;
    MPI_Type_create_struct(nx_local, block_lengths.data(), offsets.data(), types.data(), &face_type);
    MPI_Type_commit(&face_type);
    return face_type;
}

void exchange(double** data)
{
    int left = (rank_col > 0) ? rank_row * px + (rank_col - 1) : MPI_PROC_NULL;
    int right = (rank_col < px - 1) ? rank_row * px + (rank_col + 1) : MPI_PROC_NULL;
    int bottom, top;

    if (PERIODIC_BC) {
        bottom = ((rank_row - 1 + py) % py) * px + rank_col;
        top = ((rank_row + 1) % py) * px + rank_col;
    } else {
        bottom = (rank_row > 0) ? (rank_row - 1) * px + rank_col : MPI_PROC_NULL;
        top = (rank_row < py - 1) ? (rank_row + 1) * px + rank_col : MPI_PROC_NULL;
    }

    MPI_Request reqs[8];
    int count = 0;

    MPI_Datatype recv_bottom_type = create_row_datatype(data, 0, &data[1][0]);
    MPI_Datatype recv_top_type = create_row_datatype(data, ny_local + 1, &data[1][ny_local + 1]);
    MPI_Datatype send_bottom_type = create_row_datatype(data, 1, &data[1][1]);
    MPI_Datatype send_top_type = create_row_datatype(data, ny_local, &data[1][ny_local]);

	
    // Receive ghost cells
    // left ghost column: x = 0, y = 1..ny_local
    MPI_Irecv(&data[0][1], ny_local, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &reqs[count++]);

    // right ghost column: x = nx_local+1, y = 1..ny_local
    MPI_Irecv(&data[nx_local + 1][1], ny_local, MPI_DOUBLE, right, 1, MPI_COMM_WORLD, &reqs[count++]);

    // bottom ghost row: y = 0, x = 1..nx_local
    MPI_Irecv(&data[1][0], 1, recv_bottom_type, bottom, 2, MPI_COMM_WORLD, &reqs[count++]);

    // top ghost row: y = ny_local+1, x = 1..nx_local
    MPI_Irecv(&data[1][ny_local + 1], 1, recv_top_type, top, 3, MPI_COMM_WORLD, &reqs[count++]);

    // Send interior boundary cells

    // send left interior column: x = 1, y = 1..ny_local
    MPI_Isend(&data[1][1], ny_local, MPI_DOUBLE, left, 1, MPI_COMM_WORLD, &reqs[count++]);

    // send right interior column: x = nx_local, y = 1..ny_local
    MPI_Isend(&data[nx_local][1], ny_local, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &reqs[count++]);

    // send bottom interior row: y = 1, x = 1..nx_local
    MPI_Isend(&data[1][1], 1, send_bottom_type, bottom, 3, MPI_COMM_WORLD, &reqs[count++]);

    // send top interior row: y = ny_local, x = 1..nx_local
    MPI_Isend(&data[1][ny_local], 1, send_top_type, top, 2, MPI_COMM_WORLD, &reqs[count++]);

    MPI_Waitall(count, reqs, MPI_STATUSES_IGNORE);

    MPI_Type_free(&recv_bottom_type);
    MPI_Type_free(&recv_top_type);
    MPI_Type_free(&send_bottom_type);
    MPI_Type_free(&send_top_type);
}

void set_pressure_BCs(void) {
    // No slip over top and bottom walls, so zero normal gradient for pressure
    // If processes bottom
    if (j_start == 1 && !PERIODIC_BC)
        for (int i = 1; i <= nx_local; i++)
			P[i][0] = P[i][1];

    // If processes bottom
    if (j_start + ny_local == Ny - 1 && !PERIODIC_BC)
        for (int i = 1; i <= nx_local; i++)
			P[i][ny_local + 1] = P[i][ny_local];

    // No slip over half on the right wall, so zero normal gradient for pressure
    // Apply only if processes right wall
    if (i_start + nx_local == Nx - 1) {
        for (int j = 1; j <= ny_local; j++) {
            int j_global = j_start + (j - 1);
            if (j_global >= Ny/2)
                P[nx_local + 1][j] = P[nx_local][j];
        }
    }
}

void setup(void)
{

    int m = nx_local + 2;
    int n = ny_local + 2;

	setup_continuous_array(P, P_1d, m, n);
    setup_continuous_array(P_old, P_old_1d, m, n);
    setup_continuous_array(u, u_1d, m, n);
    setup_continuous_array(u_old, u_old_1d, m, n);
    setup_continuous_array(v, v_1d, m, n);
    setup_continuous_array(v_old, v_old_1d, m, n);
    setup_continuous_array(PPrhs, PPrhs_1d, m, n);

    fill(P_1d, P_1d + m*n, 0.0);
    fill(P_old_1d, P_old_1d + m*n, 0.0);
    fill(u_1d, u_1d + m*n, 0.0);
    fill(u_old_1d, u_old_1d + m*n, 0.0);
    fill(v_1d, v_1d + m*n, 0.0);
    fill(v_old_1d, v_old_1d + m*n, 0.0);
    fill(PPrhs_1d, PPrhs_1d + m*n, 0.0);


	dx = Lx / (Nx - 1);
	dy = Ly / (Ny - 1);

    if (i_start == 1)
        for (int j = 0; j <= ny_local+1; j++)
            P[0][j] = P_max;

	memcpy(P_old_1d, P_1d, m * n * sizeof(double));

	t = 0.0;
}

void calculate_ppm_RHS_central(void)
{
	for (int i = 1; i <= nx_local; i++)
		for (int j = 1; j <= ny_local; j++)
		{
			PPrhs[i][j] = rho / dt * ((u[i + 1][j] - u[i - 1][j]) / (2. * dx) + (v[i][j + 1] - v[i][j - 1]) / (2. * dy));
		}
}


int pressure_poisson_jacobi(double rtol = 1.e-5)
{
	double tol = 10. * rtol;
	int it = 0;

	while (tol > rtol)
	{
		swap(P, P_old);
		exchange(P_old);
		double sum_val = 0.0;
		tol = 0.0;
		it++;

		//Jacobi iteration
		for (int i = 1; i <= nx_local; i++) {
			for (int j = 1; j <= ny_local; j++)
			{
				P[i][j] = 1.0 / (2.0 + 2.0 * (dx * dx) / (dy * dy)) * (P_old[i + 1][j] + P_old[i - 1][j] + 
					(P_old[i][j + 1] + P_old[i][j - 1]) * (dx * dx) / (dy * dy)
					- (dx * dx) * PPrhs[i][j]);

				sum_val += fabs(P[i][j]);
				tol += fabs(P[i][j] - P_old[i][j]);
			}
		}

		set_pressure_BCs();

		double local_vals[2] = {tol, sum_val};
		double global_vals[2];
		MPI_Allreduce(local_vals, global_vals, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		
		tol = global_vals[0] / max(1.e-10, global_vals[1]);
	}

	return it;
}

void calculate_intermediate_velocity(void)
{
	for (int i = 1; i <= nx_local; i++)
		for (int j = 1; j <= ny_local; j++)
		{
			//viscous diffusion
			u[i][j] = u_old[i][j] + dt * nu * ((u_old[i + 1][j] + u_old[i - 1][j] - 2.0 * u_old[i][j]) / (dx * dx) + (u_old[i][j + 1] + u_old[i][j - 1] - 2.0 * u_old[i][j]) / (dy * dy));
			v[i][j] = v_old[i][j] + dt * nu * ((v_old[i + 1][j] + v_old[i - 1][j] - 2.0 * v_old[i][j]) / (dx * dx) + (v_old[i][j + 1] + v_old[i][j - 1] - 2.0 * v_old[i][j]) / (dy * dy));
			//advection - upwinding
			if (u_old[i][j] > 0.0)
			{
				u[i][j] -= dt * u_old[i][j] * (u_old[i][j] - u_old[i - 1][j]) / dx;
				v[i][j] -= dt * u_old[i][j] * (v_old[i][j] - v_old[i - 1][j]) / dx;
			}
			else
			{
				u[i][j] -= dt * u_old[i][j] * (u_old[i + 1][j] - u_old[i][j]) / dx;
				v[i][j] -= dt * u_old[i][j] * (v_old[i + 1][j] - v_old[i][j]) / dx;
			}
			if (v_old[i][j] > 0.0)
			{
				u[i][j] -= dt * v_old[i][j] * (u_old[i][j] - u_old[i][j - 1]) / dy;
				v[i][j] -= dt * v_old[i][j] * (v_old[i][j] - v_old[i][j - 1]) / dy;
			}
			else
			{
				u[i][j] -= dt * v_old[i][j] * (u_old[i][j + 1] - u_old[i][j]) / dy;
				v[i][j] -= dt * v_old[i][j] * (v_old[i][j + 1] - v_old[i][j]) / dy;
			}
		}
}

void set_velocity_BCs(void)
{
	//Zero velocity gradients at pressure inlet and outlet
    if (i_start == 1)
	    for (int j = 1; j <= ny_local; j++) 
			u[0][j] = u[1][j];

    int Ny_inner = Ny - 2;
	int outlet_jmax = Ny_inner / 2;

	// only processes next to right wall
	if (i_start + nx_local == Nx - 1) {
		// loop through interior nodes
		for (int j = 1; j <= ny_local; j++) {
			// only apply BC if node is on the boundary
			int j_global = j_start + (j - 1);
			if (j_global <= outlet_jmax) {
				u[nx_local + 1][j] = u[nx_local][j];
			}
		}
	}

    // obstacle handling: zero out velocities in obstacle region
    if (ADD_OBSTACLE) {
        for (int i = 1; i <= nx_local; i++) {
            for (int j = 1; j <= ny_local; j++) {
                int ig = i_start + (i - 1);
                int jg = j_start + (j - 1);
                if (is_obstacle(ig, jg)) {
                    u[i][j] = 0.0;
                    v[i][j] = 0.0;
                }
            }
        }
    }
}

double project_velocity(void)
{
	double vmax = 0.0;
	for (int i = 1; i <= nx_local; i++)
		for (int j = 1; j <= ny_local; j++)
		{
			u[i][j] = u[i][j] - dt * (1. / rho) * (P[i + 1][j] - P[i - 1][j]) / (2. * dx);
			v[i][j] = v[i][j] - dt * (1. / rho) * (P[i][j + 1] - P[i][j - 1]) / (2. * dy);

			double vel = sqrt(u[i][j] * u[i][j] + v[i][j] * v[i][j]);

			vmax = max(vmax, vel);
		}

	return vmax;
}


void solve_NS(void)
{
	double vel_max = 0.0;
	int time_it = 0;
	int its = 0;
	int out_it = 0;
	double t_out = dt_out;

	grids_to_file(out_it);

	while (t < t_end)
	{
		if (vel_max > 0.0)
		{
			dt = min(courant * min(dx, dy) / vel_max, dt_min);
		}
		else dt = dt_min;

		t += dt;
		time_it++;
		swap(u, u_old);
		swap(v, v_old);

        // communicate velocities, since were updated by the swap
		exchange(u_old);
		exchange(v_old);

		calculate_intermediate_velocity();

		// communicate velocities, since were updated by calculate_intermediate_velocity
		exchange(u);
		exchange(v);

		calculate_ppm_RHS_central();
		its = pressure_poisson_jacobi(1.e-5);

		// communicate pressure, since was updated by pressure_poisson_jacobi
		exchange(P);

		// reduce max velocity
		double local_vmax = project_velocity();
		MPI_Allreduce(&local_vmax, &vel_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		set_velocity_BCs();

		// communicate u and v, since was updated by set_velocity_BCs
		exchange(u);
        exchange(v);

		if (t >= t_out)
		{
			out_it++;
			t_out += dt_out;
			grids_to_file(out_it);

			if (id == 0) {
				cout << time_it << ": " << t << " Jacobi iterations: " << its << ", vel_max: " << vel_max << endl;
			}
		}
	}
}

void free_memory(void) {
	free_continuous_array(P, P_1d);
	free_continuous_array(P_old, P_old_1d);
	free_continuous_array(u, u_1d);
	free_continuous_array(u_old, u_old_1d);
	free_continuous_array(v, v_1d);
	free_continuous_array(v_old, v_old_1d);
	free_continuous_array(PPrhs, PPrhs_1d);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

    setup_decomposition();
    setup();
	solve_NS();
	free_memory();

    MPI_Finalize();
}

