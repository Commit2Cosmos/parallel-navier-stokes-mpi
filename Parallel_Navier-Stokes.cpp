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

const int Nx = 202;
const int Ny = 102;

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
const bool ADD_OBSTACLE = false;
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

// Abortы the MPI program if an HDF5 call reports an error
static void h5_check(herr_t status, const char* msg)
{
    if (status < 0) {
        std::cerr << "Rank " << id << " HDF5 error: " << msg << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Writeы a scalar double attribute to an HDF5 object
 static void write_double_attr_collective(hid_t loc, const char* name, double val)
{
    // Create a scalar dataspace because the attribute contains one double value
    hid_t sp = H5Screate(H5S_SCALAR);

    // Create the attribute on the target object using native double storage
    hid_t at = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT, H5P_DEFAULT);
    h5_check(at, name);

    // Write the scalar value into the newly created attribute
    h5_check(H5Awrite(at, H5T_NATIVE_DOUBLE, &val), name);

    // Close HDF5 handles
    H5Aclose(at);
    H5Sclose(sp);
}


// Write a scalar integer attribute to an HDF5 object
static void write_int_attr_collective(hid_t loc, const char* name, int val)
{
    // Create a scalar dataspace because the attribute contains one integer value
    hid_t sp = H5Screate(H5S_SCALAR);

    // Create the attribute on the target object using native integer storage
    hid_t at = H5Acreate2(loc, name, H5T_NATIVE_INT, sp, H5P_DEFAULT, H5P_DEFAULT);
    h5_check(at, name);

    // Write the scalar value into the newly created attribute
    h5_check(H5Awrite(at, H5T_NATIVE_INT, &val), name);

    // Close HDF5 handles
    H5Aclose(at);
    H5Sclose(sp);
}

// Write the current pressure and velocity fields to the shared HDF5 file
void grids_to_file(int outit)
{
    // Create a file-access property list so HDF5 uses MPI-IO collectively
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    h5_check(fapl, "H5Pcreate(H5P_FILE_ACCESS)");
    h5_check(H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL), "H5Pset_fapl_mpio");

    hid_t file_id;

    // On the first output, create a new file from scratch
    if (outit == 0) {
        file_id = H5Fcreate("fields.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        h5_check(file_id, "H5Fcreate(fields.h5)");
    } else {
        // On later outputs, reopen the existing file
        file_id = H5Fopen("fields.h5", H5F_ACC_RDWR, fapl);
        h5_check(file_id, "H5Fopen(fields.h5)");
    }

    // The file-access property list is no longer needed once the file is open
    H5Pclose(fapl);

    // Create geometry once, collectively, on the first output only
    if (outit == 0) {
        hid_t geo_grp = H5Gcreate2(file_id, "/geometry",
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        h5_check(geo_grp, "H5Gcreate2(/geometry)");

        // Allocate arrays for the global grid in x and y
        std::vector<double> xcoords(Nx), ycoords(Ny);

        // Fill x and y using uniform spacing
        for (int i = 0; i < Nx; i++) xcoords[i] = i * dx;
        for (int j = 0; j < Ny; j++) ycoords[j] = j * dy;

        // Define 1D dataspaces for the global coordinate arrays
        hsize_t dimx[1] = {static_cast<hsize_t>(Nx)};
        hsize_t dimy[1] = {static_cast<hsize_t>(Ny)};
        hid_t sp_x = H5Screate_simple(1, dimx, NULL);
        hid_t sp_y = H5Screate_simple(1, dimy, NULL);

        // Create datasets to store the x and y coordinate arrays
        hid_t ds_x = H5Dcreate2(geo_grp, "x_coords", H5T_NATIVE_DOUBLE,
                                sp_x, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t ds_y = H5Dcreate2(geo_grp, "y_coords", H5T_NATIVE_DOUBLE,
                                sp_y, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        h5_check(ds_x, "H5Dcreate2(x_coords)");
        h5_check(ds_y, "H5Dcreate2(y_coords)");

        // Use a collective transfer property so all ranks participate consistently
        hid_t dxpl_meta = H5Pcreate(H5P_DATASET_XFER);
        h5_check(dxpl_meta, "H5Pcreate(H5P_DATASET_XFER)");
        h5_check(H5Pset_dxpl_mpio(dxpl_meta, H5FD_MPIO_COLLECTIVE), "H5Pset_dxpl_mpio geometry");

        // Write the full x- and y-coordinate array to /geometry/x_coords and /geometry/y_coords
        h5_check(H5Dwrite(ds_x, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, dxpl_meta, xcoords.data()),
                 "H5Dwrite(x_coords)");
        h5_check(H5Dwrite(ds_y, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, dxpl_meta, ycoords.data()),
                 "H5Dwrite(y_coords)");

        H5Pclose(dxpl_meta);

        // Store metadata as attributes on the geometry group
        write_double_attr_collective(geo_grp, "Lx", Lx);
        write_double_attr_collective(geo_grp, "Ly", Ly);
        write_double_attr_collective(geo_grp, "rho", rho);
        write_double_attr_collective(geo_grp, "nu", nu);
        write_int_attr_collective(geo_grp, "Nx", Nx);
        write_int_attr_collective(geo_grp, "Ny", Ny);

        // Close datasets
        H5Dclose(ds_x);
        H5Dclose(ds_y);
        H5Sclose(sp_x);
        H5Sclose(sp_y);
        H5Gclose(geo_grp);
    }

    // Build the timestep group name
    char grpname[32];
    std::snprintf(grpname, sizeof(grpname), "/step_%04d", outit);

    // Create a group for the output step
    hid_t step_grp = H5Gcreate2(file_id, grpname,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5_check(step_grp, "H5Gcreate2(step)");

    // Store timestep metadata on the step group
    write_double_attr_collective(step_grp, "time", t);
    write_double_attr_collective(step_grp, "dt", dt);
    write_int_attr_collective(step_grp, "output_index", outit);

    // Writes one field into the current timestep group
    auto write_field = [&](const char* name, double** field) {
        // Define the full global dataset size (interior points only)
        hsize_t gdims[2] = {
            static_cast<hsize_t>(Ny - 2),
            static_cast<hsize_t>(Nx - 2)
        };
        hid_t filespace = H5Screate_simple(2, gdims, NULL);

        // Create current field dataset
        hid_t dset = H5Dcreate2(step_grp, name, H5T_NATIVE_DOUBLE,
                                filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        h5_check(dset, name);

        // Compute where this rank's local block begins in the global dataset
        hsize_t offset[2] = {
            static_cast<hsize_t>(j_start - 1),
            static_cast<hsize_t>(i_start - 1)
        };

        // Compute the extent of this rank's local interior block to be written
        hsize_t count[2] = {
            static_cast<hsize_t>(ny_local),
            static_cast<hsize_t>(nx_local)
        };

        // Select the target hyperslab in the global dataset for this rank
        h5_check(H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL),
                "H5Sselect_hyperslab");

        // Pack the local interior field values into row-major buffer
        std::vector<double> buf(static_cast<size_t>(nx_local) * static_cast<size_t>(ny_local));

        // Loop over local interior rows
        for (int j = 1; j <= ny_local; ++j) {
            // Loop over local interior columns
            for (int i = 1; i <= nx_local; ++i) {
                // Conversion between local indices into 1D buffer index
                buf[(j - 1) * nx_local + (i - 1)] = field[i][j];
            }
        }

        // Define the local memory dataspace
        hsize_t memdims[2] = {
            static_cast<hsize_t>(ny_local),
            static_cast<hsize_t>(nx_local)
        };
        hid_t memspace = H5Screate_simple(2, memdims, NULL);

        // Parallel I/O for field write
        hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
        h5_check(dxpl, "H5Pcreate(H5P_DATASET_XFER)");
        h5_check(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE), "H5Pset_dxpl_mpio field");

        // Write this rank's packed field block from memory into the selected global hyperslab
        h5_check(H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl, buf.data()), name);

        // Close all HDF5 handles
        H5Pclose(dxpl);
        H5Sclose(memspace);
        H5Sclose(filespace);
        H5Dclose(dset);
    };

    // Write fields for this output step
    write_field("P", P);
    write_field("u", u);
    write_field("v", v);

    // Close the timestep group and the output file
    H5Gclose(step_grp);
    H5Fclose(file_id);
}


// Allocates a 2D array interface with one contiguous 1D memory block
void setup_continuous_array(double**& array_2d, double*& array_1d, int m, int n)
{
    // Contiguous to store all values
    array_1d = new double[m * n];

    // Array of row pointers used for 2D indexing
    array_2d = new double* [m];

    // For each row i, point array_2d[i] to the start of row i inside the 1D block
    for (int i = 0; i < m; i++)
        array_2d[i] = &array_1d[i * n];
}

// Frees a 2D array created with setup_continuous_array()
void free_continuous_array(double**& array_2d, double*& array_1d)
{
    delete[] array_1d;
    delete[] array_2d;
}

// Selects decomposition for the global interior domain
void find_tiles(int p, int Nx_inner, int Ny_inner, int &px, int &py) {
    // Start with a very large cost so the first valid decomposition is accepted
    double best_cost = 1e18;

    // Try every possible number of process tiles in the x-direction
    for (int candidate_px = 1; candidate_px <= p; candidate_px++) {
        // Skip values that do not divide the total number of processes exactly
        if (p % candidate_px != 0) continue;

        // Calculate py
        int candidate_py = p / candidate_px;

        // Compute a simple cost that prefers tiles with most similar aspect ratio to the global domain
        double cost = (double)Nx_inner / candidate_px
                    + (double)Ny_inner / candidate_py;

        // Keep candidate with lower cost as the best decomposition so far
        if (cost < best_cost) {
            best_cost = cost;
            px = candidate_px;
            py = candidate_py;
        }
    }
}

// Performs current rank's domain decomposition
void setup_decomposition(void) 
{
    // Choose the 2D process-grid shape that will for tiling the interior domain
    find_tiles(p, Nx-2, Ny-2, px, py);

    // Calculate ranks
    rank_col = id % px;
    rank_row = id / px;

    // Interior grid points in the x-direction
    int Nx_inner = Nx - 2;

    // Base number of x-points assigned to every process column
    int base_x = Nx_inner / px;

    // Number of leftover x-points after even division
    int rem_x = Nx_inner % px;

    // Give one extra x-point to the first rem_x process columns
    nx_local = base_x + (rank_col < rem_x ? 1 : 0);

    // Compute the number of interior grid points available in the y-direction
    int Ny_inner = Ny - 2;

    // Base number of y-points assigned to every process row
    int base_y = Ny_inner / py;

    // Number of leftover y-points after even division
    int rem_y = Ny_inner % py;

    // Give one extra y-point to the first rem_y process rows
    ny_local = base_y + (rank_row < rem_y ? 1 : 0);

    // Compute the first global interior x-/y-index
    // +1 accounts for the outer ghost/boundary layer
    // rank_col * base_x skips the uniformly distributed blocks before this rank
    // min(rem_x, rank_col) adds the extra cells already given to earlier ranks
    i_start = 1 + rank_col*base_x + min(rem_x, rank_col);
    j_start = 1 + rank_row*base_y + min(rem_y, rank_row);
}

// Creates an MPI datatype representing one fixed-j row
MPI_Datatype create_row_datatype(double** data, int j_fixed, double* buf_base)
{
    // Create one block-length entry per local x-position
    vector<int> block_lengths(nx_local, 1);

    // Set type to MPI_DOUBLE
    vector<MPI_Datatype> types(nx_local, MPI_DOUBLE);

    // Byte displacement of each row element wrt the base address
    vector<MPI_Aint> offsets(nx_local);

    MPI_Aint base_addr;
    MPI_Get_address(buf_base, &base_addr);

    // Loop over every local x-index in the requested fixed-j row
    for (int i = 1; i <= nx_local; i++) {
        MPI_Aint addr;

        // Get actual address of element
        MPI_Get_address(&data[i][j_fixed], &addr);

        // Store the byte offset
        offsets[i - 1] = addr - base_addr;
    }

    MPI_Datatype face_type;

    // Create a struct datatype
    MPI_Type_create_struct(nx_local, block_lengths.data(), offsets.data(), types.data(), &face_type);

    // Commit the datatype
    MPI_Type_commit(&face_type);
    return face_type;
}

// Exchanges one layer of halo/ghost-cell data with neighbouring MPI ranks
void exchange(double** data)
{
    // Determine the rank of left/right neighbours
    int left = (rank_col > 0) ? rank_row * px + (rank_col - 1) : MPI_PROC_NULL;
    int right = (rank_col < px - 1) ? rank_row * px + (rank_col + 1) : MPI_PROC_NULL;

    int bottom, top;

    // Determine the rank of top/bot neighbours; wrap if periodic
    if (PERIODIC_BC) {
        bottom = ((rank_row - 1 + py) % py) * px + rank_col;
        top = ((rank_row + 1) % py) * px + rank_col;
    } else {
        bottom = (rank_row > 0) ? (rank_row - 1) * px + rank_col : MPI_PROC_NULL;
        top = (rank_row < py - 1) ? (rank_row + 1) * px + rank_col : MPI_PROC_NULL;
    }

    // Store all requests
    MPI_Request reqs[8];
    int count = 0;

    // Create derived datatypes for bottom/top ghost rows to be sent/received
    MPI_Datatype recv_bottom_type = create_row_datatype(data, 0, &data[1][0]);
    MPI_Datatype recv_top_type = create_row_datatype(data, ny_local + 1, &data[1][ny_local + 1]);
    MPI_Datatype send_bottom_type = create_row_datatype(data, 1, &data[1][1]);
    MPI_Datatype send_top_type = create_row_datatype(data, ny_local, &data[1][ny_local]);

    // Receive the left ghost column at x = 0 for interior y-indices 1..ny_local
    MPI_Irecv(&data[0][1], ny_local, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &reqs[count++]);

    // Receive the right ghost column at x = nx_local + 1 for interior y-indices 1..ny_local
    MPI_Irecv(&data[nx_local + 1][1], ny_local, MPI_DOUBLE, right, 1, MPI_COMM_WORLD, &reqs[count++]);

    // Receive the bottom ghost row at y = 0 for interior x-indices 1..nx_local
    MPI_Irecv(&data[1][0], 1, recv_bottom_type, bottom, 2, MPI_COMM_WORLD, &reqs[count++]);

    // Receive the top ghost row at y = ny_local + 1 for interior x-indices 1..nx_local
    MPI_Irecv(&data[1][ny_local + 1], 1, recv_top_type, top, 3, MPI_COMM_WORLD, &reqs[count++]);

    // Send the left interior boundary column at x = 1 to the left neighbour
    MPI_Isend(&data[1][1], ny_local, MPI_DOUBLE, left, 1, MPI_COMM_WORLD, &reqs[count++]);

    // Send the right interior boundary column at x = nx_local to the right neighbour
    MPI_Isend(&data[nx_local][1], ny_local, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &reqs[count++]);

    // Send the bottom interior boundary row at y = 1 to the bottom neighbour
    MPI_Isend(&data[1][1], 1, send_bottom_type, bottom, 3, MPI_COMM_WORLD, &reqs[count++]);

    // Send the top interior boundary row at y = ny_local to the top neighbour
    MPI_Isend(&data[1][ny_local], 1, send_top_type, top, 2, MPI_COMM_WORLD, &reqs[count++]);

    // Wait until all coms to finish
    MPI_Waitall(count, reqs, MPI_STATUSES_IGNORE);

    // Free datatypes
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

// Allocates and initialises all local field arrays for the current MPI rank
void setup(void)
{
    // Include one ghost-cell layer on each side of the local interior domain
    int m = nx_local + 2;
    int n = ny_local + 2;

    // Allocate storage for the fields
    setup_continuous_array(P, P_1d, m, n);
    setup_continuous_array(P_old, P_old_1d, m, n);
    setup_continuous_array(u, u_1d, m, n);
    setup_continuous_array(u_old, u_old_1d, m, n);
    setup_continuous_array(v, v_1d, m, n);
    setup_continuous_array(v_old, v_old_1d, m, n);
    setup_continuous_array(PPrhs, PPrhs_1d, m, n);

    // Initialise fields + ghost cells to zero everywhere
    fill(P_1d, P_1d + m*n, 0.0);
    fill(P_old_1d, P_old_1d + m*n, 0.0);
    fill(u_1d, u_1d + m*n, 0.0);
    fill(u_old_1d, u_old_1d + m*n, 0.0);
    fill(v_1d, v_1d + m*n, 0.0);
    fill(v_old_1d, v_old_1d + m*n, 0.0);
    fill(PPrhs_1d, PPrhs_1d + m*n, 0.0);

    // Compute the global grid spacing in the x-/y-direction
    dx = Lx / (Nx - 1);
    dy = Ly / (Ny - 1);

    // If this rank touches the left physical boundary, impose the inlet pressure
    // in the left ghost layer for all local y positions, including ghost rows
    if (i_start == 1)
        for (int j = 0; j <= ny_local+1; j++)
            P[0][j] = P_max;

    // Copy the initial pressure field into P_old so both pressure arrays start consistent
    memcpy(P_old_1d, P_1d, m * n * sizeof(double));

    // Initialise the simulation time to zero
    t = 0.0;
}

// Computes the right-hand side of the pressure Poisson equation using central differences for the velocity divergence
void calculate_ppm_RHS_central(void)
{
    // Loop over all local interior x-indices
    for (int i = 1; i <= nx_local; i++) {
        // Loop over all local interior y-indices
        for (int j = 1; j <= ny_local; j++) {
            // Convert the local i-index to global x-index
            int ig = i_start + (i - 1);

            // Convert the local j-index to global y-index
            int jg = j_start + (j - 1);

            // If this cell lies inside the obstacle, force the RHS to zero and skip the divergence calculation
            if (ADD_OBSTACLE && is_obstacle(ig, jg)) {
                PPrhs[i][j] = 0.0;
                continue;
            }

            // Read the neighbouring x-velocity values
            double uxp = u[i + 1][j];
            double uxm = u[i - 1][j];

            // Read the neighbouring y-velocity values
            double vyp = v[i][j + 1];
            double vym = v[i][j - 1];

            // If obstacle handling is enabled, modify the stencil when a neighbour
            // lies inside the obstacle by replacing that neighbour value with the
            // current-cell value -> cancelling that side's contribution
            if (ADD_OBSTACLE) {
                if (is_obstacle(ig + 1, jg)) uxp = u[i][j];
                if (is_obstacle(ig - 1, jg)) uxm = u[i][j];
                if (is_obstacle(ig, jg + 1)) vyp = v[i][j];
                if (is_obstacle(ig, jg - 1)) vym = v[i][j];
            }

            PPrhs[i][j] = rho / dt * ((uxp - uxm) / (2.0 * dx) + (vyp - vym) / (2.0 * dy));
        }
    }
}

// Solves the pressure Poisson equation using Jacobi iteration
int pressure_poisson_jacobi(double rtol = 1.e-5)
{
    // Initialise the residual measure above the stopping tolerance
    double tol = 10. * rtol;

    // How many Jacobi iterations have been performed
    int it = 0;

    // Continue iterating until the global relative residual is small enough
    while (tol > rtol)
    {
        // Swap current and old pressure pointers
        swap(P, P_old);

        // Exchange halo values
        exchange(P_old);

        double sum_val = 0.0;
        tol = 0.0;
        it++;

        // Perform one Jacobi update over all local interior cells
        for (int i = 1; i <= nx_local; i++) {
            for (int j = 1; j <= ny_local; j++)
            {
                // Apply the finite-difference Jacobi update for the pressure Poisson equation.
                P[i][j] = 1.0 / (2.0 + 2.0 * (dx * dx) / (dy * dy)) * (P_old[i + 1][j] + P_old[i - 1][j] +
                    (P_old[i][j + 1] + P_old[i][j - 1]) * (dx * dx) / (dy * dy)
                    - (dx * dx) * PPrhs[i][j]);

                sum_val += fabs(P[i][j]);

                tol += fabs(P[i][j] - P_old[i][j]);
            }
        }

        // Reapply boundary conditions
        set_pressure_BCs();

        // Pack the local residual sum and pressure norm for a global reduction (so only 1 communication is performed)
        double local_vals[2] = {tol, sum_val};
        double global_vals[2];

        // Sum across all MPI ranks so convergence is measured globally
        MPI_Allreduce(local_vals, global_vals, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Form the global relative residual
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

// Applies velocity boundary conditions on the local subdomain
void set_velocity_BCs(void)
{
    // Zero-gradient condition for u at the left pressure inlet boundary
    // Only ranks touching theboundary should update this layer
    if (i_start == 1)
        for (int j = 1; j <= ny_local; j++)
            u[0][j] = u[1][j];

    // Number of interior y-points in the full domain
    int Ny_inner = Ny - 2;

    // Determine the max global interior j-index that belongs to the outlet section (lower half of the right boundary)
    int outlet_jmax = Ny_inner / 2;

    // Only ranks touching the right physical boundary can apply the outlet condition
    if (i_start + nx_local == Nx - 1) {
        // Loop over all local interior y-indices along this rank's right edge
        for (int j = 1; j <= ny_local; j++) {
            // Convert the local j-index into the corresponding global y-index
            int j_global = j_start + (j - 1);

            // Apply the outlet zero-gradient condition only on the outlet section
            if (j_global <= outlet_jmax) {
                u[nx_local + 1][j] = u[nx_local][j];
            }
        }
    }

    // Zero velocity inside the obstacle region
    if (ADD_OBSTACLE) {
        // Loop over all local interior x-indices
        for (int i = 1; i <= nx_local; i++) {
            // Loop over all local interior y-indices
            for (int j = 1; j <= ny_local; j++) {
                // Calculate global x-/y-index.
                int ig = i_start + (i - 1);
                int jg = j_start + (j - 1);

                // If this local cell lies inside the obstacle, force both velocity
                // components to zero to impose a stationary internal boundary.
                if (is_obstacle(ig, jg)) {
                    u[i][j] = 0.0;
                    v[i][j] = 0.0;
                }
            }
        }
    }
}

// Projects intermediate velocity field using the pressure gradient
double project_velocity(void)
{
    // Track the largest velocity magnitude
    double vmax = 0.0;

    // Loop over all local interior x-indices
    for (int i = 1; i <= nx_local; i++)
        // Loop over all local interior y-indices
        for (int j = 1; j <= ny_local; j++)
        {
            // Read neighbouring pressure values needed for the central x-/y-grads
            double Pxp = P[i+1][j], Pxm = P[i-1][j];
            double Pyp = P[i][j+1], Pym = P[i][j-1];

            // Replace neightbour's value with the current-cell value if it's inside the obstacle
            if (ADD_OBSTACLE) {
                // Convert the local i-index into the corresponding global x-/y-index.
                int ig = i_start + (i - 1);
                int jg = j_start + (j - 1);

                // Right
                if (is_obstacle(ig + 1, jg)) Pxp = P[i][j];
                // Left
                if (is_obstacle(ig - 1, jg)) Pxm = P[i][j];
                // Top
                if (is_obstacle(ig, jg + 1)) Pyp = P[i][j];
                // Bottom
                if (is_obstacle(ig, jg - 1)) Pym = P[i][j];
            }

            u[i][j] -= dt * (1. / rho) * (Pxp - Pxm) / (2. * dx);
            v[i][j] -= dt * (1. / rho) * (Pyp - Pym) / (2. * dy);
            double vel = sqrt(u[i][j] * u[i][j] + v[i][j] * v[i][j]);
            vmax = max(vmax, vel);
        }

    return vmax;
}

// Advances the Navier-Stokes solution in time
void solve_NS(void)
{
    double vel_max = 0.0;
    int time_it = 0;
    int its = 0;
    int out_it = 0;
    double t_out = dt_out;

    // Write the initial condition fields before time integration begins
    grids_to_file(out_it);

    // Continue advancing the solution until the target final time is reached
    while (t < t_end)
    {
        if (vel_max > 0.0)
        {
            dt = min(courant * min(dx, dy) / vel_max, dt_min);
        }
        else dt = dt_min;

        t += dt;
        time_it++;

        // Swap velocity pointers so the old fields hold the previous timestep values
        // and the current fields can be overwritten with the new intermediate values
        swap(u, u_old);
        swap(v, v_old);

        // Exchange halo values for the previous x-/y-velocity fields after the swap
        exchange(u_old);
        exchange(v_old);

        calculate_intermediate_velocity();

        // Exchange halo values for the updated intermediate x-/y-velocity fields
        exchange(u);
        exchange(v);

        calculate_ppm_RHS_central();

        its = pressure_poisson_jacobi(1.e-5);

        // Exchange halo values for the updated pressure field
        exchange(P);

        double local_vmax = project_velocity();

        // Compute the global maximum speed across all ranks for timestep selection
        MPI_Allreduce(&local_vmax, &vel_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // Reapply velocity boundary conditions after projection
        set_velocity_BCs();

        // Exchange halo values for the final corrected x-/y-velocity fields
        exchange(u);
        exchange(v);

        if (t >= t_out)
        {
            out_it++;
            t_out += dt_out;

            // Write the current fields to the output file
            grids_to_file(out_it);

            // Progress info (only process 0)
            if (id == 0) {
                cout << time_it << ": " << t << " Jacobi iterations: " << its << ", vel_max: " << vel_max << endl;
            }
        }
    }
}

// Frees memory from field arrays
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

