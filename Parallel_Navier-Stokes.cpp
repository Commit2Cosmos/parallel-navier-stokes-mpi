#include <mpi.h>
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

const int Nx = 101;
const int Ny = 51;

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

void grids_to_file(int out)
{
	//Write the output for a single time step to file
	stringstream fname;
	fstream f1;
	fname << "./out/P" << "_" << out << ".dat";
	f1.open(fname.str().c_str(), ios_base::out);
	for (int i = 0; i <= nx_local; i++)
	{
		for (int j = 0; j <= ny_local; j++)
			f1 << P[i][j] << "\t";
		f1 << endl;
	}
	f1.close();
	fname.str("");
	fname << "./out/u" << "_" << out << ".dat";
	f1.open(fname.str().c_str(), ios_base::out);
	for (int i = 0; i <= nx_local; i++)
	{
		for (int j = 0; j <= ny_local; j++)
			f1 << u[i][j] << "\t";
		f1 << endl;
	}
	f1.close();
	fname.str("");
	fname << "./out/v" << "_" << out << ".dat";
	f1.open(fname.str().c_str(), ios_base::out);
	for (int i = 0; i <= nx_local; i++)
	{
		for (int j = 0; j <= ny_local; j++)
			f1 << v[i][j] << "\t";
		f1 << endl;
	}
	f1.close();
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

MPI_Datatype create_column_datatype(double** data, int col) {
	// number of elements in the block
	int block_lengths[nx_local];
	// datatype of each element
	MPI_Datatype types[nx_local];
	// byte offset to reach this block from anchor address
	MPI_Aint offsets[nx_local];

	// anchor address == address of first interior row
	MPI_Aint base_addr;
    MPI_Get_address(data[1], &base_addr);

	// loop over 
	for (int i = 0; i < nx_local; i++) {
		block_lengths[i] = 1;
		types[i] = MPI_DOUBLE;

		MPI_Aint addr;
    	MPI_Get_address(&data[i+1][col], &addr);

		offsets[i] = addr - base_addr;
	}

	// create type
	MPI_Datatype col_type;
	MPI_Type_create_struct(nx_local, block_lengths, offsets, types, &col_type);
	MPI_Type_commit(&col_type);
	return col_type;
}

// for compatibility, not strictly necessary since row data is already contiguous
MPI_Datatype create_row_datatype() {
	// number of elements in the block
	int block_lengths[1] = { ny_local };
	// datatype of each element
	MPI_Datatype types[1] = { MPI_DOUBLE };
	// byte offset to reach this block from anchor address
	MPI_Aint offsets[1] = { 0 };

	// create type
	MPI_Datatype row_type;
	MPI_Type_create_struct(1, block_lengths, offsets, types, &row_type);
	MPI_Type_commit(&row_type);
	return row_type;
}

void exchange(double** data) {
	int left = (rank_col > 0) ? rank_row * px + (rank_col - 1) : MPI_PROC_NULL;
    int right = (rank_col < px - 1) ? rank_row * px + (rank_col + 1) : MPI_PROC_NULL;
    int bottom = (rank_row > 0) ? (rank_row - 1) * px + rank_col : MPI_PROC_NULL;
    int top = (rank_row < py - 1) ? (rank_row + 1) * px + rank_col : MPI_PROC_NULL;

	MPI_Datatype send_left = create_column_datatype(data, 1);
    MPI_Datatype send_right = create_column_datatype(data, nx_local);
    MPI_Datatype recv_left = create_column_datatype(data, 0);
    MPI_Datatype recv_right = create_column_datatype(data, nx_local+1);
	
	// only need 1 row type for all communications
	MPI_Datatype row_type = create_row_datatype();
	
	MPI_Request reqs[8];
	int count = 0;
    int tag = 0;

	// Recvs
	// left
	MPI_Irecv(data[1], 1, recv_left, left, tag, MPI_COMM_WORLD, &reqs[count++]);
	// right
	MPI_Irecv(data[1], 1, recv_right, right, tag, MPI_COMM_WORLD, &reqs[count++]);
	// bottom
	MPI_Irecv(&data[0][1], 1, row_type, bottom, tag, MPI_COMM_WORLD, &reqs[count++]);
	// top
	MPI_Irecv(&data[ny_local+1][1], 1, row_type, top, tag, MPI_COMM_WORLD, &reqs[count++]);
	
	// Sends
	// left
	MPI_Isend(data[1], 1, send_left, left, tag, MPI_COMM_WORLD, &reqs[count++]);
	// right
	MPI_Isend(data[1], 1, send_right, right, tag, MPI_COMM_WORLD, &reqs[count++]);
	// bottom
	MPI_Isend(&data[1][1], 1, row_type, bottom, tag, MPI_COMM_WORLD, &reqs[count++]);
	// top
	MPI_Isend(&data[ny_local][1], 1, row_type, top, tag, MPI_COMM_WORLD, &reqs[count++]);

	// wait for all
	MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);

    // Free all types
    MPI_Type_free(&send_left);
	MPI_Type_free(&recv_left);
    MPI_Type_free(&send_right);
	MPI_Type_free(&recv_right);
    MPI_Type_free(&row_type);
}

void set_pressure_BCs(void) {
    // No slip over top and bottom walls, so zero normal gradient for pressure
    // Apply if processes top or bottom
    if (j_start == 1)
        for (int i = 1; i <= nx_local; i++)
			P[i][0] = P[i][1];

    if (j_start + ny_local == Ny - 1)
        for (int i = 1; i <= nx_local; i++)
			P[i][ny_local + 1] = P[i][ny_local];

    // No slip over half on the right wall, so zero normal gradient for pressure
    // Apply only if processes right wall
    if (i_start + nx_local == Nx - 1)
        for (int j = (Ny / 2) - j_start + 1; j <= ny_local + 1; j++)
    		P[nx_local + 1][j] = P[nx_local][j];
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

	set_pressure_BCs();

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
		double sum_val = 0.0;
		tol = 0.0;
		it++;

		//Jacobi iteration
		for (int i = 1; i <= nx_local; i++)
			for (int j = 1; j <= ny_local; j++)
			{
				P[i][j] = 1.0 / (2.0 + 2.0 * (dx * dx) / (dy * dy)) * (P_old[i + 1][j] + P_old[i - 1][j] + 
					(P_old[i][j + 1] + P_old[i][j - 1]) * (dx * dx) / (dy * dy)
					- (dx * dx) * PPrhs[i][j]);

				sum_val += fabs(P[i][j]);
				tol += fabs(P[i][j] - P_old[i][j]);
			}

		set_pressure_BCs();

		tol = tol / max(1.e-10, sum_val);
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

    if (i_start + nx_local == Nx - 1)
	    for (int j = 0; j <= ny_local / 2; j++) 
			u[nx_local + 1][j] = u[nx_local][j];
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
	int its;
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

		calculate_intermediate_velocity();
		calculate_ppm_RHS_central();
		its = pressure_poisson_jacobi(1.e-5);
		vel_max = project_velocity();
		set_velocity_BCs();

		if (t >= t_out)
		{
			out_it++;
			t_out += dt_out;
			cout << time_it << ": " << t << " Jacobi iterations: " << its << " vel_max: " << vel_max << endl;
			grids_to_file(out_it);
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

