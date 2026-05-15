# Parallel Navier-Stokes: Compile, Run, and Extensions

## Compile
Use the following command on the target system:

```bash
mpicxx -O2 -o pns Parallel_Navier-Stokes.cpp \
  -I/opt/homebrew/opt/hdf5-mpi/include \
  -L/opt/homebrew/opt/hdf5-mpi/lib \
  -lhdf5 -lm
```

## Run simulation
Run the solver with 4 MPI processes:

```bash
mpiexec -n 4 ./pns
```

The solver writes output to a single shared fields.h5 file using parallel HDF5. At each output step, every MPI process writes its local interior block of P, u, and v into the correct location of a global dataset by using hyperslab offsets and local block sizes derived from the domain decomposition. Because each local tile is inserted into its global position during writing, the full field is reconstructed automatically in the file and no separate gather or merge stage is required.

The domain size (width and height), as well as the number of nodes in x- and y-directions is adjustable via `Nx, Ny, Lx, Ly` parameters.

## Run post-processing
Run the Python visualisation script to create a GIF of the velocity field evolution:

```bash
python3 visualise.py
```

A pre-generated animation is included as `fields_evolution.gif`.

## Extensions and improvements

- Field arrays are stored in contiguous memory blocks with a 2D indexing interface to improve cache locality.
- A rectangular 2D domain decomposition is used for any number of MPI processes, with the process grid chosen to keep subdomains close to the aspect ratio of the full domain.
- A custom MPI derived datatype is used to communicate non-contiguous boundary rows directly from memory.
- MPI_Allreduce is used to compute global values such as the maximum velocity for timestep control, and the pressure-solver residual for Jacobi convergence to keep all processes synchronized.
- PERIODIC_BC provides a switch for periodic top and bottom boundaries by wrapping halo communication between the first and last process rows.
- ADD_OBSTACLE provides a switch for inserting a rectangular internal obstacle, with size and position controlled by `OBS_I_MIN`, `OBS_I_MAX`, `OBS_J_MIN`, and `OBS_J_MAX`.