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

This produces the simulation output (fields.h5 file) used by the post-processing stage. The file contains the field values of each process per saved timestep. The reconstruction is achieved while writing to the HFD5 hyperslab (using metadata, such as offset and local count, so local tiles are placed correctly inside the global dataset). The ghost nodes are not saved.

The domain size (width and height), as well as the number of nodes in x- and y-directions is adjustable via `Nx, Ny, Lx, Ly` parameters.

## Run post-processing
Run the Python visualisation script to create a GIF of the velocity field evolution:

```bash
python3 visualise.py
```

## Extensions and improvements

- Field arrays were changed to contiguous storage to improve memory locality and simplify communication handling.
- Rectangular grid decomposition is used for any number of processes, with tiles chosen to stay as close as possible to the aspect ratio of the full domain.
- A custom MPI datatype is used to package non-contiguous boundary data for communication.
- `MPI_Allreduce` is used for communicating the maximum velocity (so all processes use the dase minimum timestep value) and tolerance (so each process runs the same number of jacobi iterations).
- `PERIODIC_BC` provides a toggle for making the top and bottom boundaries periodic.
- `ADD_OBSTACLE` provides a toggle for placing an obstacle inside the domain.