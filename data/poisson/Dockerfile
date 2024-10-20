# Use the official FEniCS image as the base
FROM quay.io/fenicsproject/stable:current

# Set the working directory inside the container
WORKDIR /app

# Copy the local script (poisson_sims.py) and any other necessary files to the container
COPY poisson_sims.py /app/

# Install additional dependencies (e.g., h5py, psutil, py-cpuinfo)
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --no-cache-dir h5py psutil py-cpuinfo numpy

# Define environment variables for FEniCS (optional)
ENV MPI_EXEC=/usr/bin/mpiexec

# Define the command to run the simulation script with arguments
ENTRYPOINT ["python3", "poisson_sims.py"]
