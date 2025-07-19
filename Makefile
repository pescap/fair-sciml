# Makefile for FAIR Scientific Machine Learning - FEniCS Simulator

.PHONY: help build test run clean shell

# Default target
help:
	@echo "FAIR Scientific Machine Learning - FEniCS Simulator"
	@echo ""
	@echo "Available commands:"
	@echo "  make build    - Build the FEniCS Docker image"
	@echo "  make test     - Test the simulator setup"
	@echo "  make run      - Run a Poisson simulation (default)"
	@echo "  make shell    - Enter the Docker container shell"
	@echo "  make clean    - Clean up Docker images and containers"
	@echo ""
	@echo "Examples:"
	@echo "  make run SIMULATOR=biharmonic NUM_SIMS=5"
	@echo "  make run SIMULATOR=helmholtz MESH_SIZE=64"

# Build the Docker image
build:
	@echo "Building FEniCS simulator Docker image..."
	./scripts/build_fenics.sh

# Test the setup
test:
	@echo "Testing FEniCS simulator setup..."
	./scripts/test_simulator.sh

# Run simulations
run:
	@echo "Running $(SIMULATOR) simulator..."
	./scripts/run_simulator.sh $(SIMULATOR) $(NUM_SIMS) $(MESH_SIZE) $(OUTPUT_DIR)

# Enter container shell
shell:
	@echo "Entering FEniCS simulator container..."
	docker run -it --rm \
		-v $(PWD)/src:/app/src:ro \
		-v $(PWD)/utils:/app/utils:ro \
		-v $(PWD)/simulations:/app/simulations \
		-v $(PWD)/data:/app/data \
		-e PYTHONPATH=/app \
		fair-sciml-fenics:latest

# Clean up
clean:
	@echo "Cleaning up Docker resources..."
	docker rmi fair-sciml-fenics:latest 2>/dev/null || true
	docker system prune -f

# Quick Poisson simulation
poisson:
	@echo "Running quick Poisson simulation..."
	./scripts/run_simulator.sh poisson 5 16 quick_test

# Quick Biharmonic simulation
biharmonic:
	@echo "Running quick Biharmonic simulation..."
	./scripts/run_simulator.sh biharmonic 3 16 quick_test

# Quick Helmholtz simulation
helmholtz:
	@echo "Running quick Helmholtz simulation..."
	./scripts/run_simulator.sh helmholtz 3 16 quick_test

# Show status
status:
	@echo "Docker images:"
	docker images | grep fair-sciml-fenics || echo "No fair-sciml-fenics image found"
	@echo ""
	@echo "Available containers:"
	docker ps -a | grep fair-sciml-fenics || echo "No fair-sciml-fenics containers found" 