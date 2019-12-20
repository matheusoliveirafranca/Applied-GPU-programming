/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"


#include <stdio.h>

// Cuda
#include <cuda.h>

int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);
    
    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp= 0.0;
    
    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield field;
    field_allocate(&grd,&field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd,&ids[is],is);
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    
    // Allocate Particles
    particles *part = new particles[param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&part[is],is);
    }
    
    // Initialization
    initGEM(&param,&grd,&field,&field_aux,part,ids);



    // **********************************************************//
    // ********************** GPU allocation ********************//
    // **********************************************************//

    //Particle vars
    FPpart* part_x_gpu, part_y_gpu, part_z_gpu;
    FPpart* part_u_gpu, part_v_gpu, part_w_gpu;
    //EMField vars
    FPfield* Ex_gpu, Ey_gpu, Ez_gpu;
    FPfield* Bxn_gpu, Byn_gpu, Bzn_gpu;
    //Grd vars
    FPfield* XN_gpu, YN_gpu, ZN_gpu;


    int field_size = grd.nxn * grd.nyn * grd.nzn;
    int grd_size   = grd.nxn * grd.nyn * grd.nzn;

    // Allocate GPU memory
    cudaMalloc(&part_x_gpu, part->npmax * sizeof(FPpart));
    cudaMalloc(&part_y_gpu, part->npmax * sizeof(FPpart));
    cudaMalloc(&part_z_gpu, part->npmax * sizeof(FPpart));
    cudaMalloc(&part_u_gpu, part->npmax * sizeof(FPpart));
    cudaMalloc(&part_v_gpu, part->npmax * sizeof(FPpart));
    cudaMalloc(&part_w_gpu, part->npmax * sizeof(FPpart));

    cudaMalloc(&Ex_gpu , field_size * sizeof(FPfield));
    cudaMalloc(&Ey_gpu , field_size * sizeof(FPfield));
    cudaMalloc(&Ez_gpu , field_size * sizeof(FPfield));
    cudaMalloc(&Bxn_gpu, field_size * sizeof(FPfield));
    cudaMalloc(&Byn_gpu, field_size * sizeof(FPfield));
    cudaMalloc(&Bzn_gpu, field_size * sizeof(FPfield));

    cudaMalloc(&XN_gpu, grd_size * sizeof(FPfield));    
    cudaMalloc(&YN_gpu, grd_size * sizeof(FPfield));
    cudaMalloc(&ZN_gpu, grd_size * sizeof(FPfield));

    // **********************************************************//

    
    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);
        
        
        // implicit mover
        iMover = cpuSecond(); // start timer for mover

        // **********************************************************//
        // *********************** GPU Version **********************//
        // **********************************************************//
        loop_mover_PC_gpu(&part[is],&field,&grd,&param);

        for (int is=0; is < param.ns; is++)
            // mover_PC_cpu(&part[is],&field,&grd,&param);
            mover_PC_gpu(&part[is] , &field,&grd,     &param, part_x_gpu, part_y_gpu, part_z_gpu, 
                         part_u_gpu, part_v_gpu , part_w_gpu, Ex_gpu    , Ey_gpu    , Ez_gpu    ,
                         Bxn_gpu   , Byn_gpu    , Bzn_gpu   , XN_gpu    , YN_gpu    , ZN_gpu    ,
                         field_size, grd_size);
            
        // **********************************************************//
        eMover += (cpuSecond() - iMover); // stop timer for mover
        
        
        
        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++)
            interpP2G(&part[is],&ids[is],&grd);
        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&grd,&param);
        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&param);
        
        
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd,ids,&idn);
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
        
    
    }  // end of one PIC cycle
    
    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }

    // GPU deallocate memory 
    cudaFree(part_x_gpu);
    cudaFree(part_y_gpu);
    cudaFree(part_z_gpu);
    cudaFree(part_u_gpu);
    cudaFree(part_v_gpu);
    cudaFree(part_w_gpu);

    cudaFree(Ex_gpu);
    cudaFree(Ey_gpu);
    cudaFree(Ez_gpu);
    cudaFree(Bxn_gpu);
    cudaFree(Byn_gpu);
    cudaFree(Bzn_gpu);

    cudaFree(XN_gpu);
    cudaFree(YN_gpu);
    cudaFree(ZN_gpu);
    
    
    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    // exit
    return 0;
}


