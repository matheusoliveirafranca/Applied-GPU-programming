#ifndef PARTICLES_H
#define PARTICLES_H

#include <math.h>

#include "Alloc.h"
#include "Parameters.h"
#include "PrecisionTypes.h"
#include "Grid.h"
#include "EMfield.h"
#include "InterpDensSpecies.h"


struct particles {
    
    /** species ID: 0, 1, 2 , ... */
    int species_ID;
    
    /** maximum number of particles of this species on this domain. used for memory allocation */
    long npmax;
    /** number of particles of this species on this domain */
    long nop;
    
    /** Electron and ions have different number of iterations: ions moves slower than ions */
    int NiterMover;
    /** number of particle of subcycles in the mover */
    int n_sub_cycles;
    
    
    /** number of particles per cell */
    int npcel;
    /** number of particles per cell - X direction */
    int npcelx;
    /** number of particles per cell - Y direction */
    int npcely;
    /** number of particles per cell - Z direction */
    int npcelz;
    
    
    /** charge over mass ratio */
    FPpart qom;
    
    /* drift and thermal velocities for this species */
    FPpart u0, v0, w0;
    FPpart uth, vth, wth;
    
    /** particle arrays: 1D arrays[npmax] */
    FPpart* x; FPpart*  y; FPpart* z; FPpart* u; FPpart* v; FPpart* w;
    /** q must have precision of interpolated quantities: typically double. Not used in mover */
    FPinterp* q;
    
    
    
};

/** allocate particle arrays */
void particle_allocate(struct parameters*, struct particles*, int);

/** deallocate */
void particle_deallocate(struct particles*);

/** particle mover */
int mover_PC_cpu(struct particles*, struct EMfield*, struct grid*, struct parameters*);

/** Interpolation Particle --> Grid: This is for species */
void interpP2G_cpu(struct particles*, struct interpDensSpecies*, struct grid*);

__global__ void mover_PC_kernel(FPpart* part_x_gpu  , FPpart* part_y_gpu  , FPpart* part_z_gpu  ,
                                FPpart* part_u_gpu  , FPpart* part_v_gpu  , FPpart* part_w_gpu  ,
                                FPfield* Ex_flat_gpu , FPfield* Ey_flat_gpu , FPfield* Ez_flat_gpu ,
                                FPfield* Bxn_flat_gpu, FPfield* Byn_flat_gpu, FPfield* Bzn_flat_gpu,
                                FPfield* XN_flat_gpu , FPfield* YN_flat_gpu , FPfield* ZN_flat_gpu ,
                                int nop   , int n_sub_cycles, int NiterMover, FPpart dt_sub_cycling, 
                                FPpart dto2, FPpart qomdt2, struct grid grd, struct parameters param);

int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param,
                  FPpart* part_x_gpu  , FPpart* part_y_gpu     , FPpart* part_z_gpu     , FPpart* part_u_gpu   , 
                  FPpart* part_v_gpu  , FPpart* part_w_gpu     , FPfield* Ex_flat_gpu   , FPfield* Ey_flat_gpu , 
                  FPfield* Ez_flat_gpu, FPfield* Bxn_flat_gpu  , FPfield* Byn_flat_gpu  , FPfield* Bzn_flat_gpu, 
                  FPfield* XN_flat_gpu, FPfield* YN_flat_gpu   , FPfield* ZN_flat_gpu   , int field_size    , int grd_size);

__global__ void interpP2G_kernel(FPpart* part_x_gpu   , FPpart* part_y_gpu   , FPpart* part_z_gpu  ,
                                 FPpart* part_u_gpu   , FPpart* part_v_gpu   , FPpart* part_w_gpu  ,
                                 FPinterp *part_q_gpu , FPfield* XN_flat_gpu , FPfield* YN_flat_gpu, 
                                 FPfield* ZN_flat_gpu , FPinterp *rhon_gpu   , FPinterp *rhoc_gpu  , 
                                 FPinterp *Jx_gpu     , FPinterp *Jy_gpu     , FPinterp *Jz_gpu    , 
                                 FPinterp *pxx_gpu    , FPinterp *pxy_gpu    , FPinterp *pxz_gpu   , 
                                 FPinterp *pyy_gpu    , FPinterp *pyz_gpu    , FPinterp *pzz_gpu   , 
                                 int nop  , struct grid grd);

void interpP2G_gpu(struct particles* part , struct interpDensSpecies* ids, struct grid* grd, FPpart* part_x_gpu  , 
                    FPpart* part_y_gpu    , FPpart* part_z_gpu     , FPpart* part_u_gpu    , FPpart* part_v_gpu  , 
                    FPpart* part_w_gpu    , FPinterp *part_q_gpu   , FPinterp *Jx_gpu      , FPinterp *Jy_gpu    , 
                    FPinterp *Jz_gpu      , FPinterp *pxx_gpu      , FPinterp *pxy_gpu     , FPinterp *pxz_gpu   , 
                    FPinterp *pyy_gpu     , FPinterp *pyz_gpu      , FPinterp *pzz_gpu     , FPinterp *rhon_gpu  , 
                    FPinterp *rhoc_gpu    , FPfield* XN_flat_gpu   , FPfield* YN_flat_gpu  , FPfield* ZN_flat_gpu, 
                    int grd_size);


#endif
