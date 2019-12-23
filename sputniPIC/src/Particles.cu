#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#define TPB 64

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

__global__ void mover_PC_kernel(FPpart* part_x_gpu  , FPpart* part_y_gpu  , FPpart* part_z_gpu  ,
                                FPpart* part_u_gpu  , FPpart* part_v_gpu  , FPpart* part_w_gpu  ,
                                FPfield* Ex_flat_gpu , FPfield* Ey_flat_gpu , FPfield* Ez_flat_gpu ,
                                FPfield* Bxn_flat_gpu, FPfield* Byn_flat_gpu, FPfield* Bzn_flat_gpu,
                                FPfield* XN_flat_gpu , FPfield* YN_flat_gpu , FPfield* ZN_flat_gpu ,
                                int nop   , int n_sub_cycles, int NiterMover, FPpart dt_sub_cycling, 
                                FPpart dto2, FPpart qomdt2, struct grid grd, struct parameters param){
    // thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > nop) return;

    // auxiliary variables
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    xptilde = part_x_gpu[i];
    yptilde = part_y_gpu[i];
    zptilde = part_z_gpu[i];
    
    // start subcycling
    for (int i_sub=0; i_sub <  n_sub_cycles; i_sub++){
            // calculate the average velocity iteratively
            for(int innter=0; innter < NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part_x_gpu[i] - grd.xStart)*grd.invdx);
                iy = 2 +  int((part_y_gpu[i] - grd.yStart)*grd.invdy);
                iz = 2 +  int((part_z_gpu[i] - grd.zStart)*grd.invdz);
                
                // calculate weights
                xi[0]   = part_x_gpu[i] - XN_flat_gpu[get_idx(ix - 1, iy, iz, grd.nyn, grd.nzn)];
                eta[0]  = part_y_gpu[i] - YN_flat_gpu[get_idx(ix, iy - 1, iz, grd.nyn, grd.nzn)];
                zeta[0] = part_z_gpu[i] - ZN_flat_gpu[get_idx(ix, iy, iz - 1, grd.nyn, grd.nzn)];
                xi[1]   = XN_flat_gpu[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - part_x_gpu[i];
                eta[1]  = YN_flat_gpu[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - part_y_gpu[i];
                zeta[1] = ZN_flat_gpu[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - part_z_gpu[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd.invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*Ex_flat_gpu[get_idx(ix- ii, iy -jj, iz- kk, grd.nyn, grd.nzn)];
                            Eyl += weight[ii][jj][kk]*Ey_flat_gpu[get_idx(ix- ii, iy -jj, iz- kk, grd.nyn, grd.nzn)];
                            Ezl += weight[ii][jj][kk]*Ez_flat_gpu[get_idx(ix- ii, iy -jj, iz- kk, grd.nyn, grd.nzn)];
                            Bxl += weight[ii][jj][kk]*Bxn_flat_gpu[get_idx(ix- ii, iy -jj, iz- kk, grd.nyn, grd.nzn)];
                            Byl += weight[ii][jj][kk]*Byn_flat_gpu[get_idx(ix- ii, iy -jj, iz- kk, grd.nyn, grd.nzn)];
                            Bzl += weight[ii][jj][kk]*Bzn_flat_gpu[get_idx(ix- ii, iy -jj, iz- kk, grd.nyn, grd.nzn)];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part_u_gpu[i] + qomdt2*Exl;
                vt= part_v_gpu[i] + qomdt2*Eyl;
                wt= part_w_gpu[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part_x_gpu[i] = xptilde + uptilde*dto2;
                part_y_gpu[i] = yptilde + vptilde*dto2;
                part_z_gpu[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part_u_gpu[i]= 2.0*uptilde - part_u_gpu[i];
            part_v_gpu[i]= 2.0*vptilde - part_v_gpu[i];
            part_w_gpu[i]= 2.0*wptilde - part_w_gpu[i];
            part_x_gpu[i] = xptilde + uptilde*dt_sub_cycling;
            part_y_gpu[i] = yptilde + vptilde*dt_sub_cycling;
            part_v_gpu[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part_x_gpu[i] > grd.Lx){
                if (param.PERIODICX==true){ // PERIODIC
                    part_x_gpu[i] = part_x_gpu[i] - grd.Lx;
                } else { // REFLECTING BC
                    part_u_gpu[i] = -part_u_gpu[i];
                    part_x_gpu[i] = 2*grd.Lx - part_x_gpu[i];
                }
            }
                                                                        
            if (part_x_gpu[i] < 0){
                if (param.PERIODICX==true){ // PERIODIC
                   part_x_gpu[i] = part_x_gpu[i] + grd.Lx;
                } else { // REFLECTING BC
                    part_u_gpu[i] = -part_u_gpu[i];
                    part_x_gpu[i] = -part_x_gpu[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part_y_gpu[i] > grd.Ly){
                if (param.PERIODICY==true){ // PERIODIC
                    part_y_gpu[i] = part_y_gpu[i] - grd.Ly;
                } else { // REFLECTING BC
                    part_v_gpu[i] = -part_v_gpu[i];
                    part_y_gpu[i] = 2*grd.Ly - part_y_gpu[i];
                }
            }
                                                                        
            if (part_y_gpu[i] < 0){
                if (param.PERIODICY==true){ // PERIODIC
                    part_y_gpu[i] = part_v_gpu[i] + grd.Ly;
                } else { // REFLECTING BC
                    part_v_gpu[i] = -part_v_gpu[i];
                    part_y_gpu[i] = -part_y_gpu[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part_z_gpu[i] > grd.Lz){
                if (param.PERIODICZ==true){ // PERIODIC
                    part_z_gpu[i] = part_z_gpu[i] - grd.Lz;
                } else { // REFLECTING BC
                    part_w_gpu[i] = -part_w_gpu[i];
                    part_z_gpu[i] = 2*grd.Lz - part_z_gpu[i];
                }
            }
                                                                        
            if (part_z_gpu[i] < 0){
                if (param.PERIODICZ==true){ // PERIODIC
                    part_z_gpu[i] = part_z_gpu[i] + grd.Lz;
                } else { // REFLECTING BC
                    part_w_gpu[i] = -part_w_gpu[i];
                    part_z_gpu[i] = -part_z_gpu[i];
                }
            }
                                                                        
    } // end of one particle


}


/** particle mover GPU */
int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param    ,
                 FPpart* part_x_gpu  , FPpart* part_y_gpu     , FPpart* part_z_gpu     , FPpart* part_u_gpu   , 
                 FPpart* part_v_gpu  , FPpart* part_w_gpu     , FPfield* Ex_flat_gpu   , FPfield* Ey_flat_gpu , 
                 FPfield* Ez_flat_gpu, FPfield* Bxn_flat_gpu  , FPfield* Byn_flat_gpu  , FPfield* Bzn_flat_gpu, 
                 FPfield* XN_flat_gpu, FPfield* YN_flat_gpu   , FPfield* ZN_flat_gpu   , int field_size , int grd_size)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
    
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    int n_particles = part->nop;

    // Copy CPU arrays to GPU
    cudaMemcpy(part_x_gpu, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_y_gpu, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_z_gpu, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMemcpy(part_u_gpu, part->u, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_v_gpu, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_w_gpu, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMemcpy(Ex_flat_gpu, field->Ex_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Ey_flat_gpu, field->Ey_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Ez_flat_gpu, field->Ez_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMemcpy(Bxn_flat_gpu, field->Bxn_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Byn_flat_gpu, field->Byn_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Bzn_flat_gpu, field->Bzn_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMemcpy(XN_flat_gpu, grd->XN_flat, grd_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(YN_flat_gpu, grd->YN_flat, grd_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(ZN_flat_gpu, grd->ZN_flat, grd_size * sizeof(FPfield), cudaMemcpyHostToDevice);

    mover_PC_kernel<<<(n_particles+TPB-1)/TPB, TPB>>>(part_x_gpu  , part_y_gpu  , part_z_gpu  ,
                                                      part_u_gpu  , part_v_gpu  , part_w_gpu  ,
                                                      Ex_flat_gpu , Ey_flat_gpu , Ez_flat_gpu ,
                                                      Bxn_flat_gpu, Byn_flat_gpu, Bzn_flat_gpu,
                                                      XN_flat_gpu , YN_flat_gpu , ZN_flat_gpu ,
                                                      part->nop   , part->n_sub_cycles, part->NiterMover,
                                                      dt_sub_cycling, dto2, qomdt2, *grd, *param);
    cudaDeviceSynchronize();

    // Copy GPU arrays back to CPU
    cudaMemcpy(part->x, part_x_gpu, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, part_y_gpu, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, part_z_gpu, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);

    cudaMemcpy(part->u, part_u_gpu, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, part_v_gpu, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, part_w_gpu, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);

    cudaMemcpy(field->Ex_flat, Ex_flat_gpu, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ey_flat, Ey_flat_gpu, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ez_flat, Ez_flat_gpu, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);

    cudaMemcpy(field->Bxn_flat, Bxn_flat_gpu, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Byn_flat, Byn_flat_gpu, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bzn_flat, Bzn_flat_gpu, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);

    cudaMemcpy(grd->XN_flat, XN_flat_gpu, grd_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(grd->YN_flat, YN_flat_gpu, grd_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(grd->ZN_flat, ZN_flat_gpu, grd_size * sizeof(FPfield), cudaMemcpyDeviceToHost);

    return(0); // exit successfully
}

/** particle mover */
int mover_PC_cpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover



/** Interpolation Particle --> Grid: This is for species */
void interpP2G_cpu(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{

    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}

__global__ void interpP2G_kernel(FPpart* part_x_gpu   , FPpart* part_y_gpu   , FPpart* part_z_gpu  ,
                                 FPpart* part_u_gpu   , FPpart* part_v_gpu   , FPpart* part_w_gpu  ,
                                 FPinterp *part_q_gpu , FPfield* XN_flat_gpu , FPfield* YN_flat_gpu, 
                                 FPfield* ZN_flat_gpu , FPinterp *rhon_gpu   , FPinterp *rhoc_gpu  , 
                                 FPinterp *Jx_gpu     , FPinterp *Jy_gpu     , FPinterp *Jz_gpu    , 
                                 FPinterp *pxx_gpu    , FPinterp *pxy_gpu    , FPinterp *pxz_gpu   , 
                                 FPinterp *pyy_gpu    , FPinterp *pyz_gpu    , FPinterp *pzz_gpu   , 
                                 int nop  , struct grid grd)
{

    // thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > nop) return;

        // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
        
    // determine cell: can we change to int()? is it faster?
    ix = 2 + int (floor((part_x_gpu[i] - grd.xStart) * grd.invdx));
    iy = 2 + int (floor((part_y_gpu[i] - grd.yStart) * grd.invdy));
    iz = 2 + int (floor((part_z_gpu[i] - grd.zStart) * grd.invdz));
    
    // distances from node
    xi[0]   = part_x_gpu[i] - XN_flat_gpu[get_idx(ix - 1, iy, iz, grd.nyn, grd.nzn)];
    eta[0]  = part_y_gpu[i] - YN_flat_gpu[get_idx(ix, iy - 1, iz, grd.nyn, grd.nzn)];
    zeta[0] = part_z_gpu[i] - ZN_flat_gpu[get_idx(ix, iz, iz - 1, grd.nyn, grd.nzn)];
    xi[1]   = XN_flat_gpu[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - part_x_gpu[i];
    eta[1]  = YN_flat_gpu[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - part_y_gpu[i];
    zeta[1] = ZN_flat_gpu[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - part_z_gpu[i];
    
    // calculate the weights for different nodes
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                weight[ii][jj][kk] = part_q_gpu[i] * xi[ii] * eta[jj] * zeta[kk] * grd.invVOL;
    
    //////////////////////////
    // add charge density
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&rhon_gpu[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);
    
    
    ////////////////////////////
    // add current density - Jx
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_u_gpu[i] * weight[ii][jj][kk];
    
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&Jx_gpu[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);
    
    
    ////////////////////////////
    // add current density - Jy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_v_gpu[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&Jy_gpu[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);

    
    
    
    ////////////////////////////
    // add current density - Jz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_w_gpu[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&Jz_gpu[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);
    
    
    ////////////////////////////
    // add pressure pxx
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_u_gpu[i] * part_u_gpu[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&pxx_gpu[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);

    
    
    ////////////////////////////
    // add pressure pxy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_u_gpu[i] * part_v_gpu[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&pxy_gpu[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);
    
    
    
    /////////////////////////////
    // add pressure pxz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_u_gpu[i] * part_w_gpu[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&pxz_gpu[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);
    
    
    /////////////////////////////
    // add pressure pyy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_v_gpu[i] * part_v_gpu[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&pyy_gpu[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);
    
    
    /////////////////////////////
    // add pressure pyz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_v_gpu[i] * part_w_gpu[i] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&pyz_gpu[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);
    
    
    /////////////////////////////
    // add pressure pzz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part_w_gpu[i] * part_w_gpu[i] * weight[ii][jj][kk];
    for (int ii=0; ii < 2; ii++)
        for (int jj=0; jj < 2; jj++)
            for(int kk=0; kk < 2; kk++)
                atomicAdd(&pzz_gpu[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)], weight[ii][jj][kk] * grd.invVOL);
    
}

void interpP2G_gpu(struct particles* part , struct interpDensSpecies* ids, struct grid* grd, FPpart* part_x_gpu  , 
                    FPpart* part_y_gpu    , FPpart* part_z_gpu     , FPpart* part_u_gpu    , FPpart* part_v_gpu  , 
                    FPpart* part_w_gpu    , FPinterp *part_q_gpu   , FPinterp *Jx_gpu      , FPinterp *Jy_gpu    , 
                    FPinterp *Jz_gpu      , FPinterp *pxx_gpu      , FPinterp *pxy_gpu     , FPinterp *pxz_gpu   , 
                    FPinterp *pyy_gpu     , FPinterp *pyz_gpu      , FPinterp *pzz_gpu     , FPinterp *rhon_gpu  , 
                    FPinterp *rhoc_gpu    , FPfield* XN_flat_gpu   , FPfield* YN_flat_gpu  , FPfield* ZN_flat_gpu, 
                    int grd_size){

    int n_particles = part->nop;

    // Copy CPU arrays to GPU
    cudaMemcpy(rhon_gpu, ids->rhon_flat, grd_size * sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(rhoc_gpu, ids->rhoc_flat, grd_size * sizeof(FPinterp), cudaMemcpyHostToDevice);

    cudaMemcpy(Jx_gpu, ids->Jx_flat, grd_size * sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(Jy_gpu, ids->Jy_flat, grd_size * sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(Jz_gpu, ids->Jz_flat, grd_size * sizeof(FPinterp), cudaMemcpyHostToDevice);

    cudaMemcpy(pxx_gpu, ids->pxx_flat, grd_size * sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(pxy_gpu, ids->pxy_flat, grd_size * sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(pxz_gpu, ids->pxz_flat, grd_size * sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(pyy_gpu, ids->pyy_flat, grd_size * sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(pyz_gpu, ids->pyz_flat, grd_size * sizeof(FPinterp), cudaMemcpyHostToDevice);
    cudaMemcpy(pzz_gpu, ids->pzz_flat, grd_size * sizeof(FPinterp), cudaMemcpyHostToDevice);

    interpP2G_kernel<<<(n_particles+TPB-1)/TPB, TPB>>>(part_x_gpu  , part_y_gpu  , part_z_gpu  ,
                                                       part_u_gpu  , part_v_gpu  , part_w_gpu  , 
                                                       part_q_gpu  , Jx_gpu      , Jy_gpu      , 
                                                       Jz_gpu      , pxx_gpu     , pxy_gpu     , 
                                                       pxz_gpu     , pyy_gpu     , pyz_gpu     , 
                                                       pzz_gpu     , rhon_gpu    , rhoc_gpu    , 
                                                       XN_flat_gpu , YN_flat_gpu , ZN_flat_gpu , 
                                                       part->nop   , *grd);
    cudaDeviceSynchronize();

    // Copy GPU arrays back to CPU
    cudaMemcpy(ids->rhon_flat, rhon_gpu, grd_size * sizeof(FPinterp), cudaMemcpyDeviceToHost)
    cudaMemcpy(ids->rhoc_flat, rhoc_gpu, grd_size * sizeof(FPinterp), cudaMemcpyDeviceToHost)

    cudaMemcpy(ids->Jx_flat, Jx_gpu, grd_size * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jy_flat, Jy_gpu, grd_size * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jz_flat, Jz_gpu, grd_size * sizeof(FPinterp), cudaMemcpyDeviceToHost);

    cudaMemcpy(ids->pxx_flat, pxx_gpu, grd_size * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxy_flat, pxy_gpu, grd_size * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxz_flat, pxz_gpu, grd_size * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyy_flat, pyy_gpu, grd_size * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyz_flat, pyz_gpu, grd_size * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pzz_flat, pzz_gpu, grd_size * sizeof(FPinterp), cudaMemcpyDeviceToHost);
}
