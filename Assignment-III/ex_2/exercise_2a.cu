#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/time.h>

#define NUM_ITERATIONS 10000
#define EPSILON 0.005

int NUM_PARTICLES = 10000;
int TPB = 32;

unsigned long get_time();

unsigned long get_time() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        unsigned long ret = tv.tv_usec;
        ret /= 1000;
        ret += (tv.tv_sec * 1000);
        return ret;
}

struct Particle 
{ 
   float3 position;
   float3 velocity; 
};

float3 randomFloat3() {
  float3 f;
  f.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  f.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  f.z = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  return f;
}

// inicialize the array of Particles with random position and velocity
void inicializeParticles(Particle* particles){
  for(int i = 0; i < NUM_PARTICLES ; i++){
    particles[i].position = randomFloat3();
    particles[i].velocity = randomFloat3();
  }
}

// Inicialize all the necessary random variables at once in a matrix randNumbers
void inicializeRandNumbes(float3* randNumbers){
  for(int i = 0; i < NUM_PARTICLES; i++) {
    randNumbers[i] = randomFloat3();
  }
}


__global__ void performStepGPU(Particle* particles, float3* rand_vel_update, int NUM_PARTICLES, float dt=1.0)
{
  const int p_id = blockIdx.x*blockDim.x + threadIdx.x;

  // only calculate if particle is inside the bounds
  if (p_id < NUM_PARTICLES){ 
    particles[p_id].position.x += dt * particles[p_id].velocity.x;
    particles[p_id].position.y += dt * particles[p_id].velocity.y;
    particles[p_id].position.z += dt * particles[p_id].velocity.z;

    particles[p_id].velocity.x += rand_vel_update[p_id].x;
    particles[p_id].velocity.y += rand_vel_update[p_id].y;
    particles[p_id].velocity.z += rand_vel_update[p_id].z;
  }
}



void performStepCPU(Particle* particles, float3* rand_vel_update, float dt=1.0){
  for (int p_id = 0; p_id < NUM_PARTICLES; p_id++){
        particles[p_id].position.x += dt * particles[p_id].velocity.x;
        particles[p_id].position.y += dt * particles[p_id].velocity.y;
        particles[p_id].position.z += dt * particles[p_id].velocity.z;

        particles[p_id].velocity.x += rand_vel_update[p_id].x;
        particles[p_id].velocity.y += rand_vel_update[p_id].y;
        particles[p_id].velocity.z += rand_vel_update[p_id].z;
  }

}

bool equalFinalState(Particle* p1, Particle* p2){
  for(int i = 0 ; i < NUM_PARTICLES; i++){
      if (std::abs(p1[i].position.x - p2[i].position.x) > EPSILON ||
          std::abs(p1[i].position.y - p2[i].position.y) > EPSILON ||
          std::abs(p1[i].position.z - p2[i].position.z) > EPSILON){
        return false;
    }
  }
  return true;
}


int main(int argc,  char** argv)
{

  NUM_PARTICLES = (argc >= 2) ? atoi(argv[1]) : 2000000;
  TPB = argc >= 3 ? atoi(argv[2]) : 128;

  // seed for random number
  srand (static_cast <unsigned> (time(0)));

  // Array of particles
  Particle* particles = (Particle*) malloc (sizeof(Particle)*NUM_PARTICLES);
  inicializeParticles(particles);

  // Array of random numbers
  float3* randNumbers = (float3*) malloc (sizeof(float3)*NUM_PARTICLES);
  inicializeRandNumbes(randNumbers);


  // CPU execution
  long start_time_cpu = get_time();
  Particle* particles_cpu = (Particle*) malloc (sizeof(Particle)*NUM_PARTICLES);

  // copy vector to use in the CPU
  std::memcpy(particles_cpu, particles, NUM_PARTICLES*sizeof(Particle));

  printf("Computing particles system on the CPU…");
  for(int i = 0 ; i < NUM_ITERATIONS ; i++){
    //performStepCPU(particles_cpu, randNumbers);
  }
  printf("Done\n");
  long end_time_cpu = get_time();

  // GPU execution
  long start_time_gpu = get_time();
  Particle* particles_gpu = 0;
  float3* randNumbers_gpu = 0;

  // Allocate host memory
  Particle* particles_gpu_res = (Particle*) malloc (sizeof(Particle)*NUM_PARTICLES); // paged memory
  // Particle* particles_gpu_res = NULL;
  // cudaMallocHost(&particles_gpu_res, NUM_PARTICLES*sizeof(Particle)); // pinned memory

  // Allocate device memory
  cudaMalloc(&particles_gpu, NUM_PARTICLES*sizeof(Particle));
  cudaMalloc(&randNumbers_gpu, NUM_PARTICLES*sizeof(float3));


  // Copy array to device
  cudaMemcpy(randNumbers_gpu, randNumbers, NUM_PARTICLES*sizeof(float3), cudaMemcpyHostToDevice);
  cudaMemcpy(particles_gpu, particles, NUM_PARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);

  // Launch kernel to compute the final state of particles
  printf("Computing particles system on the GPU...");
  for(int i = 0 ; i < NUM_ITERATIONS ; i++){
    
    // Copy from host to device
    if (i > 0)
      cudaMemcpy(particles_gpu, particles_gpu_res, NUM_PARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);

    performStepGPU<<<(NUM_PARTICLES+TPB-1)/TPB, TPB>>>(particles_gpu, randNumbers_gpu, NUM_PARTICLES);
    cudaDeviceSynchronize();
    
    // Copy back from device to host
    cudaMemcpy(particles_gpu_res, particles_gpu, NUM_PARTICLES*sizeof(Particle), cudaMemcpyDeviceToHost);
  
  }
  printf("Done\n");

  long end_time_gpu = get_time();

  // Compare results
  printf("Comparing the output for each implementation…");
  equalFinalState(particles_gpu_res, particles_cpu) ? printf("Correct\n") : printf("Uncorrect\n");

  printf("-----------------------------------------------\n");
  printf("block size: %d  ;  NUM_PARTICLES: %d\n", TPB, NUM_PARTICLES);
  printf("CPU time: %ld ms\n", end_time_cpu-start_time_cpu);
  printf("GPU time: %ld ms\n", end_time_gpu-start_time_gpu);
  printf("-----------------------------------------------\n");

 // printf("%d %d %ld %ld\n", TPB, NUM_PARTICLES, end_time_cpu - start_time_cpu, end_time_gpu - start_time_gpu);


  // Free the memory
  cudaFree(particles_gpu);
  cudaFree(randNumbers_gpu);
  // cudaFree(particles_gpu_res);

  free(particles_cpu);
  free(particles_gpu_res);
  free(randNumbers);
  free(particles);

  return 0;
}