#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <ctime>
#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 10000
#define TPB 32
#define EPSILON 0.005

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


__global__ void performStepGPU(Particle* particles, float3* rand_vel_update, float dt=1.0)
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


int main()
{
  // seed for random number
  srand (static_cast <unsigned> (time(0)));

  // Array of particles
  Particle* particles = (Particle*) malloc (sizeof(Particle)*NUM_PARTICLES);
  inicializeParticles(particles);

  // Array of random numbers
  float3* randNumbers = (float3*) malloc (sizeof(float3)*NUM_PARTICLES);
  inicializeRandNumbes(randNumbers);

  // CPU execution
  Particle* particles_cpu = (Particle*) malloc (sizeof(Particle)*NUM_PARTICLES);

  // copy vector to use in the CPU
  std::memcpy(particles_cpu, particles, NUM_PARTICLES*sizeof(Particle));

  printf("Computing particles moves on the CPU…");
  for(int i = 0 ; i < NUM_ITERATIONS ; i++){
    performStepCPU(particles_cpu, randNumbers);
    cudaDeviceSynchronize();
  }
  printf("Done\n");


  // GPU execution
  Particle* particles_gpu = 0;
  float3* randNumbers_gpu = 0;

  Particle* particles_gpu_res = (Particle*) malloc (sizeof(Particle)*NUM_PARTICLES);

  // Allocate device memory
  cudaMalloc(&particles_gpu, NUM_PARTICLES*sizeof(Particle));
  cudaMalloc(&randNumbers_gpu, NUM_PARTICLES*sizeof(float3));


  // Copy array to device
  cudaMemcpy(particles_gpu, particles, NUM_PARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);
  cudaMemcpy(randNumbers_gpu, randNumbers, NUM_PARTICLES*sizeof(float3), cudaMemcpyHostToDevice);

  // Launch kernel to compute the final state of particles
  
  printf("Computing particles moves on the GPU...");
  for(int i = 0 ; i < NUM_ITERATIONS ; i++){
    performStepGPU<<<(NUM_PARTICLES+TPB-1)/TPB, TPB>>>(particles_gpu, randNumbers_gpu);
    //cudaDeviceSynchronize();
  }
  printf("Done\n");

  // Copy back from device to CPU
  cudaMemcpy(particles_gpu_res, particles_gpu, NUM_PARTICLES*sizeof(Particle), cudaMemcpyDeviceToHost);


  // Compare results
  printf("Comparing the output for each implementation…");
  equalFinalState(particles_gpu_res, particles_cpu) ? printf("Correct\n") : printf("Uncorrect\n");

  // Free the memory
  cudaFree(particles_gpu);
  cudaFree(randNumbers_gpu);

  free(particles_cpu);
  free(particles_gpu_res);
  free(randNumbers);
  free(particles);

  return 0;
}