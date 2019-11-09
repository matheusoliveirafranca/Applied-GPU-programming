# Applied-GPU-programming
Repository for DD2360 course at KTH

## Assigment II
To compile the programs, run:
```sh
nvcc -arch=$sm $exercise_x.cu -o $exercise_x.out
```

And to see the results:
```sh
 srun -n 1 ./$exercise_x.out
```

For exercise_3, Threads Per BLock (TPB) and NUM_PARTICLES can be passed as arguments:
```sh
 srun -n 1 ./$exercise_3.out ($NUM_PARTICLES) ($TPB)
```
