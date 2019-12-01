# Applied-GPU-programming
Repository for DD2360 course at KTH. 

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

## Assigment III
### Compile
To compile the programs, run:
```sh
nvcc -O3 -arch=$sm $ex.cu -o $ex.out
```

To compile exercise 3, you should add two other arguments:
```sh
nvcc -O3 -arch=$sm exercise_3.cu -o exercise_3.out -lcurand -lcublas
```
### Results
For exercise_1, you can see the result on an image:
```sh
 srun -n 1 ./hw3_ex1.out $image_path.bmp
```

for exercise_3:
```sh
 srun -n 1 ./exercise_3.out -s $size_matrix -v
```


