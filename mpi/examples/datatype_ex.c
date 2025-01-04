#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[])
 {
     int rank, size, i, n=18;
     int blocklen[3] = { 2, 5, 3 };
     int disp[3] = { 0, 5, 15 };
     MPI_Datatype type1, type2, type3;
     //int buffer[n], buffer1[n], buffer2[n], buffer3[n];
     MPI_Status status;

     MPI_Init(&argc, &argv);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
     if (size < 2)
     {
         printf("Please run with 2 processes.\n");
         MPI_Finalize();
         return 1;
     }
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

     MPI_Type_contiguous(n, MPI_INT, &type1);
     MPI_Type_commit(&type1);
     MPI_Type_vector(3, 4, 7, MPI_INT, &type2);
     MPI_Type_commit(&type2);
     MPI_Type_indexed(3, blocklen, disp, MPI_INT, &type3);
     MPI_Type_commit(&type3);

     if (rank == 0)
     {
         int buffer[n];
         for (i=0; i<n; i++) buffer[i] = i+1;
         MPI_Send(buffer, 1, type1, 1, 101, MPI_COMM_WORLD);
         MPI_Send(buffer, 1, type2, 1, 102, MPI_COMM_WORLD);
         MPI_Send(buffer, 1, type3, 1, 103, MPI_COMM_WORLD);
     } else if (rank == 1) {
         int buffer1[n], buffer2[n], buffer3[n];
         for (i=0; i<n; i++){ 
               buffer1[i] = 0;
               buffer2[i] = 0;
               buffer3[i] = 0;
         }
         MPI_Recv(buffer1, 1, type1, 0, 101, MPI_COMM_WORLD, &status);
         MPI_Recv(buffer2, 1, type2, 0, 102, MPI_COMM_WORLD, &status);
         MPI_Recv(buffer3, 1, type3, 0, 103, MPI_COMM_WORLD, &status);
         for (i=0; i<n; i++) printf("buffer_contiguous[%d] = %d\n", i, buffer1[i]);
         printf("----------------------------------- \n");
         for (i=0; i<n; i++) printf("buffer_vector[%d] = %d\n", i, buffer2[i]);
         printf("----------------------------------- \n");
         for (i=0; i<n; i++) printf("buffer_indexed[%d] = %d\n", i, buffer3[i]);
         fflush(stdout);
     }

     MPI_Finalize();
     return 0;
 }
