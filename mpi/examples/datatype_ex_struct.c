#include "mpi.h"
 #include <stdio.h>
  
 /*
 *  * This test checks to see if we can create a simple datatype
 *   * made from many contiguous copies of a single struct. The
 *    * struct is built with monotone decreasing displacements to
 *     * avoid any struct->contig optimizations.
 *      */
  
 int main( int argc, char **argv )
 {

     int psize;
     int blocklens[3] = { 2, 5, 3 };
     MPI_Aint disp[3] = { 0, 5*sizeof(int), 5*sizeof(int)+10*sizeof(double) };
     //MPI_Aint disp[3];
     MPI_Datatype oldtypes[3], newtype;
     oldtypes[0] = MPI_INT;
     oldtypes[1] = MPI_DOUBLE;
     oldtypes[2] = MPI_CHAR;

     MPI_Init( &argc, &argv );

     MPI_Type_struct( 3, blocklens, disp, oldtypes, &newtype );
     MPI_Type_commit( &newtype );
  
     MPI_Pack_size( 1, newtype, MPI_COMM_WORLD, &psize );
  
     //printf("%d, %d, %d, %d\n", sizeof(int), sizeof(float), sizeof(char), sizeof(double));
     printf("pack size = %d\n", psize);

     MPI_Type_free( &newtype );
  
     MPI_Finalize();
     return 0;
 }

