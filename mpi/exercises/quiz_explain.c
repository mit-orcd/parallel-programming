/*MPI_Allreduce is equivalent of doing MPI_Reduce followed by an MPI_Bcast. 
The MPI_Reduce does two things: 
1. Send all local values from non-zero PEs to the PE 0; 
2. The PE 0 receive all values from non-zero PEs, then calculate the sum. 
The MPI_Bcast also does two things: 
1. Send the sum from PE 0 to all non-zero PEs; 
2. All non-zero PEs received the sum from PE 0. Finally all the PEs have the same value of sum. 
 
If we write down what the exercise code really does using the MPI_Send and MPI_Recv functions, it is as following.
*/ 

if(rank % 2 == 0){  // even processes

    // Following is a code block equivalent to MPI_Reduce
     if(rank==0){  // This block will only be executed by PE 0
       MPI_Recv( local_values from non-zero PEs to PE 0);  // Step 2. Here the local_values are the ranks of non-zero PEs
       evensum+=local_values;   // Step 3 
     }
     else{   // This block will only be executed by PEs 2, 4 and 6
        MPI_Send( local_values from non-zero PEs to PE 0);  // Step 1
     }

    // Following is a code block equivalent to MPI_Bcast
    if(rank==0){ // This block will only be executed by PE 0
        MPI_Send( evensum from PE 0 to non-zero PEs);    // Step 4
    }
    else{   // Step 5. This block will only be executed by PEs 2, 4 and 6
        MPI_Recv( evensum from PE0 to non-zero PEs);  
    }

}else{  // odd processes

   // Following is a code block equivalent to MPI_Reduce
     if(rank==0){    // This block will NEVER be executed
       MPI_Recv( local_values from non-zero PEs to PE 0);  
       oddsum+=local_values; 
     }
     else{   // This block will only be executed by PEs 1, 3, 5 and 7
        MPI_Send( local_values from non-zero PEs to PE 0);  // Step 1
     }

    // Following is a code block equivalent to MPI_Bcast
    if(rank==0){ // This block will NEVER be executed
        MPI_Send( oddnsum from PE 0 to non-zero PEs);  
    }
    else{   // Step 5. This block will only be executed by PEs 1, 3, 5 and 7
        MPI_Recv( evensum from PE0 and assign it to oddsum on non-zero PEs);  
    }

}
 
/*
After analysis of the above code, the following steps are executed in sequence:
 
1. All non-zero PEs send the local values, which are the ranks in this case.  
2. PE 0 receives all of the local values from non-zeron PEs. 
3. PE 0 calculates the sum of all local values (which is 28 in the case) and assign it to the variable named evensum. 
4. PE 0 sends the value of evensum (which is 28).
5. The PEs 2, 4, 6 receive the value of evensum from PE0 and assign it to their local variable evensum. In the same time, PEs 1 , 3, 5, 7 receive the value of evensum from PE0 and assign it to their local variable oddsum. 
 
So finally the variable evensum on all even PEs has the value 28 and the variable oddsum on all odd PEs has the value 28 too. Now you can see why answer C is correct. An addition conclusion is that the variable evensum on all odd PEs has the value 0 and the variable oddsum on all even PEs has the value 0 too.
*/

