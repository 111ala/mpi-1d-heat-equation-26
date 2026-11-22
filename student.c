#include "utils_1d.h"

/* Implement the MPI solver here. You may add includes and helper functions.
 * Keep the explicit finite difference scheme and the public signature.
 * MPI is initialized/finalized by the driver. The full result and cumulative
 * error must be available on rank 0 when this function returns.
 */

 /*
 * @brief Solve the heat equation using an explicit finite difference scheme.
 *
 * This functions solves the heat equation iterating over ITER time steps.
 *
 * @param heat_array Array of `double` values representing the temperatures of the points.
 * @param err Pointer to a `double` value representing the error in the simulation.
 * @param num_pts Number of points in the simulation.
 *
 * @return 0 on success, -1 on error.
 */


int heat_eqn_solver_1d(double *heat_array, double *err, size_t num_pts){
  double dx = 1.0/(num_pts+1);
  double dt = 0.5*dx*dx;
  
  int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  size_t points = num_pts / (size_t)size;
  size_t rem = num_pts % (size_t)size;
    
  size_t my_beg = (size_t)rank * points + ((size_t)rank < rem ? (size_t)rank : rem) + 1;
  size_t my_count_sz = points + ((size_t)rank < rem ? 1 : 0);
  size_t my_end = (my_count_sz > 0) ? (my_beg + my_count_sz - 1) : my_beg;

  double *aux = NULL;
  if (my_count_sz > 0) {
    aux = (double *)malloc(my_count_sz * sizeof(double));
    if (aux == NULL) return -1;
  }

  double my_err = 0.0;

  for (size_t i = 0; i < ITER; i++){
    if (my_count_sz > 0) {
    
      if (rank > 0){
          MPI_Sendrecv(&heat_array[my_beg], 1, MPI_DOUBLE, rank - 1, 0, // send part 
                      &heat_array[my_beg - 1], 1, MPI_DOUBLE, rank - 1, 0, // receive part 
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }

      
      if (rank < size -1 && (my_end + 1) <= num_pts){
          MPI_Sendrecv(&heat_array[my_end], 1, MPI_DOUBLE, rank + 1, 0, // send part 
                      &heat_array[my_end + 1], 1, MPI_DOUBLE, rank + 1, 0, // receive part 
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);}
              

      for (size_t j = my_beg; j <= my_end; j++){
        aux[j - my_beg] = heat_array[j] + dt/(dx*dx)*(heat_array[j-1] - 2.0*heat_array[j] + heat_array[j+1]);
        my_err += dx*fabs(aux[j-my_beg] - heat_array[j]);
      }

      for (size_t j = my_beg; j <= my_end; j++){
        heat_array[j] = aux[j- my_beg];
      }
    }   
  }

  
  MPI_Reduce(&my_err, err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


  if (rank == 0) {
      int *recvcounts = (int *)malloc(size * sizeof(int));
      int *displs = (int *)malloc(size * sizeof(int));

      if (!recvcounts || !displs) {
            free(aux); free(recvcounts); free(displs);
            return -1;
        }

      for (int r = 0; r < size; r++) {
          size_t r_beg = (size_t)r * points + (size_t)(r < (int)rem ? r : (int)rem) + 1;
            size_t r_cnt = points + (r < (int)rem ? 1 : 0);
            recvcounts[r] = (int)r_cnt;
            displs[r] = (int)r_beg;
      }

      MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE,
                    heat_array, recvcounts, displs, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        free(recvcounts);
        free(displs);
  }
  else{
      MPI_Gatherv(my_count_sz > 0 ? &heat_array[my_beg] : NULL, (int)my_count_sz, MPI_DOUBLE,
                NULL, NULL, NULL, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
  }

  free(aux);
  return 0;
}
