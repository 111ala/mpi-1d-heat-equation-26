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
  const double dx = 1.0/(num_pts+1);
  const double dt = 0.5*dx*dx;
  const double alpha = dt / (dx * dx);
  
  int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  size_t points = num_pts / (size_t)size;
  size_t rem = num_pts % (size_t)size;
    
  size_t my_beg = (size_t)rank * points + ((size_t)rank < rem ? (size_t)rank : rem) + 1;
  size_t my_count_sz = points + ((size_t)rank < rem ? 1 : 0);

  int left_rank = (rank > 0 && my_count_sz > 0) ? (rank - 1) : MPI_PROC_NULL;
  int right_rank = (rank < size - 1 && (my_beg + my_count_sz) <= num_pts && my_count_sz > 0) ? (rank + 1): MPI_PROC_NULL;

  double *curr = NULL;
  double *next = NULL;

  if (my_count_sz > 0) {
    curr = (double *)malloc((my_count_sz +2) * sizeof(double));
    next = (double *)malloc((my_count_sz +2) * sizeof(double));
    if (curr == NULL || next == NULL) {
        free(curr);
        free(next);
        return -1;
    }

    for (size_t j = 0; j < my_count_sz + 2; j++) {
            curr[j] = heat_array[my_beg - 1 + j];
        }
        next[0] = curr[0];
        next[my_count_sz + 1] = curr[my_count_sz + 1];
  }

  double my_err = 0.0;

  for (size_t i = 0; i < ITER; i++){
    if (my_count_sz > 0) {
      MPI_Request reqs[4];
      int nreq = 0;

      if (left_rank != MPI_PROC_NULL) {
          MPI_Irecv(&curr[0], 1, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD, &reqs[nreq++]);
          MPI_Isend(&curr[1], 1, MPI_DOUBLE, left_rank, 1, MPI_COMM_WORLD, &reqs[nreq++]);
      }
      if (right_rank != MPI_PROC_NULL) {
          MPI_Irecv(&curr[my_count_sz + 1], 1, MPI_DOUBLE, right_rank, 1, MPI_COMM_WORLD, &reqs[nreq++]);
          MPI_Isend(&curr[my_count_sz],     1, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD, &reqs[nreq++]);
      }

      double step_err = 0.0;
      for (size_t j = 2; j <= my_count_sz - 1; j++) {
        double val = curr[j] + alpha * (curr[j - 1] - 2.0 * curr[j] + curr[j + 1]);
                next[j] = val;
                step_err += fabs(val - curr[j]);
      }

      if (nreq > 0) {
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
      }

      
      double val1 = curr[1] + alpha * (curr[0] - 2.0 * curr[1] + curr[2]);
      next[1] = val1;
      step_err += fabs(val1 - curr[1]);
      

      if (my_count_sz > 1) {
        double valN = curr[my_count_sz] + alpha * (curr[my_count_sz - 1] - 2.0 * curr[my_count_sz] + curr[my_count_sz + 1]);
        next[my_count_sz] = valN;
        step_err += fabs(valN - curr[my_count_sz]);
      }

      my_err += step_err;

      double *tmp = curr;
      curr = next;
      next = tmp;
    }   
  }

  my_err *= dx;
  MPI_Reduce(&my_err, err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


  if (rank == 0) {
      int *recvcounts = (int *)malloc((size_t)size * sizeof(int));
      int *displs = (int *)malloc((size_t)size * sizeof(int));

      if (!recvcounts || !displs) {
            free(curr); free(next); free(recvcounts); free(displs);
            return -1;
        }

      for (int r = 0; r < size; r++) {
          size_t r_beg = (size_t)r * points + (size_t)(r < (int)rem ? r : (int)rem) + 1;
            size_t r_cnt = points + (r < (int)rem ? 1 : 0);
            recvcounts[r] = (int)r_cnt;
            displs[r] = (int)r_beg;
      }

      MPI_Gatherv(my_count_sz > 0 ? &curr[1] : NULL, (int)my_count_sz, MPI_DOUBLE,
                    heat_array, recvcounts, displs, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        free(recvcounts);
        free(displs);
  }
  else{
      MPI_Gatherv(my_count_sz > 0 ? &curr[1] : NULL, (int)my_count_sz, MPI_DOUBLE,
                NULL, NULL, NULL, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
  }

  free(curr);
  free(next);
  return 0;
}
