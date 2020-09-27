#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<unistd.h>
#include<math.h>
#include<float.h>
#include<errno.h>
#include<mpi.h>

int d = 0;
int k = 0;
int MAX_ITER = 0;

const char* getfield(char* line, int num){
    const char* tok;
    for (tok = strtok(line, ",");
            tok && *tok;
            tok = strtok(NULL, ",\n"))
    {
        if (!--num)
            return tok;
    }
    free((char*)tok);
    return NULL;
}


float randomInRange(float a, float b){
    return (a + 1) + (((float) rand())/(float) RAND_MAX)*(b-(a+1));
}

void fill(float *array, int size, int val){
    for(int i = 0; i < size; i++){
        array[i] = val;
    }
}

void fillInt(int *array, int size, int val){
    for(int i = 0; i < size; i++){
        array[i] = val;
    }
}

void initialCentres(float *data, float *centres, int size){
    float *minVals = (float*)malloc(d*sizeof(float));
    fill(minVals, d, 100000);
    float *maxVals = (float*)malloc(d*sizeof(float));
    fill(maxVals, d, -100000);
    for(int i = 0; i < size; i++){
        for(int j = 0; j < d; j++){
            if(data[i*d + j] < minVals[j]){
                minVals[j] = data[i*d + j];
            }
            if(data[i*d + j] > maxVals[j]){
                maxVals[j] = data[i*d + j];
            }
        }
    }
    for(int ki = 0; ki < k; ki++){
        for(int i = 0; i < d; i++){
            float r = randomInRange(minVals[i], maxVals[i]);
            centres[ki*d + i] = r;
        }
    }
    free(minVals);
    free(maxVals);
}

float distance(float *a, float *b){
    float dist = 0;
    for(int i = 0; i < d; i++){
        float diff = a[i] - b[i];
        dist += diff*diff;
    }
    return dist;
}

void assignPoints(float *data, float *centres, float *sum, int *count, int *assignments, int size){
    for(int i = 0; i < size; i++){
        int id = 0;
        float min = distance(&data[i*d], &centres[0]);
        for(int ki = 1; ki < k; ki++){
            float dist = distance(&data[i*d], &centres[ki*d]);
            if(dist < min){
                min = dist;
                id = ki;
            }
        }
        count[id] += 1;
        assignments[i] = id;
        for(int j = 0; j < d; j++){
            sum[id*d + j] += data[i*d + j];
        }
    }
}

void updateCentres(float *centres, float *sum, int *count){
    for(int ki = 0; ki < k; ki++){
        for(int j = 0; j < d; j++){
            sum[ki*d + j] /= count[ki];
            centres[ki*d + j] = sum[ki*d + j];
        }
    }
}

void print2D(float *data, int size){
    for(int i = 0; i < size; i++){
        for(int j = 0; j < d; j++){
            printf("%f ", data[i*d + j]);
        }
        printf("\n");
    }
}

void print1D(int *data, int size){
    for(int i = 0; i < size; i++){
        printf("%i\n", data[i]);
    }
}
void printMeans(int *data, int size){
    printf("[");
    for(int i = 0; i < size; i++){
        printf("%i", data[i]);
        if(i != size-1){
            printf(", ");
        }
    }
    printf("]\n");
}

void printCount(int *data, int size){
    printf("Count : ");
    for(int i = 0; i < size; i++){
        printf("%i = %i ", i, data[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[]){
    MPI_Init(NULL, NULL);
    double stime = 0.0;
    clock_t start, end;
    int rank, size_MPI;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size_MPI);
    MPI_Barrier(MPI_COMM_WORLD);

    float *g_data, *g_centres, *g_sum, *l_data, *l_centres, *l_sum;
    int size, *g_count, *g_assignments, *l_count, *l_assignments, *disp, *subcount, *disp2, *subcount2;

    if(rank == 0){
        if(argc != 5){
            if(argc < 5){
                printf("Not enough arguments given.\n");
            } else{
                printf("Too many arguments given.\n");
            }
            printf("Please input the following arguments in the given order:\n");
            printf("  -<file_name>.csv\n");
            printf("  -the number of rows to read from csv\n");
            printf("  -the number of clusters to create\n");
            printf("  -the number of iterations to run\n");
            exit(-1);
        } else if(strstr(argv[1], ".csv") == NULL){
            printf("File name entered is not a csv.\n");
            exit(-1);
        }
    }
    
    d = atoi(argv[2]);
    k = atoi(argv[3]);
    MAX_ITER = atoi(argv[4]);
    
    if(rank == 0){
        srand((unsigned int)time(NULL));
        FILE* cf = fopen(argv[1], "r");
        FILE* fp = fopen(argv[1], "r");
        if(!fp){
            perror("fopen");
            exit(-1);
        }
        if(!cf){
            perror("fopen");
            exit(-1);
        }
        char line[1024];
        size = 0;
        while(fgets(line, 1024, cf)){
            size++;
        }
        
        g_data = (float*)malloc(size*d*sizeof(float));
        
        int j = 0;
        while(fgets(line, 1024, fp)){
            for(int i = 1; i < d + 1; i++){
                char* tmp = strdup(line);
                g_data[j*d + (i - 1)] = atof(getfield(tmp, i));
                free(tmp);
            }
            j++;
        }
        
        g_centres = (float*)malloc(k*d*sizeof(float));
        l_centres = (float*)malloc(k*d*sizeof(float));
        initialCentres(g_data, g_centres, size);
        for(int i = 0; i < k*d; i++){
            l_centres[i] = g_centres[i];
        }
        g_sum = (float*)malloc(k*d*sizeof(float));
        g_count = (int*)malloc(k*sizeof(int));
        g_assignments = (int*)malloc(size*sizeof(int));
        fclose(cf);
        fclose(fp);
    }
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int data_size_split = size / size_MPI;
    if(rank != size_MPI - 1){
        l_data = (float*)malloc(data_size_split*d*sizeof(float));
        l_assignments = (int*)malloc(data_size_split*sizeof(int));
    }else if(rank == size_MPI - 1){
        data_size_split += (size % size_MPI);
        l_data = (float*)malloc(data_size_split*d*sizeof(float));
        l_assignments = (int*)malloc(data_size_split*sizeof(int));
    }
    
    
    l_sum = (float*)malloc(k*d*sizeof(float));
    l_count = (int*)malloc(k*sizeof(int));
    
    if(rank != 0){
        l_centres = (float*)malloc(k*d*sizeof(float));
    }

    disp = (int*)malloc(size_MPI*sizeof(int));
    subcount = (int*)malloc(size_MPI*sizeof(int));
    disp2 = (int*)malloc(size_MPI*sizeof(int));
    subcount2 = (int*)malloc(size_MPI*sizeof(int));

    for(int i = 0; i < size_MPI; i++){
        disp[i] = i*d*data_size_split;
        subcount[i] = d*data_size_split;
        disp2[i] = i*data_size_split;
        subcount2[i] = data_size_split;
    }
	char name[MPI_MAX_PROCESSOR_NAME];
	int len;
	MPI_Get_processor_name( name, &len );
	printf( "Node number %d/%d is %s\n", rank, size_MPI, name);
    MPI_Scatterv(g_data, subcount, disp, MPI_FLOAT, l_data, d*data_size_split, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if(rank == 0){
        start = clock();
    }
    int iter = 0;
    while(iter < MAX_ITER){
        MPI_Bcast(l_centres, k*d, MPI_FLOAT, 0, MPI_COMM_WORLD);
        fill(l_sum, k*d, 0);
        fillInt(l_count, k, 0);
        assignPoints(l_data, l_centres, l_sum, l_count, l_assignments, data_size_split);

        MPI_Reduce(l_sum, g_sum, k*d, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(l_count, g_count, k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank == 0){
            updateCentres(g_centres, g_sum, g_count);
            for(int i = 0; i < k*d; i++){
                l_centres[i] = g_centres[i];
            }
        }
        iter++;
        MPI_Bcast(&iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(rank == 0){
        end = clock();
    }
    
    if(rank == 0){
        printf("\n");
        stime += (double) (end - start) / CLOCKS_PER_SEC;
        printf("Time = %f seconds\n", stime);
        printf("Centres:\n");
        print2D(g_centres, k);
        printCount(g_count, k);
        free(g_data);
        free(g_centres);
        free(g_sum);
        free(g_count);
        free(g_assignments);
    }
    MPI_Finalize();
    free(l_data);
    free(l_centres);
    free(l_sum);
    free(l_count);
    free(l_assignments);
    free(disp);
    free(subcount);
    free(disp2);
    free(subcount2);
} 
