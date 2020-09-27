#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<unistd.h>
#include<math.h>
#include<float.h>
#include<errno.h>
#include <ctype.h>
#include <stdbool.h>

int k = 0;
int d = 0;
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
        dist += (a[i] - b[i])*(a[i] - b[i]);
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
    
    d = atoi(argv[2]);
    k = atoi(argv[3]);
    MAX_ITER = atoi(argv[4]);
    
    srand((unsigned int)time(NULL));
    double time = 0.0;
    clock_t start, end;
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
    int size = 0;
    while(fgets(line, 1024, cf)){
        size++;
    }
    
    float *data = (float*)malloc(size*d*sizeof(float));
    int j = 0;
    while(fgets(line, 1024, fp)){
        for(int i = 1; i < d + 1; i++){
            char* tmp = strdup(line);
            data[j*d + (i - 1)] = atof(getfield(tmp, i));
            free(tmp);
        }
        j++;
    }
    fclose(cf);
    fclose(fp);
    
    float *centres = (float*)malloc(k*d*sizeof(float));
    float *sum = (float*)malloc(k*d*sizeof(float));
    int *count = (int*)malloc(k*sizeof(int));
    int *assignments = (int*)malloc(size*sizeof(int));
    
    initialCentres(data, centres, size);
    
    start = clock();
    int iter = 0;
    while(iter < MAX_ITER){
        fill(sum, k*d, 0);
        fillInt(count, k, 0);
        assignPoints(data, centres, sum, count, assignments, size);
        updateCentres(centres, sum, count);
        iter++;
    }
    
    end = clock();
    
    time += (double) (end - start) / CLOCKS_PER_SEC;
    
    printf("Time = %f seconds\n", time);
    printf("Centres:\n");
    print2D(centres, k);
    printCount(count, k);
    //printMeans(assignments, size);
    
    free(data);
    free(centres);
    free(sum);
    free(count);
    free(assignments);
    
    return 0;
} 
