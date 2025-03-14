#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>

#define N 1000000000

void init_array(int* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand();
    }
}

double getTimeMicroseconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}


void insertSort(size_t numberOfElements, int* data) {
    // check for edge case
    if (numberOfElements <= 1) return;

    for (size_t i = 1; i < numberOfElements; i++) {
        int val = data[i];
        // j is the element which comes before the current one
        int j = i - 1;

        // while j is greater than 0 and the value at j is greater than our value
        while (j >= 0 && data[j] > val) {
            // we move each element to the right of the array.
            data[j + 1] = data[j];
            // and then decrease our j
            j--;
        }

        // after we found the correct place for our value
        // we insert it there.
        data[j + 1] = val;
    }
}

void merge(int* data, int left, int mid, int right, int* temp) {
// i is for the left array, j is for the right array and k is for the temp array
    int i = left, j = mid, k = left;
// while the index for the left array is inside its bounds
// adn the index of the right array is inside its bounds
	while (i < mid && j < right) {
// we want to compare the corresponding elements from each array
// if the value of the right array is bigger we want to copy
// the value of the left array to our temp
		if (data[i] <= data[j]) {
			temp[k] = data[i];
            k++; i++;
        }
// otherwise if the left array has a bigger value, we copy the
// value of the right array into our temp
		else {
			temp[k] = data[j];
            k++; j++;
        }
	}

    // copy the remaining elements from left
    while(i < mid) {
        temp[k] = data[i];
        k++; i++;
    }

    // copy the remaining elements from right
    while(j < right) {
        temp[k] = data[j];
        k++; j++;
    }

    for (i = left; i < right; i++) {
        data[i] = temp[i];
    }
}

void mergeSort(size_t n, int* data) {
    int* temp = malloc(n * sizeof(int));
    if (!temp) {
        perror("Failed to allocate memory!");
        exit(1);
    }

    // width of the subarrays
    for (int width = 1; width < n; width *= 2) {
        // here we have 2 * width, because we are mergin 2 subarrays

        #pragma omp parallel for
        for (int i = 0; i < n; i += 2 * width) {

            int left = i;
            // these conditions are needed so that mid and right don't exceed the
            // array length
            int mid = (i + width < n) ? i + width : n;
            int right = (i + 2 * width < n) ? i + 2 * width : n;

            __builtin_prefetch(&data[right], 1, 3);

            merge(data, left, mid, right, temp);
        }
    }

    free(temp);
}

int main() {
    int* data = (int*)malloc(N * sizeof(int));
    init_array(data, N);

    double start = getTimeMicroseconds();

    mergeSort(N, data);

    double end = getTimeMicroseconds();

	printf("\n\nExecution in microseconds: %f\n\n", end - start);

    free(data);
    return 0;
}
