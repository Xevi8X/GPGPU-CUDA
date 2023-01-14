f/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 /*
	 This example demonstrates how to use the Cuda OpenGL bindings to
	 dynamically modify a vertex buffer using a Cuda kernel.

	 The steps are:
	 1. Create an empty vertex buffer object (VBO)
	 2. Register the VBO with Cuda
	 3. Map the VBO for writing from Cuda
	 4. Run Cuda kernel to modify the vertex positions
	 5. Unmap the VBO
	 6. Render the results using OpenGL

	 Host code
 */

 // includes, system
#define _USE_MATH_DEFINES
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <thrust/version.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include <vector_types.h>


typedef struct
{
	int* cellNo;
	int* fishNo;

} CellFishPair;


#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

#define MAXINITVEL 3.0f
#define SOFTEN_FACTOR 0.001f
#define TURNABLE_FACTOR 0.1f
#define TURNABLE_EDGE 0.95f
#define MINSPEED 0.5f
#define MAXSPEED 5.0f

long NUMOFFISHES;
float FISHSIZE;
float3 color;

float3* d_pos;
float3* d_vel;
int* d_cellNo;
int* d_fishNo;
CellFishPair* d_fishesGrid;

float3* cpuPos;
float3* cpuVel;

int* d_offsets;

float angle = 120.0f;
float cosAngle = cosf(angle * 4.0f * atanf(1) / 180.0f);
float maxDistance = 0.1f;
float separationCoff = .001f;
float aligmentCoff = 6.0f;
float cohesionCoff = 0.03f;

bool isRunning = true;
bool GPUrendering = true;


////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 1024;
const unsigned int window_height = 1024;


// vbo variables
GLuint vbo;
struct cudaGraphicsResource* cuda_vbo_resource;
void* d_vbo_buffer = NULL;

float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 30.0, rotate_y = 30.0;
float translate_z = -3.0;

StopWatchInterface* timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;


#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// 
__host__ __device__ float3 cross(float3 a, float3 b);
__host__ __device__ float dot(float3 a, float3 b);
__host__ __device__ float3 operator*(const float& a, const float3& b);
__host__ __device__ float3 normalize(float3 v);
__host__ __device__ float3 operator+(const float3& a, const float3& b);
__host__ __device__ float3 operator-(const float3& a, const float3& b);
__host__ __device__ float length(float3 v);
// declaration, forward
bool runTest();
void cleanup();

int main(int argc, char** argv);

// GL functionality
bool initGL();
void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res,
	unsigned int vbo_res_flags);
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res);
// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);
// Cuda functionality
void runCuda(struct cudaGraphicsResource** vbo_resource);

void allocMem();
void initRandomValue();
void freeMem();
void handleArgs(int argc, char** argv);

const char* sSDKsample = "simpleGL (VBO)";


__device__ int calcCellNo(float3 pos, int gridSize)
{
	float step = 2.0f / gridSize;
	int cellNo = ((int)((pos.x + 1.0f) / step)) +
		gridSize * ((int)((pos.y + 1.0f) / step)) +
		gridSize * gridSize * ((int)((pos.z + 1.0f) / step));
	int gridSize3 = gridSize * gridSize * gridSize;

	return  cellNo < 0 ? 0 : (cellNo >= gridSize3 ? gridSize : cellNo);
}

__global__ void setFishesGrid(float3* pos, int numOfFishes, CellFishPair* fishesGrid, int gridSize)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numOfFishes)
	{
		float3 myPos = pos[i];
		int j = calcCellNo(myPos, gridSize);
		fishesGrid->cellNo[i] = j;
		fishesGrid->fishNo[i] = i;
	}
}

__host__ __device__ bool isVisable(float3 pos, float3 vel, float3 anotherFish, float cosAngle, float maxDistance)
{
	float3 diff = anotherFish - pos;
	if (length(diff) > maxDistance) return false;
	float3 vec = normalize(diff);
	float compatibilty = dot(vec, normalize(vel));
	return compatibilty > cosAngle;
}

__global__ void interactBetweenWithGrid(float3* pos, float3* vel, int numOfFishes, float time, float dt,
	float separationCoff, float aligmentCoff, float cohesionCoff, float cosAngle, float maxDistance,
	CellFishPair* fishesGrid, int gridSize, int* starts, int* ends,int* offsets)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numOfFishes)
	{
		float3 myPos = pos[i];
		float3 myVel = vel[i];
		int myCellNo = calcCellNo(myPos, gridSize);
		int gridSize3 = gridSize * gridSize * gridSize;

		float3 sumOfPos = make_float3(0.0f, 0.0f, 0.0f);
		float3 sumOfVel = make_float3(0.0f, 0.0f, 0.0f);
		float3 accSeparation = make_float3(0.0f, 0.0f, 0.0f);
		int counter = 0;
		
		for (int offsetIndex = 0; offsetIndex < 27; offsetIndex++)
		{
			int cellNo = myCellNo + offsets[offsetIndex];
			if (cellNo < 0 || cellNo >= gridSize3) continue;
			for (int pairIndex = starts[cellNo]; pairIndex < ends[cellNo]; pairIndex++)
			{
				int j = fishesGrid->fishNo[pairIndex];
				if (i == j) continue;
				float3 anotherPos = pos[j];
				if (!isVisable(myPos, myVel, anotherPos, cosAngle, maxDistance)) continue;
				float3 anotherVel = vel[j];
				sumOfPos = sumOfPos + anotherPos;
				sumOfVel = sumOfVel + anotherVel;
				float3 toUs = myPos - anotherPos;
				float lengthToUs = length(toUs);
				accSeparation = accSeparation + (1.0f / (lengthToUs + SOFTEN_FACTOR) * (lengthToUs + SOFTEN_FACTOR)) * normalize(toUs);
				counter++;
			}
		}

		if (counter == 0) return;

		accSeparation = separationCoff * accSeparation;
		float3 accAlignment = aligmentCoff * ((1.0f / counter) * sumOfVel - myVel);
		float3 accCohesion = cohesionCoff * ((1.0f / counter) * sumOfPos - myPos);

		__syncthreads();
		vel[i] = myVel + dt * (accSeparation + accAlignment + accCohesion);
	}
}

void interactBetweenCPU(float3* pos, float3* vel, int numOfFishes, float time, float dt,
	float separationCoff, float aligmentCoff, float cohesionCoff, float cosAngle, float maxDistance)
{
	float3* newVel = (float3*)malloc(numOfFishes * sizeof(float3));
	for (int i = 0; i < numOfFishes; i++)
	{
		float3 myPos = pos[i];
		float3 myVel = vel[i];

		float3 sumOfPos = make_float3(0.0f, 0.0f, 0.0f);
		float3 sumOfVel = make_float3(0.0f, 0.0f, 0.0f);
		float3 accSeparation = make_float3(0.0f, 0.0f, 0.0f);
		int counter = 0;
		for (int j = 0; j < numOfFishes; j++)
		{
			if (i == j) continue;
			float3 anotherPos = pos[j];
			if (!isVisable(myPos, myVel, anotherPos, cosAngle, maxDistance)) continue;
			float3 anotherVel = vel[j];
			sumOfPos = sumOfPos + anotherPos;
			sumOfVel = sumOfVel + anotherVel;
			float3 toUs = myPos - anotherPos;
			float lengthToUs = length(toUs);
			accSeparation = accSeparation + (1.0f / (lengthToUs + SOFTEN_FACTOR) * (lengthToUs + SOFTEN_FACTOR)) * normalize(toUs);
			counter++;
		}
		if (counter == 0) return;

		accSeparation = separationCoff * accSeparation;
		float3 accAlignment = aligmentCoff * ((1.0f / counter) * sumOfVel - myVel);
		float3 accCohesion = cohesionCoff * ((1.0f / counter) * sumOfPos - myPos);

		newVel[i] = myVel + dt * (accSeparation + accAlignment + accCohesion);
	}
	free(vel);
	vel = newVel;
}

__global__ void interactBetween(float3* pos, float3* vel, int numOfFishes, float time, float dt,
	float separationCoff, float aligmentCoff, float cohesionCoff, float cosAngle, float maxDistance)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numOfFishes)
	{
		float3 myPos = pos[i];
		float3 myVel = vel[i];

		float3 sumOfPos = make_float3(0.0f, 0.0f, 0.0f);
		float3 sumOfVel = make_float3(0.0f, 0.0f, 0.0f);
		float3 accSeparation = make_float3(0.0f, 0.0f, 0.0f);
		int counter = 0;
		for (int j = 0; j < numOfFishes; j++)
		{
			if (i == j) continue;
			float3 anotherPos = pos[j];
			if (!isVisable(myPos, myVel, anotherPos, cosAngle, maxDistance)) continue;
			float3 anotherVel = vel[j];
			sumOfPos = sumOfPos + anotherPos;
			sumOfVel = sumOfVel + anotherVel;
			float3 toUs = myPos - anotherPos;
			float lengthToUs = length(toUs);
			accSeparation = accSeparation + (1.0f / (lengthToUs + SOFTEN_FACTOR) * (lengthToUs + SOFTEN_FACTOR)) * normalize(toUs);
			counter++;
		}
		if (counter == 0) return;

		accSeparation = separationCoff * accSeparation;
		float3 accAlignment = aligmentCoff * ((1.0f / counter) * sumOfVel - myVel);
		float3 accCohesion = cohesionCoff * ((1.0f / counter) * sumOfPos - myPos);

		__syncthreads();
		vel[i] = myVel + dt * (accSeparation + accAlignment + accCohesion);
	}
}

__device__ void insertFish(float3* address, float3 pos, float3 vel, float fishSize)
{
	float3 vel_normal = normalize(vel);
	float3 x;
	if (vel_normal.x == 0.0f && vel_normal.y == 0.0f && vel_normal.z == 1.0f) x = make_float3(1.0, 0.0, 0.0);
	else x = cross(vel_normal, make_float3(0.0f, 0.0f, 1.0f));

	float3 y = cross(x, vel_normal);
	float3 p1 = pos + fishSize * 0.25f * y - fishSize * vel_normal;
	float3 p2 = pos + fishSize * 0.21f * x - fishSize * 0.125f * y - fishSize * vel_normal;
	float3 p3 = pos - fishSize * 0.21f * x - fishSize * 0.125f * y - fishSize * vel_normal;

	address[0] = pos;
	address[1] = p1;
	address[2] = p2;
	address[3] = pos;
	address[4] = p2;
	address[5] = p3;
	address[6] = pos;
	address[7] = p3;
	address[8] = p1;

}

__host__ __device__ float3 minMaxVel(float3 old_vel)
{
	float l = length(old_vel);
	if (l < MINSPEED) return (MINSPEED / l) * old_vel;
	if (l > MAXSPEED) return (MAXSPEED / l) * old_vel;
	return old_vel;
}

void softMoveFishesCPU(float3* pos, float3* vel, int numOfFishes, float time, float dt)
{
	for (int i = 0; i < numOfFishes; i++)
	{
		float3 myPos = pos[i];
		float3 myVel = vel[i];

		myPos = myPos + dt * myVel;
		myVel = minMaxVel(myVel);
		if (myPos.x > TURNABLE_EDGE)
		{
			if (myPos.x > 1.0) myPos.x = 1.0;
			myVel.x -= TURNABLE_FACTOR;
		}
		if (myPos.y > TURNABLE_EDGE)
		{
			if (myPos.y > 1.0) myPos.y = 1.0;
			myVel.y -= TURNABLE_FACTOR;
		}
		if (myPos.z > TURNABLE_EDGE)
		{
			if (myPos.z > 1.0) myPos.y = 1.0;
			myVel.z -= TURNABLE_FACTOR;
		}
		if (myPos.x < -TURNABLE_EDGE)
		{
			if (myPos.x < -1.0) myPos.x = -1.0;
			myVel.x += TURNABLE_FACTOR;
		}
		if (myPos.y < -TURNABLE_EDGE)
		{
			if (myPos.y < -1.0) myPos.y = -1.0;
			myVel.y += TURNABLE_FACTOR;;
		}
		if (myPos.z < -TURNABLE_EDGE)
		{
			if (myPos.z < -1.0) myPos.z = -1.0;
			myVel.z += TURNABLE_FACTOR;
		}
		pos[i] = myPos;
		vel[i] = myVel;
	}
}

__global__ void softMoveFishes(float3* pos, float3* vel, int numOfFishes, float time, float dt)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numOfFishes)
	{
		float3 myPos = pos[i];
		float3 myVel = vel[i];

		myPos = myPos + dt * myVel;
		myVel = minMaxVel(myVel);
		if (myPos.x > TURNABLE_EDGE)
		{
			if (myPos.x > 1.0) myPos.x = 1.0;
			myVel.x -= TURNABLE_FACTOR;
		}
		if (myPos.y > TURNABLE_EDGE)
		{
			if (myPos.y > 1.0) myPos.y = 1.0;
			myVel.y -= TURNABLE_FACTOR;
		}
		if (myPos.z > TURNABLE_EDGE)
		{
			if (myPos.z > 1.0) myPos.y = 1.0;
			myVel.z -= TURNABLE_FACTOR;
		}
		if (myPos.x < -TURNABLE_EDGE)
		{
			if (myPos.x < -1.0) myPos.x = -1.0;
			myVel.x += TURNABLE_FACTOR;
		}
		if (myPos.y < -TURNABLE_EDGE)
		{
			if (myPos.y < -1.0) myPos.y = -1.0;
			myVel.y += TURNABLE_FACTOR;;
		}
		if (myPos.z < -TURNABLE_EDGE)
		{
			if (myPos.z < -1.0) myPos.z = -1.0;
			myVel.z += TURNABLE_FACTOR;
		}
		pos[i] = myPos;
		vel[i] = myVel;
	}
}

__global__ void moveFishes(float3* pos, float3* vel, int numOfFishes, float time, float dt)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numOfFishes)
	{
		float3 myPos = pos[i];
		float3 myVel = vel[i];
		myPos = myPos + dt * myVel;

		if (myPos.x > 1.0)
		{
			myPos.x = 1.0;
			myVel.x = myVel.x > 0.0 ? -myVel.x : myVel.x;
		}
		if (myPos.y > 1.0)
		{
			myPos.y = 1.0;
			myVel.y = myVel.y > 0.0 ? -myVel.y : myVel.y;
		}
		if (myPos.z > 1.0)
		{
			myPos.z = 1.0;
			myVel.z = myVel.z > 0.0 ? -myVel.z : myVel.z;
		}
		if (myPos.x < -1.0)
		{
			myPos.x = -1.0;
			myVel.x = myVel.x < 0.0 ? -myVel.x : myVel.x;
		}
		if (myPos.y < -1.0)
		{
			myPos.y = -1.0;
			myVel.y = myVel.y < 0.0 ? -myVel.y : myVel.y;
		}
		if (myPos.z < -1.0)
		{
			myPos.z = -1.0;
			myVel.z = myVel.z < 0.0 ? -myVel.z : myVel.z;
		}
		pos[i] = myPos;
		vel[i] = myVel;
	}
}

__global__ void drawFishes(float3* fishes, float3* pos, float3* vel, int numOfFishes, float fishSize)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numOfFishes)
	{
		float3 myPos = pos[i];
		float3 myVel = vel[i];
		insertFish(&(fishes[9 * i]), myPos, myVel, fishSize);
	}
}

void thrust_magic(int gridSize, int numOfFishes, int** d_starts, int** d_ends)
{
	int gridSize3 = gridSize * gridSize * gridSize;
	thrust::device_ptr<int> t_cellNo(d_cellNo);
	thrust::device_ptr<int> t_fishNo(d_fishNo);
	sort_by_key(thrust::device, t_cellNo, t_cellNo + numOfFishes, t_fishNo);
	thrust::device_vector<int> t_ones(numOfFishes, 1);
	thrust::device_vector<int> distinct_keys(gridSize3);
	thrust::device_vector<int> distinct_key_freq(gridSize3);
	thrust::device_vector<int> tmp(gridSize3);
	thrust::device_vector<int> starts(gridSize3, -1);
	thrust::device_vector<int> ends(gridSize3, -1);
	thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;
	new_end = thrust::reduce_by_key(t_cellNo, t_cellNo + numOfFishes, t_ones.begin(), distinct_keys.begin(), distinct_key_freq.begin());
	int distinctNo = new_end.first - distinct_keys.begin();
	//Start
	thrust::exclusive_scan(distinct_key_freq.begin(), distinct_key_freq.begin() + distinctNo, tmp.begin());
	thrust::scatter(tmp.begin(), tmp.begin() + distinctNo, distinct_keys.begin(), starts.begin());
	//End
	thrust::inclusive_scan(distinct_key_freq.begin(), distinct_key_freq.begin() + distinctNo, tmp.begin());
	thrust::scatter(tmp.begin(), tmp.begin() + distinctNo, distinct_keys.begin(), ends.begin());

	*d_starts = thrust::raw_pointer_cast(starts.data());
	*d_ends = thrust::raw_pointer_cast(ends.data());

	/*thrust::copy(distinct_keys.begin(), distinct_keys.end(), std::ostream_iterator<int>(std::cout, ","));
	std::cout << std::endl;
	thrust::copy(distinct_key_freq.begin(), distinct_key_freq.end(), std::ostream_iterator<int>(std::cout, ","));
	std::cout << std::endl;
	thrust::copy(starts.begin(), starts.end(), std::ostream_iterator<int>(std::cout, ","));
	std::cout << std::endl;
	thrust::copy(ends.begin(), ends.end(), std::ostream_iterator<int>(std::cout, ","));*/
}

int* fillOffsets(int n)
{
	int* offsets = (int*)malloc(27 * sizeof(int));
	
	int mn2 = -n * n;
	offsets[0]  = mn2 - n - 1;
	offsets[1]  = mn2 - n;
	offsets[2]  = mn2 - n + 1;
	offsets[3]  = mn2 - 1;
	offsets[4]  = mn2;
	offsets[5]  = mn2 + 1;
	offsets[6]  = mn2 + n - 1;
	offsets[7]  = mn2 + n;
	offsets[8]  = mn2 + n + 1;
	offsets[9]  = -n - 1;
	offsets[10] = -n;
	offsets[11] = -n + 1;
	offsets[12] = -1;
	offsets[13] = 0;
	for (int i = 0; i < 13; i++) offsets[26 - i] = -offsets[i];
	return offsets;
}

void launch_kernel(float3* fishes, float3* pos, float3* vel, CellFishPair* fishesGrid, int numOfFishes, float time, float3* cpuPos, float3* cpuVel)
{
	// execute the kernel
	dim3 block(1024, 1, 1);
	dim3 grid(numOfFishes / block.x + 1, 1, 1);
	if (isRunning)
	{
		if (GPUrendering)
		{
			// GRID SPLITING
			int gridSize = (int)(2.0f / maxDistance);
			setFishesGrid << <grid, block >> > (pos, numOfFishes, fishesGrid, gridSize);
			int* d_starts = nullptr, * d_ends = nullptr;
			thrust_magic(gridSize, numOfFishes, &d_starts, &d_ends);
			int* offsets = fillOffsets(gridSize);
			cudaMemcpy(d_offsets, offsets, 27 * sizeof(int), cudaMemcpyHostToDevice);
			free(offsets);
			interactBetweenWithGrid << <grid, block >> > (pos, vel, numOfFishes, time, REFRESH_DELAY / 1000.0,
				separationCoff, aligmentCoff, cohesionCoff, cosAngle, maxDistance,
				fishesGrid, gridSize, d_starts, d_ends,d_offsets);
			softMoveFishes << <grid, block >> > (pos, vel, numOfFishes, time, REFRESH_DELAY / 1000.0);
		}

		else
		{
			interactBetweenCPU(cpuPos, cpuVel, numOfFishes, time, REFRESH_DELAY / 1000.0,
				separationCoff, aligmentCoff, cohesionCoff, cosAngle, maxDistance);
			softMoveFishesCPU(cpuPos, cpuVel, numOfFishes, time, REFRESH_DELAY / 1000.0);
			cudaMemcpy(d_pos, cpuPos, NUMOFFISHES * sizeof(float3), cudaMemcpyHostToDevice);
			cudaMemcpy(d_vel, cpuVel, NUMOFFISHES * sizeof(float3), cudaMemcpyHostToDevice);
		}
	}
	drawFishes << <grid, block >> > (fishes, pos, vel, numOfFishes, FISHSIZE);
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	std::cout << std::endl;

#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif

	printf("%s starting...\n", sSDKsample);
	printf("\n");
	handleArgs(argc, argv);

	allocMem();
	printf("allocMem done!\n");
	initRandomValue();
	runTest();
	freeMem();
	printf("freeMem done!\n");

	printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
	exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void handleArgs(int argc, char* argv[])
{
	unsigned int rgb;
	int r, g, b;
	NUMOFFISHES = 50000;
	FISHSIZE = 0.02f;
	color = make_float3(255.0f / 255.0f, 153.0f / 255.0f, 19.0f / 255.0f);

	switch (argc)
	{
	case 4:
		sscanf_s(argv[3], "%f", &FISHSIZE);
	case 3:
		sscanf_s(argv[2], "%X", &rgb);
		r = rgb / (256 * 256);
		rgb %= (256 * 256);
		g = rgb / 256;
		b = rgb % 256;
		color = color = make_float3(r / 255.0f, g / 255.0f, b / 255.0f);
	case 2:
		sscanf_s(argv[1], "%ld", &NUMOFFISHES);
		break;
	default:
		fprintf(stderr, "USAGE: Wrong args: using default value instead");
	case 1:
		break;
	}
}

void computeFPS()
{
	float newAvgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
	if (newAvgFPS / avgFPS < 0.9 || newAvgFPS / avgFPS > 1.1) avgFPS = newAvgFPS;
	sdkResetTimer(&timer);
	char fps[256];
	sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
	glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL()

{
	int argc = 1;
	glutInit(&argc, nullptr);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda GL Interop (VBO)");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	// initialize necessary OpenGL extensions
	if (!isGLVersionSupported(2, 0))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);

	SDK_CHECK_ERROR_GL();

	return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest()
{
	sdkCreateTimer(&timer);

	if (false == initGL())
	{
		return false;
	}

	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
#if defined (__APPLE__) || defined(MACOSX)
	atexit(cleanup);
#else
	glutCloseFunc(cleanup);
#endif

	// create VBO
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	// run the cuda part
	runCuda(&cuda_vbo_resource);

	// start rendering mainloop
	glutMainLoop();

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource** vbo_resource)
{
	// map OpenGL buffer object for writing from CUDA
	float3* dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes,
		*vbo_resource));
	//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	// execute the kernel
	//    dim3 block(8, 8, 1);
	//    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	//    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

	launch_kernel(dptr, d_pos, d_vel, d_fishesGrid, NUMOFFISHES, g_fAnim, cpuPos, cpuVel);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res,
	unsigned int vbo_res_flags)
{
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = NUMOFFISHES * 9 * sizeof(float3);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res)
{

	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

void drawBox()
{
	glColor4f(0.0f, 0.0f, 1.0f, 0.01f);
	/* draws the sides of a unit cube (0,0,0)-(1,1,1) */
	glBegin(GL_POLYGON);/* f1: front */
	//glNormal3f(-1.0f, 0.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glVertex3f(-1.0f, 1.0f, -1.0f);
	glVertex3f(1.0f, 1.0f, -1.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);
	glEnd();
	glBegin(GL_POLYGON);/* f2: bottom */
	//glNormal3f(0.0f, 0.0f, -1.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);
	glVertex3f(-1.0f, 1.0f, 1.0f);
	glVertex3f(1.0f, 1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);
	glEnd();
	glBegin(GL_POLYGON);/* f3:back */
	//glNormal3f(1.0f, 0.0f, 0.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glEnd();
	glBegin(GL_POLYGON);/* f4: top */
	//glNormal3f(0.0f, 0.0f, 1.0f);
	glVertex3f(1.0f, 1.0f, -1.0f);
	glVertex3f(1.0f, 1.0f, 1.0f);
	glVertex3f(-1.0f, 1.0f, 1.0f);
	glVertex3f(-1.0f, 1.0f, -1.0f);
	glEnd();
	glBegin(GL_POLYGON);/* f5: left */
	//glNormal3f(0.0f, 1.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glVertex3f(-1.0f, 1.0f, -1.0f);
	glVertex3f(-1.0f, 1.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);
	glEnd();
	glBegin(GL_POLYGON);/* f6: right */
	//glNormal3f(0.0f, -1.0f, 0.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);
	glVertex3f(1.0f, 1.0f, -1.0f);
	glVertex3f(1.0f, 1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);
	glEnd();

	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glLineWidth(3.0f);
	glBegin(GL_LINE_LOOP);
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);
	glVertex3f(1.0f, 1.0f, -1.0f);
	glVertex3f(-1.0f, 1.0f, -1.0f);
	glEnd();
	glBegin(GL_LINE_LOOP);
	glVertex3f(-1.0f, 1.0f, 1.0f);
	glVertex3f(1.0f, 1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(-1.0f, 1.0f, 1.0f);
	glVertex3f(-1.0f, 1.0f, -1.0f);
	glVertex3f(1.0f, 1.0f, 1.0f);
	glVertex3f(1.0f, 1.0f, -1.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glEnd();
}


////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	sdkStartTimer(&timer);

	// run CUDA kernel to generate vertex positions
	runCuda(&cuda_vbo_resource);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);


	drawBox();
	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(color.x, color.y, color.z);
	glDrawArrays(GL_TRIANGLES, 0, 27 * NUMOFFISHES);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
	g_fAnim += 0.01f;
	sdkStopTimer(&timer);
	if(isRunning) computeFPS();
}

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}

void cleanup()
{
	sdkDeleteTimer(&timer);

	if (vbo)
	{
		deleteVBO(&vbo, cuda_vbo_resource);
	}
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	const float multipler = 1.25f;

	switch (key)
	{
	case (27):
#if defined(__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif
	case 'q':
		separationCoff *= multipler;
		printf("Separation %f\n", separationCoff);
		break;
	case 'a':
		separationCoff /= multipler;
		printf("Separation %f\n", separationCoff);
		break;

	case 'w':
		aligmentCoff *= multipler;
		printf("Alignment %f\n", aligmentCoff);
		break;
	case 's':
		aligmentCoff /= multipler;
		printf("Alignment %f\n", aligmentCoff);
		break;

	case 'e':
		cohesionCoff *= multipler;
		printf("Cohesion %f\n", cohesionCoff);
		break;
	case 'd':
		cohesionCoff /= multipler;
		printf("Cohesion %f\n", cohesionCoff);
		break;

	case 'r':
		maxDistance *= multipler;
		printf("Visablility distance %f\n", maxDistance);
		break;
	case 'f':
		maxDistance /= multipler;
		printf("Visablility distance %f\n", maxDistance);
		break;

	case 't':
		angle += 10.0f;
		cosAngle = cosf(angle * 4.0f * atanf(1) / 180.0f);
		printf("Visablility angle %f degree\n", angle);
		break;
	case 'g':
		angle -= 10.0f;
		cosAngle = cosf(angle * 4.0f * atanf(1) / 180.0f);
		printf("Visablility angle %f degree\n", angle);
		break;
	case 'i':
		GPUrendering = !GPUrendering;
		printf(GPUrendering ? "GPU rendering\n" : "CPU rendering\n");
		if (GPUrendering)
		{
			cudaMemcpy(d_pos, cpuPos, NUMOFFISHES * sizeof(float3), cudaMemcpyHostToDevice);
			cudaMemcpy(d_vel, cpuVel, NUMOFFISHES * sizeof(float3), cudaMemcpyHostToDevice);
		}
		else
		{
			cudaMemcpy(cpuPos, d_pos,  NUMOFFISHES * sizeof(float3), cudaMemcpyDeviceToHost);
			cudaMemcpy(cpuVel, d_vel,  NUMOFFISHES * sizeof(float3), cudaMemcpyDeviceToHost);
		}
		break;
	case 'o':
		initRandomValue();
		break;
	case 'p':
		isRunning = !isRunning;
		break;

	case 'h':
		printf("HELP:\n"
			"(PARAMETER, UP KEY, DOWN KEY)\n"
			"sepatation     q        a\n"
			"alignment      w        s\n"
			"cohesion       e        d\n"
			"distance       r        f\n"
			"angle          t        g\n"
			"GPU/CPU             i\n"
			"reset               o\n"
			"pause               p\n"
			"help                h\n"
		);
		break;

	}

}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void allocMem()
{
	cudaMalloc(&d_pos, NUMOFFISHES * sizeof(float3));
	cudaMalloc(&d_vel, NUMOFFISHES * sizeof(float3));
	cudaMalloc(&d_fishesGrid, sizeof(CellFishPair));

	CellFishPair* tmp_struct = (CellFishPair*)malloc(sizeof(CellFishPair));
	cudaMalloc(&d_cellNo, NUMOFFISHES * sizeof(int));
	tmp_struct->cellNo = d_cellNo;
	cudaMalloc(&d_fishNo, NUMOFFISHES * sizeof(int));
	tmp_struct->fishNo = d_fishNo;
	cudaMemcpy(d_fishesGrid, tmp_struct, sizeof(CellFishPair), cudaMemcpyHostToDevice);
	free(tmp_struct);

	cpuPos = (float3*)malloc(NUMOFFISHES * sizeof(float3));
	cpuVel = (float3*)malloc(NUMOFFISHES * sizeof(float3));

	cudaMalloc(&d_offsets, 27 * sizeof(int));
}

void initRandomValue()
{
	for (int i = 0; i < NUMOFFISHES; i++)
	{
		cpuPos[i].x = (2.0f * rand()) / RAND_MAX - 1.0f;
		cpuPos[i].y = (2.0f * rand()) / RAND_MAX - 1.0f;
		cpuPos[i].z = (2.0f * rand()) / RAND_MAX - 1.0f;

		cpuVel[i].x = (2.0f * MAXINITVEL * rand()) / RAND_MAX - MAXINITVEL;
		cpuVel[i].y = (2.0f * MAXINITVEL * rand()) / RAND_MAX - MAXINITVEL;
		cpuVel[i].z = (2.0f * MAXINITVEL * rand()) / RAND_MAX - MAXINITVEL;
	}

	cudaMemcpy(d_pos, cpuPos, NUMOFFISHES * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel, cpuVel, NUMOFFISHES * sizeof(float3), cudaMemcpyHostToDevice);
}

void freeMem()
{
	cudaFree(d_pos);
	cudaFree(d_vel);

	cudaFree(d_fishNo);
	cudaFree(d_cellNo);
	cudaFree(d_fishesGrid);

	free(cpuPos);
	free(cpuVel);

	free(d_offsets);
}

//MY CG

__host__ __device__ float3 cross(float3 a, float3 b)
{
	return make_float3(a.y * b.z - a.z * b.y, b.x * a.z - b.z * a.x, a.x * b.y - a.y * b.x);
}

__host__ __device__ float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float3 operator*(const float& a, const float3& b) {

	return make_float3(a * b.x, a * b.y, a * b.z);

}

__host__ __device__ float3 normalize(float3 v)
{
	return rsqrt(dot(v, v)) * v;
}

__host__ __device__ float3 operator+(const float3& a, const float3& b) {

	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}

__host__ __device__ float3 operator-(const float3& a, const float3& b) {

	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);

}

__host__ __device__ float length(float3 v)
{
	return sqrt(dot(v, v));
}