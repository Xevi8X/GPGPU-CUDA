#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_timer.h>
#include <vector_types.h>

#define REFRESH_DELAY     10 //ms
#define FREQ 0.01f
#define ELECTRON_NO 10000
#define dt 0.01f
#define M_k 20000.0f
#define FRICION 0.3f
#define SOFTENING_FACTOR 1.0f

const unsigned int window_width = 512;
const unsigned int window_height = 512;

GLuint viewGLTexture;
cudaGraphicsResource_t viewCudaResource;

float g_fAnim = 0.0;

StopWatchInterface* timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;

float2 *d_field;
float3* d_pos_charge;
float2* d_vel;


#define MAX(a,b) ((a > b) ? a : b)
#define MIN(a,b) ((a > b) ? b : a)

#define check(ans) { _check((ans), __FILE__, __LINE__); }
inline void _check(cudaError_t code, char* file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

void initGLandCUDA() {
    int argc = 0;
    char** argv = NULL;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("CUDA GL Interop");
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    glewInit();

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &viewGLTexture);
    glBindTexture(GL_TEXTURE_2D, viewGLTexture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, window_width, window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    check(cudaGLSetGLDevice(0));

    check(cudaGraphicsGLRegisterImage(&viewCudaResource, viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}


__global__ void fieldKernel(cudaSurfaceObject_t image, float time, int electron_no,int tile_no,int lastTileSize, float2 *d_field, float3* d_pos_charge) {

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int offset = window_height * x + y;
    float fieldx = 0.0f;
    float fieldy = 0.0f;
    float dx;
    float dy;
    float r2;
    float invr;
    float E;
    float3 pos_charge;


    __shared__ float3 shm_pos_charge[1024];
    
    int index = threadIdx.x * 32 + threadIdx.y;

    for (int j = 0; j < tile_no; j++)
    {
        int k = (j + blockIdx.x) % tile_no;
        int shm_size = k == tile_no - 1? lastTileSize : 1024;
        if(index < shm_size) shm_pos_charge[index] = d_pos_charge[index+ 1024 *k];
        __syncthreads();
        for (int i = 0; i < shm_size; i++)
        {
            pos_charge = shm_pos_charge[i];
            dx = pos_charge.x - x;
            dy = pos_charge.y - y;
            r2 = dx * dx + dy * dy + SOFTENING_FACTOR * SOFTENING_FACTOR;
            //if (r2 <= 3.0) continue;
            invr = rsqrtf(r2);
            E = M_k * -pos_charge.z * invr * invr * invr;
            fieldx += E * dx;
            fieldy += E * dy;
        }
        __syncthreads();
    }

    fieldx += 10000.0f * sinf((float)x * FREQ + time);
    fieldy += 10000.0f * cosf((float)y * FREQ + time);
    //fieldx += 5000.0f;

    d_field[offset].x = fieldx;
    d_field[offset].y = fieldy;

    
    float strength = sqrtf(fieldx * fieldx + fieldy * fieldy);
    uchar4 color = make_uchar4(0, (unsigned char)(0.01 * strength), 0, 127);
    surf2Dwrite(color, image, x * sizeof(color), y, cudaBoundaryModeClamp);
}

__global__ void electronsKernel(cudaSurfaceObject_t image, int electron_no, float2* d_field, float3* d_pos_charge, float2* d_vel) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < electron_no)
    {
        float3 pos_charge = d_pos_charge[i];
        int x = round(pos_charge.x);
        int y = round(pos_charge.y);
        float ax = d_field[window_height * x + y].x * pos_charge.z - FRICION* d_vel[i].x;
        float ay = d_field[window_height * x + y].y * pos_charge.z - FRICION * d_vel[i].y;
        d_vel[i].x += ax * dt;
        d_vel[i].y += ay * dt;

        pos_charge.x += d_vel[i].x * dt;
        pos_charge.y += d_vel[i].y * dt;

        if (pos_charge.x < 0.0)
        {
            pos_charge.x = 0.0;
            d_vel[i].x *= -1.0f;
        }
        if (pos_charge.x > window_width)
        {
            pos_charge.x = window_width;
            d_vel[i].x *= -1.0f;
        }
        if (pos_charge.y < 0.0)
        {
            pos_charge.y = 0.0;
            d_vel[i].y *= -1.0f;
        }
        if (pos_charge.y > window_height)
        {
            pos_charge.y = window_height;
            d_vel[i].y *= -1.0f;
        }
        d_pos_charge[i] = pos_charge;
        uchar4 color = make_uchar4(pos_charge.z > 0.0 ? 255 : 0, 0, pos_charge.z < 0.0 ? 255 : 0, 127);
        surf2Dwrite(color, image, x * sizeof(color), y, cudaBoundaryModeClamp);
    } 
}


void callCUDAKernel(cudaSurfaceObject_t image, float time) {
    dim3 block1(32, 32, 1);
    dim3 grid1(window_width / block1.x, window_height / block1.y, 1);
    int tile_no = (int)ceilf(ELECTRON_NO / 1024.0f);
    fieldKernel << <grid1, block1 >> > (image, time, ELECTRON_NO, tile_no, ELECTRON_NO - (tile_no-1)*1024, d_field, d_pos_charge);
    dim3 block2(1024, 1, 1);
    dim3 grid2(ELECTRON_NO/1024 + 1,1, 1);
    electronsKernel << <grid2, block2 >> > (image, ELECTRON_NO, d_field, d_pos_charge, d_vel);
    check(cudaPeekAtLastError());
    check(cudaDeviceSynchronize());

}

void renderFrame() {
    sdkStartTimer(&timer);

    check(cudaGraphicsMapResources(1, &viewCudaResource));

    cudaArray_t viewCudaArray;
    check(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, viewCudaResource, 0, 0));

    cudaResourceDesc viewCudaArrayResourceDesc;
    memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
    viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
    viewCudaArrayResourceDesc.res.array.array = viewCudaArray;

    cudaSurfaceObject_t viewCudaSurfaceObject;
    check(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));

    callCUDAKernel(viewCudaSurfaceObject,g_fAnim);

    check(cudaDestroySurfaceObject(viewCudaSurfaceObject));

    check(cudaGraphicsUnmapResources(1, &viewCudaResource));

    check(cudaStreamSynchronize(0));

    glBindTexture(GL_TEXTURE_2D, viewGLTexture);
    {
        glBegin(GL_QUADS);
        {
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
            glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
            glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, +1.0f);
        }
        glEnd();
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glFinish();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}

void allocMem()
{
    cudaMalloc(&d_field, window_height * window_width * sizeof(float2));
    cudaMalloc(&d_pos_charge, ELECTRON_NO * sizeof(float3));
    cudaMalloc(&d_vel, ELECTRON_NO * sizeof(float2));
}

void initRandomValue()
{
    float3 pos_charge[ELECTRON_NO];
    float2 vel[ELECTRON_NO];

    for (int i = 0; i < ELECTRON_NO; i++)
    {
        pos_charge[i].x = rand() % window_width;
        pos_charge[i].y = rand() % window_height;
        vel[i].x = 20.0f* rand()/ RAND_MAX - 010.0f ;
        vel[i].y = 20.0f * rand() / RAND_MAX - 10.0f;
        pos_charge[i].z = (rand() % 2) * 2 - 1;
    }

    cudaMemcpy(d_pos_charge, pos_charge, ELECTRON_NO * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, vel, ELECTRON_NO * sizeof(float2), cudaMemcpyHostToDevice);
}

void freeMem()
{
    cudaFree(d_field);
    cudaFree(d_pos_charge);
    cudaFree(d_vel);

}

int main(int argc, char** argv)
{
    initGLandCUDA();
    allocMem();
    initRandomValue();
    sdkCreateTimer(&timer);
    glutDisplayFunc(renderFrame);
    glutMainLoop();
    freeMem();
}