#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#include "GL/glew.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#include "cuda_runtime.h"
#include "kernel.cuh"

#include "text.h"
#include "particule.h"

#define PRINT_DEBUG 0
#define NB_TREADS_OMP 4

#define numThreads 1024	//CUDA define

static float g_inertia = 0.5f;

static float point_size = 2.0f;

static float oldCamPos[] = { 0.0f, 0.0f, -45.0f };
static float oldCamRot[] = { 0.0f, 0.0f, 0.0f };
static float newCamPos[] = { 0.0f, 0.0f, -45.0f };
static float newCamRot[] = { 0.0f, 0.0f, 0.0f };

static particule_t Particules[NB_PARTICULE];

static bool g_showGrid = true;
static bool g_showAxes = true;
static bool g_showSimu = true;

void ShowParticule();

inline bool CUDA_MALLOC( void ** devPtr, size_t size ) {
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc( devPtr, size );
	if ( cudaStatus != cudaSuccess ) {
		printf( "error: unable to allocate buffer\n");
		return false;
	}
	return true;
}

inline bool CUDA_MEMCPY( void * dst, const void * src, size_t count, enum cudaMemcpyKind kind ) {
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy( dst, src, count, kind );
	if ( cudaStatus != cudaSuccess ) {
		printf( "error: unable to copy buffer\n");
		return false;
	}
	return true;
}
	
void RandomizeFloatArray( int n, float * arr ) {
	for ( int i = 0; i < n; i++ ) {
		arr[i] = (float)rand() / ( (float)RAND_MAX / 2.0f ) - 1.0f;
	}
}


void DrawGridXZ( float ox, float oy, float oz, int w, int h, float sz ) {
	
	int i;

	glLineWidth( 1.0f );

	glBegin( GL_LINES );

	glColor3f( 0.48f, 0.48f, 0.48f );

	for ( i = 0; i <= h; ++i ) {
		glVertex3f( ox, oy, oz + i * sz );
		glVertex3f( ox + w * sz, oy, oz + i * sz );
	}

	for ( i = 0; i <= h; ++i ) {
		glVertex3f( ox + i * sz, oy, oz );
		glVertex3f( ox + i * sz, oy, oz + h * sz );
	}

	glEnd();

}

void ShowAxes() {

	glLineWidth( 2.0f );

	glBegin( GL_LINES );
	
	glColor3f( 1.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 2.0f, 0.0f, 0.0f );

	glColor3f( 0.0f, 1.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 2.0f, 0.0f );

	glColor3f( 0.0f, 0.0f, 1.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 2.0f );
	
	glEnd();

}

int colorGalaxy(int index) {
	int value = (index * (NB_PARTICULE_TOTAL/NB_PARTICULE));
	if ( (value >= 0) && (value < 16384) ) return 1;
	if ( (value >= 16384) && (value < 32768) ) return 0;
	if ( (value >= 32768) && (value < 40960) ) return 1;
	if ( (value >= 40960) && (value < 49152) ) return 0;
	if ( (value >= 49152) && (value < 65536) ) return 1;
	if ( (value >= 65536) && (value < 81920) ) return 0;
	return -1;
}

void ShowParticules() {
	int i;
	glPointSize(point_size);
	glBegin( GL_POINTS ); 
	for( i = 0 ; i < NB_PARTICULE ; i++)
	{	
		if (Particules[i].Galaxy == MILKYWAY)
			glColor3f(5.0f, 3.0f, 0.0f);
		else
			glColor3f(0.0f, 3.0f, 5.0f);
	
		glVertex3f(Particules[i].PosX, Particules[i].PosY, Particules[i].PosZ);
	}
	glEnd();
}

int initParticules() {
	FILE* dubFILE;
	dubFILE = fopen("dubinski.tab","r");
	if (dubFILE == NULL)
	{
		SDL_Log( "error: unable to open dubinski.tab\n" );
		return -1;
	}

	int i, index;
	index = 0;
	for (i = 0 ; i < NB_PARTICULE_TOTAL ; i++)
	{
		fscanf(dubFILE, "%f %f %f %f %f %f %f",
			&Particules[index].Masse,
			&Particules[index].PosX,
			&Particules[index].PosY,
			&Particules[index].PosZ,
			&Particules[index].VelX,
			&Particules[index].VelY,
			&Particules[index].VelZ);
		int mod = NB_PARTICULE_TOTAL / NB_PARTICULE;
		if ( (i%mod) == 0 ) index++;
		
	}
	fclose(dubFILE);
	
	#pragma omp parallel for
	for (i = 0 ; i < NB_PARTICULE ; i++){
		if(colorGalaxy(i)){
			Particules[i].Galaxy = MILKYWAY;
		} else {
			Particules[i].Galaxy = ANDROMEDA;
		}
	}
	
#if PRINT_DEBUG
	for (i = 0 ; i < 10 ; i++)
	{
		SDL_Log("%d --> Masse :%f PosX:%f PoxY:%f PosZ:%f VelX:%f VelY:%f VelZ:%f \n",
				i,
				Particules[i].Masse,
				Particules[i].PosX,
				Particules[i].PosY,
				Particules[i].PosZ,
				Particules[i].VelX,
				Particules[i].VelY,
				Particules[i].VelZ);	
	}
	SDL_Log("puissance 2 : %d\t puissance 3 :%d\n",Pow2(2), Pow3(2));
#endif

	return 0;

}

int main( int argc, char ** argv ) {
	
	SDL_Event event;
	SDL_Window * window;
	SDL_DisplayMode current;

  	int index_loop;
	int width = 1024;
	int height = 640;

	bool done = false;

	float mouseOriginX = 0.0f;
	float mouseOriginY = 0.0f;

	float mouseMoveX = 0.0f;
	float mouseMoveY = 0.0f;

	float mouseDeltaX = 0.0f;
	float mouseDeltaY = 0.0f;

	struct timeval begin, end;
	float fps = 0.0;
	float fps_max = 0.0;
	float fps_min = 1000.0;
	char sfps[40] = "FPS: ";

/*
 * Start USER Code 0
 */
	char sfpsmax[40] = "Max FPS: ";
	char sfpsmin[40] = "Min FPS: ";

	int numBlocks = ( NB_PARTICULE + ( numThreads - 1 ) ) / numThreads;

	omp_set_num_threads( NB_TREADS_OMP );

	cudaError_t cudaStatus;
	particule_t * Particule_deviceSrc = NULL;
	particule_t * Particule_deviceDst = NULL;

	cudaStatus = cudaSetDevice( 0 );
	if ( cudaStatus != cudaSuccess ) {
		SDL_Log( "error: unable to setup cuda device\n");
	}
	SDL_Log("Setup CUDA device : OK");

	CUDA_MALLOC( (void**)&Particule_deviceSrc, NB_PARTICULE * sizeof( particule_t ) );
	CUDA_MALLOC( (void**)&Particule_deviceDst, NB_PARTICULE * sizeof( particule_t ) );
	SDL_Log("Allocation CUDA : OK");

	if (!initParticules()){
		SDL_Log("initParticules : OK");
	}else{
		SDL_Log("initParticules : KO");
		return -1;
	}

	CUDA_MEMCPY( Particule_deviceDst, Particules, NB_PARTICULE * sizeof( particule_t ), cudaMemcpyHostToDevice );
	SDL_Log("Initialisation deviceDst");

	SDL_Log("Init : OK \t Nb Particules : %d", NB_PARTICULE);
/*
 * End USER Code 0
 */

	if ( SDL_Init ( SDL_INIT_EVERYTHING ) < 0 ) {
		printf( "error: unable to init sdl\n" );
		return -1;
	}

	if ( SDL_GetDesktopDisplayMode( 0, &current ) ) {
		printf( "error: unable to get current display mode\n" );
		return -1;
	}

	window = SDL_CreateWindow( "SDL", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_OPENGL );
  
	SDL_GLContext glWindow = SDL_GL_CreateContext( window );

	GLenum status = glewInit();

	if ( status != GLEW_OK ) {
		printf( "error: unable to init glew\n" );
		return -1;
	}

	if ( ! InitTextRes( "./bin/DroidSans.ttf" ) ) {
		printf( "error: unable to init text resources\n" );
		return -1;
	}

	SDL_GL_SetSwapInterval( 1 );

	while ( !done ) {
  		
		int i;

		while ( SDL_PollEvent( &event ) ) {
      
			unsigned int e = event.type;
			
			if ( e == SDL_MOUSEMOTION ) {
				mouseMoveX = event.motion.x;
				mouseMoveY = height - event.motion.y - 1;
			} else if ( e == SDL_KEYDOWN ) {
				if ( event.key.keysym.sym == SDLK_F1 ) {
					g_showGrid = !g_showGrid;
				} else if ( event.key.keysym.sym == SDLK_F2 ) {
					g_showAxes = !g_showAxes;
				} else if (event.key.keysym.sym == SDLK_SPACE){
					g_showSimu = !g_showSimu;
				} else if ( event.key.keysym.sym == SDLK_ESCAPE ) {
 				 	done = true;
				}
			}

			if ( e == SDL_QUIT ) {
				printf( "quit\n" );
				done = true;
			}

		}

		mouseDeltaX = mouseMoveX - mouseOriginX;
		mouseDeltaY = mouseMoveY - mouseOriginY;

		if ( SDL_GetMouseState( 0, 0 ) & SDL_BUTTON_LMASK ) {
			oldCamRot[ 0 ] += -mouseDeltaY / 5.0f;
			oldCamRot[ 1 ] += mouseDeltaX / 5.0f;
		}else if ( SDL_GetMouseState( 0, 0 ) & SDL_BUTTON_RMASK ) {
			oldCamPos[ 2 ] += ( mouseDeltaY / 100.0f ) * 0.5 * fabs( oldCamPos[ 2 ] );
			oldCamPos[ 2 ]  = oldCamPos[ 2 ] > -5.0f ? -5.0f : oldCamPos[ 2 ];
		}

		mouseOriginX = mouseMoveX;
		mouseOriginY = mouseMoveY;

		glViewport( 0, 0, width, height );
		glClearColor( 0.2f, 0.2f, 0.2f, 1.0f );
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
		glEnable( GL_BLEND );
		glBlendEquation( GL_FUNC_ADD );
		glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
		glDisable( GL_TEXTURE_2D );
		glEnable( GL_DEPTH_TEST );
		glMatrixMode( GL_PROJECTION );
		glLoadIdentity();
		gluPerspective( 50.0f, (float)width / (float)height, 0.1f, 100000.0f );
		glMatrixMode( GL_MODELVIEW );
		glLoadIdentity();

		for ( i = 0; i < 3; ++i ) {
			newCamPos[ i ] += ( oldCamPos[ i ] - newCamPos[ i ] ) * g_inertia;
			newCamRot[ i ] += ( oldCamRot[ i ] - newCamRot[ i ] ) * g_inertia;
		}

		glTranslatef( newCamPos[0], newCamPos[1], newCamPos[2] );
		glRotatef( newCamRot[0], 1.0f, 0.0f, 0.0f );
		glRotatef( newCamRot[1], 0.0f, 1.0f, 0.0f );
		
		if ( g_showGrid ) {
			DrawGridXZ( -100.0f, 0.0f, -100.0f, 20, 20, 10.0 );
		}

		if ( g_showAxes ) {
			ShowAxes();
		}

		gettimeofday( &begin, NULL );
/*
 * Start USER Code 1
 */
		// Simulation should be computed here
		ShowParticules();
		if (g_showSimu){
			CUDA_MEMCPY( Particule_deviceSrc, Particules, NB_PARTICULE * sizeof( particule_t ), cudaMemcpyHostToDevice );
		
			cuda_calcul_acceleration(numBlocks, numThreads, NB_PARTICULE, Particule_deviceSrc, Particule_deviceDst);

			CUDA_MEMCPY( Particules, Particule_deviceDst, NB_PARTICULE * sizeof( particule_t ), cudaMemcpyDeviceToHost );

			cudaStatus = cudaDeviceSynchronize();
		
			if ( cudaStatus != cudaSuccess ) {
				SDL_Log( "error: unable to synchronize threads\n");
			}
		}

/*
 * End USER Code 1
 */

		gettimeofday( &end, NULL );

		fps = (float)1.0f / ( ( end.tv_sec - begin.tv_sec ) * 1000000.0f + end.tv_usec - begin.tv_usec) * 1000000.0f;
/*
 * Start USER Code 2
 */		
		if (g_showSimu){
	 		fps_max = (fps > fps_max)? fps : fps_max;
			fps_min = (fps < fps_min)? fps : fps_min;
 		}
/*
 * End USER Code 2
 */
		
		sprintf( sfps, "FPS : %.4f", fps );
		sprintf( sfpsmax, "Max FPS : %.4f", fps_max );
		sprintf( sfpsmin, "Min FPS : %.4f", fps_min );

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D(0, width, 0, height);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		DrawText( 10, height - 20, sfpsmax, TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, height - 40, sfps, TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, height - 60, sfpsmin, TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 70, "'SPACE' : pause simulation", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 50, "'ESC' : quit", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 30, "'F1' : show/hide grid", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 10, "'F2' : show/hide axes", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText (10, 90, " Milky way", TEXT_ALIGN_LEFT, RGBA(200,125,0,255));
		DrawText (10, 110, " Andromeda", TEXT_ALIGN_LEFT, RGBA(0,125,200,255));

		SDL_GL_SwapWindow( window );
		SDL_UpdateWindowSurface( window );
	}
/*
 * Start USER Code 3
 */
	SDL_Log("Fermeture Simulation");

	//Reset du périphérique CUDA
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		SDL_Log( "(EE) Unable to reset device\n" );
	}
/*
 * End USER Code 3
 */
	SDL_GL_DeleteContext( glWindow );
	DestroyTextRes();
	SDL_DestroyWindow( window );
	SDL_Quit();

	return 1;
}

