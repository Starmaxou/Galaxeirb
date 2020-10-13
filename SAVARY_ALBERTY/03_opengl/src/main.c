#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

#include "GL/glew.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#include "text.h"
#include "particule.h"

#define PRINT_DEBUG 0

static float g_inertia = 0.5f;

static float g_t = 0.1f;

static float oldCamPos[] = { 0.0f, 0.0f, -45.0f };
static float oldCamRot[] = { 0.0f, 0.0f, 0.0f };
static float newCamPos[] = { 0.0f, 0.0f, -45.0f };
static float newCamRot[] = { 0.0f, 0.0f, 0.0f };

static particule_t Particules[NB_PARTICULE];

static bool g_showGrid = true;
static bool g_showAxes = true;

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

int colorGalaxy(int index)
{
	int value = (index * (NB_PARTICULE_TOTAL/NB_PARTICULE - 1));
	if ( (value >= 0) && (value < 16384) ) return 1;
	if ( (value >= 16384) && (value < 32768) ) return 0;
	if ( (value >= 32768) && (value < 40960) ) return 1;
	if ( (value >= 40960) && (value < 49152) ) return 0;
	if ( (value >= 49152) && (value < 65536) ) return 1;
	if ( (value >= 65536) && (value < 81920) ) return 0;
	return -1;
}

void ShowParticules()
{
	
	glPointSize(2.0f);
	glBegin( GL_POINTS );

	int i, colorFlag; 
	float Masse;
	for( i = 0 ; i < NB_PARTICULE ; i++)
	{	
		colorFlag = colorGalaxy(i);

		if (colorFlag== 1){
			glColor3f(5.0f, 3.0f, 0.0f);
			Particules[i].Galaxy = MILKYWAY;
		}else if (colorFlag == 0){
			glColor3f(0.0f, 3.0f, 5.0f);
			Particules[i].Galaxy = ANDROMEDA;
		}else{
			glColor3f(10.0f, 0.0f, 0.0f);
		}
		glVertex3f(Particules[i].PosX, Particules[i].PosY, Particules[i].PosZ);
		
	}
	glEnd();
}

void DrawParticule(float PosX, float PosY, float PosZ, enum Galaxy_name Galaxy)
{
	glPointSize(2.0f);
	glBegin( GL_POINTS );

	if (Galaxy == MILKYWAY)
		glColor3f(5.0f, 3.0f, 0.0f);
	else if (Galaxy == ANDROMEDA)
		glColor3f(0.0f, 3.0f, 5.0f);
	
	glVertex3f(PosX, PosY, PosZ);

	glEnd();
}

void acc_part_calcul(int index)
{
	float sumX, sumY, sumZ ,dX, dY, dZ, distance;
	int i;
	int M = 10;
	int E = 1;

	sumX = 0;
	sumY = 0;
	sumZ = 0;
	for (i = 0 ; i < NB_PARTICULE ; i++){
		if (i != index){
			dX = Particules[i].PosX - Particules[index].PosX;
			dY = Particules[i].PosY - Particules[index].PosY;
			dZ = Particules[i].PosZ - Particules[index].PosZ;

			distance = sqrt( Pow2(dX) + Pow2(dY) + Pow2(dZ));
			sumX = sumX + dX * M * E * (1/Pow3(distance)) * Particules[i].Masse;
			sumY = sumY + dY * M * E * (1/Pow3(distance)) * Particules[i].Masse;
			sumZ = sumZ + dZ * M * E * (1/Pow3(distance)) * Particules[i].Masse;
		}
	}

	Particules[index].VelX = Particules[index].VelX + sumX;
	Particules[index].VelY = Particules[index].VelY + sumY;
	Particules[index].VelZ = Particules[index].VelZ + sumZ;
}

void update_position(int index)
{
	Particules[index].PosX = Particules[index].PosX + Particules[index].VelX * g_t;
	Particules[index].PosY = Particules[index].PosY + Particules[index].VelY * g_t;
	Particules[index].PosZ = Particules[index].PosZ + Particules[index].VelZ * g_t;
}

int initParticules()
{
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
	
		if ( (i%80) == 0 )
		{
			ShowParticules(Particules[index].PosX, Particules[index].PosY,Particules[index].PosZ);
			index++;
		}
		
	}
	fclose(dubFILE);
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

	int index_loop;
	SDL_Event event;
	SDL_Window * window;
	SDL_DisplayMode current;
  	
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
	char sfps[40] = "FPS: ";

/*
 * Start USER Code 0
 */
	if (!initParticules()){
		SDL_Log("initParticules : OK");
	}else{
		SDL_Log("initParticules : KO");
		return -1;
	}
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

	window = SDL_CreateWindow( "SDL", 	SDL_WINDOWPOS_CENTERED, 
										SDL_WINDOWPOS_CENTERED, 
										width, height, 
										SDL_WINDOW_OPENGL );
  
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

		// Simulation should be computed here
		ShowParticules();
		for ( index_loop = 0 ;  index_loop < NB_PARTICULE ; index_loop++){
			acc_part_calcul(index_loop);
			update_position(index_loop);
		}
		
		gettimeofday( &end, NULL );

		fps = (float)1.0f / ( ( end.tv_sec - begin.tv_sec ) * 1000000.0f + end.tv_usec - begin.tv_usec) * 1000000.0f;
		sprintf( sfps, "FPS : %.4f", fps );

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D(0, width, 0, height);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		DrawText( 10, height - 20, sfps, TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 30, "'F1' : show/hide grid", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 10, "'F2' : show/hide axes", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText (10, 60, " Milky way", TEXT_ALIGN_LEFT, RGBA(200,125,0,255));
		DrawText (10, 80, " Andromeda", TEXT_ALIGN_LEFT, RGBA(0,125,200,255));

		SDL_GL_SwapWindow( window );
		SDL_UpdateWindowSurface( window );
	}

	SDL_GL_DeleteContext( glWindow );
	DestroyTextRes();
	SDL_DestroyWindow( window );
	SDL_Quit();

	return 1;
}

