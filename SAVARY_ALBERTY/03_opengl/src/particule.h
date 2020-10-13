#ifndef __PARTICULE_H__
#define __PARTICULE_H__

#define NB_PARTICULE_TOTAL 81920
#define NB_PARTICULE 1024
#define MODULO_PARTICULE NB_PARTICULE_TOTAL/NB_PARTICULE

#define Pow2(_a) (_a*_a)
#define Pow3(_a) (_a*_a*_a)

enum Galaxy_name{
MILKYWAY = 0,
ANDROMEDA
};

typedef struct{
	float Masse;
	float PosX;
	float PosY;
	float PosZ;
	float VelX;
	float VelY;
	float VelZ;
	enum Galaxy_name Galaxy;
}particule_t;

#endif
