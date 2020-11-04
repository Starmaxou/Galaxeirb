# Galaxeirb
Projet MI205
Simulation de colision de galaxy

## Sans optimisation
- Lecture du fichier dubinski.tab : OK
- Affichage des galaxies : OK
- Calcule des positions : OK

## Optimisation des calculs
- Utilisation de macro pour les puissances.
- Pas de répétition pour les calculs des constantes.

## Parallèlisation par OpenMP
```
#pragma omp parallel for
	for (i = 0 ; i < NB_PARTICULE ; i++){
		if (colorGalaxy(i)){
			Particules[i].Galaxy = MILKYWAY;
		} else {
			Particules[i].Galaxy = ANDROMEDA;
		}
	}
```
```
#pragma omp parallel for
		for ( index_loop = 0 ;  index_loop < NB_PARTICULE ; index_loop++){
			particule_calcul(index_loop);
		}
```
