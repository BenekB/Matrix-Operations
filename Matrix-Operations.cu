//	author: Benedykt Bela

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdio.h>

using namespace std;


//	deklaracje funkcji wykonywanych na CPU
void dodawanie(double *A, double *B, double *C, int wymiar);
void odejmowanie(double *A, double *B, double *C, int wymiar);
void mnozenie(double *A, double *B, double *C, int wymiar);
void mnozenie_skalar(double *A, double *C, double skalar, int wymiar);
void transpozycja(double *A, double *C, int wymiar);


//	deklaracje odpowiednio funkcji wykonujacej dzialanie na CPU i GPU
void CPU(double *A, double *B, double *C, double *D, int wymiar);
void GPU(double *A, double *B, double *C, double *D, int *wymiar);



//	definicje funkcji obliczen na macierzach
__global__ void dodawanie_CUDA(double *A, double *B, double *wynik, int *wymiar)
{
	int i = blockIdx.x * wymiar[0];
	int j = blockIdx.y + gridDim.y * threadIdx.x;
	if (j < wymiar[0])
		wynik[j+i] = A[j+i] + B[j+i];
}


__global__ void odejmowanie_CUDA(double *A, double *B, double *wynik, int *wymiar)
{
	int i = blockIdx.x * wymiar[0];
	int j = blockIdx.y + gridDim.y * threadIdx.x;
	if (j < wymiar[0])
		wynik[i+j] = A[i + j] - B[i + j];
}


__global__ void mnozenie_skalar_CUDA(double *A, double *wynik, double skalar , int *wymiar)
{
	int i = blockIdx.x * wymiar[0];
	int j = blockIdx.y + gridDim.y * threadIdx.x;
	if (j < wymiar[0])
		wynik[i+j] = skalar * A[i+j];
}


__global__ void transpozycja_CUDA(double *A, double *wynik, int *wymiar)
{
	int i = blockIdx.x;
	int j = blockIdx.y + gridDim.y * threadIdx.x;
	if (j < wymiar[0])
		wynik[i*wymiar[0] + j] = A[i + j*wymiar[0]];
}


__global__ void mnozenie_CUDA(double *A, double *B, double *wynik, int *wymiar)
{
	int i = blockIdx.x;
	int j = blockIdx.y + gridDim.y * threadIdx.x;
	double help = 0.0;

	//	dodaje kolejne wyniki mnozenia poszczegolnych elementow
	if (j < wymiar[0])
	{
		for (int k = 0; k < wymiar[0]; k++)
			help = help + A[i*wymiar[0] + k] * B[k*wymiar[0] + j];

		//	i zapisuje odpowiedni wynik do macierzy wynikowej
		wynik[i*wymiar[0] + j] = help;
	}
	
}



//	szuka najmniejszego i najwiekszego elementu zadanej macierzy
 void minmax(double *A, double *min, double *max, long long int wymiar)
{
	//	przegladam kazdy element macierzy i zapisuje najwiekszy oraz najmniejszy
	for (int i = 0; i < wymiar; i++)
	{
		 if (A[i] > *max)
			 *max = A[i];
		 else if (A[i] < *min)
			 *min = A[i];
	}
}



//	funkcja realizujaca dzialania na macierzach korzystajac z GPU
void GPU(double *A, double *B, double *C, double *D, int *wymiar)
{
	//	tworze odpowiednie adresy do alokacji zmiennych na GPU
	double *d_A = new double[wymiar[0]*wymiar[0]];
	double *d_B = new double[wymiar[0]*wymiar[0]];
	double *d_C = new double[wymiar[0]*wymiar[0]];
	double *d_D = new double[wymiar[0]*wymiar[0]];
	int *d_wymiar = new int;

	//	alokuje pamiec na GPU i kopiuje potrzebne dane
	cudaMalloc(&d_A, wymiar[0]*wymiar[0]*sizeof(double));
	cudaMalloc(&d_B, wymiar[0]*wymiar[0]*sizeof(double));
	cudaMalloc(&d_C, wymiar[0]*wymiar[0]*sizeof(double));
	cudaMalloc(&d_D, wymiar[0]*wymiar[0]*sizeof(double));
	cudaMalloc(&d_wymiar, sizeof(int));

	cudaMemcpy(d_A, A, wymiar[0] * wymiar[0] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, wymiar[0] * wymiar[0] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_wymiar, wymiar, sizeof(int), cudaMemcpyHostToDevice);

	
	//	ustalam odpowiednie wymiary siatki blokow i watkow w blokach w taki sposob,
	//	zeby w bloku bylo mozliwie najwiecej watkow (najblizej 1024)
	int sqrroot = (int)(wymiar[0] / 1024) + 1;
	dim3 grid_dim(wymiar[0], sqrroot, 1);
	dim3 block_dim(wymiar[0]/sqrroot + 1, 1, 1);

	//	wywolywanie poszczegolnych funkcji wykonywanych na GPU w kolejnosci dajacej 
	//	pozadany wynik zadanego dzialania
	mnozenie_CUDA << <grid_dim, block_dim >> > (d_A, d_B, d_C, d_wymiar);
	transpozycja_CUDA << <grid_dim, block_dim >> > (d_A, d_D, d_wymiar);
	mnozenie_skalar_CUDA << <grid_dim, block_dim >> > (d_D, d_D, 4.0, d_wymiar);
	dodawanie_CUDA << <grid_dim, block_dim >> > (d_D, d_C, d_C, d_wymiar);
	dodawanie_CUDA << <grid_dim, block_dim >> > (d_A, d_C, d_C, d_wymiar);
	odejmowanie_CUDA << <grid_dim, block_dim >> > (d_C, d_B, d_C, d_wymiar);

	//	kopiuje z GPU na CPU tylko macierz wyniku
	cudaMemcpy(C, d_C, wymiar[0] * wymiar[0] * sizeof(double), cudaMemcpyDeviceToHost);
}




int main()
{
	std::cout << "Podaj wymiar macierzy:" << endl;

	int *wymiar = new int;
	cin >> *wymiar;

	std::cout << "Prosze czekac... " << endl;

	int rozmiar = wymiar[0] * wymiar[0];

	//	macierz A, B, wynikowa oraz pomocnicza
	double *macierz_A = new double[rozmiar];
	double *macierz_B = new double[rozmiar];
	double *macierz_X = new double[rozmiar];
	double *macierz_pomocnicza = new double[rozmiar];

	//	uzupelnianie randomowymi liczbami z zakresu od 0 do 10
	for (int i = 0; i < *wymiar; i++)
		for (int j = 0; j < *wymiar; j++)
		{
			macierz_A[i*wymiar[0] + j] = (double)(rand() % 10000) / 7.3;
			macierz_B[i*wymiar[0] + j] = (double)(rand() % 10000) / 8.5;
		}

	std::cout << "Czy wyswietlic macierze A, B oraz wynik?\nJezeli tak wcisnij 't'" << endl;

	char znak;
	cin >> znak;

	std::cout << endl;

	//	jezeli podamy na wejscie znak 't', to program wyswietla macierz A oraz B
	if (znak == 't')
	{
		//	zwykle wypisywanie macierzy
		for (int i = 0; i < wymiar[0]; i++)
		{
			for (int j = 0; j < wymiar[0]; j++)
				std::cout << macierz_A[i*wymiar[0] + j] << "  ";

			std::cout << endl;
		}

		std::cout << endl << endl;

		for (int i = 0; i < wymiar[0]; i++)
		{
			for (int j = 0; j < wymiar[0]; j++)
				std::cout << macierz_B[i*wymiar[0] + j] << "  ";

			std::cout << endl;
		}
	}

	//	zmienne pomocnicze do okreslenia czasu wykonywania programu
	double start, end;

	//	zmienne do sprawdzenia poprawnosci dzialania kodu na CPU i GPU
	double max, max_CPU;
	double min, min_CPU;

	//	ustawiam zmienna start na aktualny czas, wykonuje program na GPU, pobieram czas 
	//	zakonczenia i odejmujac te dwie wartosci wypisuje ile trwalo wykonanie programu
	start = clock();

	GPU(macierz_A, macierz_B, macierz_X, macierz_pomocnicza, wymiar);

	end = clock();

	//	szukam najmniejszej i najwiekszej wartosci w macierzy wynikowej
	min = macierz_X[0];
	max = min;
	minmax(macierz_X, &min, &max, wymiar[0]*wymiar[0]);

	std::cout << endl <<"Czas wykonania na GPU: " << end - start << endl;



	//	jezeli podamy na wejscie znak 't', to program wyswietla macierz wynikowa
	if (znak == 't')
	{
		std::cout << endl << "Wynik GPU: " << endl << endl;

		for (int i = 0; i < *wymiar; i++)
		{
			for (int j = 0; j < *wymiar; j++)
				std::cout << macierz_X[i*wymiar[0] + j] << "  ";

			std::cout << endl;
		}
	}




	//	analogicznie jak wyzej ale dla CPU
	start = clock();

	CPU(macierz_A, macierz_B, macierz_X, macierz_pomocnicza, *wymiar);

	end = clock();

	min_CPU = macierz_X[0];
	max_CPU = min_CPU;
	minmax(macierz_X, &min_CPU, &max_CPU, wymiar[0] * wymiar[0]);

	std::cout << endl <<"Czas wykonania na CPU: " << end - start << endl;


	if (znak == 't')
	{
		std::cout << endl << "Wynik CPU: " << endl << endl;

		for (int i = 0; i < *wymiar; i++)
		{
			for (int j = 0; j < *wymiar; j++)
				std::cout << macierz_X[i*wymiar[0] + j] << "  ";

			std::cout << endl;
		}
	}

	std::cout << endl << "Blad CPU:    " << (max_CPU - min_CPU) / max_CPU << endl;
	std::cout << endl << "Blad GPU:    " << (max - min) / max_CPU << endl;
}



//	funkcja do wykonywania obliczenia na CPU
void CPU(double *A, double *B, double *X, double *D, int wymiar)
{
	mnozenie(A, B, X, wymiar);
	transpozycja(A, D, wymiar);
	mnozenie_skalar(D, D, 4.0, wymiar);
	dodawanie(D, X, X, wymiar);
	dodawanie(A, X, X, wymiar);
	odejmowanie(X, B, X, wymiar);
}



void mnozenie(double *A, double *B, double *C, int wymiar)
{
	double x;

	for (int i = 0; i < wymiar; i++)
		for (int j = 0; j < wymiar; j++)
		{
			x = 0;

			for (int k = 0; k < wymiar; k++)
			{
				x = x + A[i*wymiar + k] * B[k*wymiar + j];
			}

			C[i*wymiar + j] = x;
		}
}



void transpozycja(double *A, double *C, int wymiar)
{
	for (int i = 0; i < wymiar; i++)
		for (int j = 0; j < wymiar; j++)
		{
			C[i*wymiar + j] = A[j*wymiar + i];
		}
}



void dodawanie(double *A, double *B, double *C, int wymiar)
{
	for (int i = 0; i < wymiar; i++)
		for (int j = 0; j < wymiar; j++)
		{
			C[i*wymiar + j] = A[i*wymiar + j] + B[i*wymiar + j];
		}
}



void odejmowanie(double *A, double *B, double *C, int wymiar)
{
	for (int i = 0; i < wymiar; i++)
		for (int j = 0; j < wymiar; j++)
		{
			C[i*wymiar + j] = A[i*wymiar + j] - B[i*wymiar + j];
		}
}



void mnozenie_skalar(double *A, double *C, double skalar, int wymiar)
{
	for (int i = 0; i < wymiar; i++)
		for (int j = 0; j < wymiar; j++)
		{
			C[i*wymiar + j] = skalar * A[i*wymiar + j];
		}
}



