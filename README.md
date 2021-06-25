Required: g++ compiler, gurobi

Compiling and linking: (just make sure to use correct paths for gurobi)

g++ -std=c++11 -O3 -I/home/s34/gurobi911/linux64/include -c -o challengeRTE.o src/rc2020.cpp 

g++ -o challengeRTE challengeRTE.o -L/home/s34/gurobi911/linux64/lib -lgurobi_g++5.2 -lgurobi91 -lpthread


(note: this is how executable uploaded to this repository, challengeRTE, has been built on our machine)



