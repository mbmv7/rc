Required: g++ compiler, gurobi

Compile and link:

g++ -std=c++11 -O3 -I/home/mirsad/gurobi911/linux64/include -c -o challengeRTE.o src/rc2020.cpp 

g++ -o challengeRTE challengeRTE.o -L/home/mirsad/gurobi911/linux64/lib -lgurobi_g++5.2 -lgurobi91 -lpthread

