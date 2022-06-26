ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
VPATH=./cython_solver:./
EIGEN_LIB_LOCATION?=/usr/local/include
OPENCV_LIB_LOCATION?=/usr/local/include/opencv2

SRC_DIR=./
DEMO_DIR=./
LIB_SOLVER_SRC=LPboxADMMsolver.cpp
CFLAGS?=-O3
CXX = g++
ifeq ($(PREFIX),)
	PREFIX := /usr/local
endif

all: solver test   

release: CFLAGS=-O3
release: solver

install: liblpbox_solver.so LPboxADMMsolver.h
	install -d $(DESTDIR)$(PREFIX)/lib/
	install -m 644 liblpbox_solver.so $(DESTDIR)$(PREFIX)/lib/
	install -d $(DESTDIR)$(PREFIX)/include/
	install -m 644 LPboxADMMsolver.h $(DESTDIR)$(PREFIX)/include/

uninstall:
	rm $(DESTDIR)$(PREFIX)/lib/liblpbox_solver.so 
	rm $(DESTDIR)$(PREFIX)/include/LPboxADMMsolver.h 

test: test.cpp LPboxADMMsolver.h LPboxADMMsolver.cpp
	$(CXX) $(CFLAGS) -o test -I $(EIGEN_LIB_LOCATION) -I $(ROOT_DIR) -I $(SRC_DIR) test.cpp liblpbox_solver.so -lpthread

solver: LPboxADMMsolver.h LPboxADMMsolver.cpp
	$(CXX) $(CFLAGS) -fPIC -shared $(LIB_SOLVER_SRC) -o liblpbox_solver.so

clean:
	rm -f liblpbox_solver.so test 
