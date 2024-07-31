#------Makefile for fortran 90 modules  ----------

# modules

OBJ = ffind_peaks2.o lfitm1.o

INCLUDE    = ./

F2PY     = f2py

F2PY_F1   = --include-paths $(INCLUDE) --overwrite-signature -m 
F2PY_F2   = -c --fcompiler=gfortran --build-dir ./f2py_make 

#----------------------------------------------------------
# $< is the first prerequisite
# $(basename $<) is the filename without extension

%.o: %.f90
	$(F2PY) $(F2PY_F1) $(basename $<) -h sgn_$(basename $<).pyf $<
	$(F2PY) $(F2PY_F2) sgn_$(basename $<).pyf $<

#----------------------------------------------------------
all: $(OBJ)

# this makes sure that each element in OBJ is compiled individually

$(OBJ): %.o: %.f90   

.PHONY : clean





clean:
	rm -f *.so *.pyf *.o
	rm -rf *.dSYM ./f2py_make


