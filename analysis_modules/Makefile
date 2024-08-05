#------Makefile for fortran 90 modules  ----------
#
# when ffindpeaks is modified the signature files needs to be adjusted and checked
#


# modules

OBJ = ffind_peaks2.o lfitm1.o

# signature files (customized if necessary)

PYF = ffind_peaks2.pyf lfitm1.pyf

INCLUDE    = ./

F2PY     = f2py

F2PY_F1   = --include-paths $(INCLUDE) --overwrite-signature -m 
F2PY_F2   = -c --fcompiler=gfortran --build-dir ./f2py_make 

#----------------------------------------------------------
# $< is the first prerequisite
# $(basename $<) is the filename without extension

%.o: %.f90
	$(F2PY) $(F2PY_F1) $(basename $<) -h sgn_$(basename $<)_auto.pyf $<
	# an automatic signature file is created which can be edited if needed and safed in the standard form
	# use automatic file
	$(F2PY) sgn_$(basename $<)_auto.pyf $(F2PY_F2)  $<
	# use default file (should not be needed)
	# $(F2PY) sgn_$(basename $<).pyf $(F2PY_F2)  $<
#----------------------------------------------------------
all: $(OBJ)

# this makes sure that each element in OBJ is compiled individually

$(OBJ): %.o: %.f90   

.PHONY : clean





clean:
	rm -f *.so *.o
	rm -rf *.dSYM ./f2py_make

