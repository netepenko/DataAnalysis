# DataAnalysis
Data anaylysis procedures controled by parameters stored in a SQLITE data base. The goal of these procedures is to perform a fine analysis of the digitizer files where particle signals are fond and the peak height above a polynomial background is determiend. The measured digitizer data can also be corrected by the fitted backgroup to improve pead finding and fitting in several iteration steps. Using the fittes results 2d histograms are build which can then be used to determine particle rates as a function of time. All results and parameters are recorded in the SQLITE data base. 


## Main Content:

- **calc\_examples** : directory where analysis examplesm aremlocated such as:
- *example\_fitting.py*  : perform the multi-stage fitting analysis to obtain the best signal height determinations
-  *example\_rate\_analysis.py* : determine particle rates and fille 2d time vs signal height histogram
-  *online\_analysis.py* : module to for a quick analysis of a shot
-  *online\_analyze\_new\_shot.py* : short example script to analyze a new shot (includes automaticall adding new database entries)
-  *LR\_deconvolute\_histo.py* : example of performing a Lucy-Richardson deconvolution of a histogram to improve resolution. The point spread function can be fitted either with a modified Voigt function or a sum of gaussian (SOG)
-  *various .db files* : several example SQLITE data base files - **analysis\_modules** : directory where the needed modules are located
- **db\_edit** : contains a PyQt based data base editor. The main script is *edit\_shot\_db.py*. Run this program from within this ndirectory by doing *python edit\_shot\_db.py*.

- to work on your own analysis **DO NOT WORK IN THIS DIRECTORY !** Create a dedicated analysis directory and copy the file *setup\_calc.py* to this directory. Edit it and make sure that the variable **analysis\_modules\_dir** in *setup\_calc.py* points to the correct directory.



This an ongoing development and not yet in its final state. 

## Compilation of *.f90 routines

Normally you can use the Make file as follows:

- $ make clean
- $ make

This should build the two necessary libraries *ffind\_peaks2* (for peak finding) and *lfitm1* (line shape fitting). 
If the Makefile does not work for some reason. The work around is to compile the 2 needed fortran files as follows:

- *ffind\_peaks2.f90*  : f2py \-m ffind\_peaks2 \-c ffind\_peaks2.f90
- *lfitm1.f90* : f2py \-m lfitm1 \-c lfitm1.f90