"""
Combined datafile and parameterfile

 * All lines before the **header line** (starting with ``#!`` ) that begin with ``#\``
   are interpreted as parameter file data 
 * if these are present a parameterfile object is created : *pf.par* (*pf*
   is the name of the **pdatafile** object)
   if nothing was present the value of *pf.par* is None


Example data:: 

   #\ my_var = 5.0   
   #\ my_other_var = 10.0
   #
   # just a regular comment
   #
   # the data start below
   #! p_miss[0]/ siglt[2]/ s01[3]/ alt[4]/
   200. 1.35e-4 -1.e-3    0.1
   220. 2.56e-4 -2.e-4    -0.1
   230. 3.47e-6 -3.e-5    1.1


Opening this file as following

>>> from LT.pdatafile import pdfile
>>> pp = pdfile('my_file.data')

To get the raw (string) values

>>> pp.par.get_data('my_var')
 
To see all variables and their values

>>> pp.par.show_all_data()

To convert the value to e.g. a float or a bool

>>> x = pp.par.get_value('my_var')

To convert to an int:

>>> i = pp.par.get_value('my_var', var_type = int)
>>> j = pp.par.get_value('my_var', var_type = int)

In general **pp.par** is a :class:`~LT.parameterfile.pfile` object attached to a
:class:`~LT.datafile.dfile`.

"""

from fdatafile import fdfile
from parameterfile import pfile
import re
# pdatafile inherits from datafile

class fpdfile(fdfile):
    """
    open a pdata file an scan the contents for for a format that can be 
    interpreted
    
    example::
    
    >>> pp=pdfile('my_datafile')
    
    """
    def __init__(self, filename, debug = False):
        # initializations that are specific for pdatafile
        self.par = None
        self.p = self.par
        self.parameter_data = []
        self.parameter_head_index = []
        # extract the parameterfile part
        self.P=re.compile("^#\\\\") # pattern for line to be interpreted by parameterfile
        # call the dfile constructor of dthe parent class: datafile with the right parameters
        fdfile.__init__(self, filename, debug = debug)
        # now analyze the data to extract parameter information
        for i,l in enumerate(self.adata[:self.headindex]):
	    if (self.P.match(l) == None):
                # not a parameter file line
                continue
            else:
                # append except the #/ control character
                self.parameter_data.append(l[2:])
                self.parameter_head_index.append(i)
        if self.parameter_data != []:
            self.par = pfile('',data = self.parameter_data)
            self.p = self.par # for compatibility with an older version
    def set_parameter(self, name, value):
        """
        set a parameters value in the header so that it is saved when the file
        is saved with full header
        
        name: variable name
        value: the value to be stored this MUST be a string
        """
        try:
            k_index = self.par.keys.index(name)
        except:
            print('problem with ', name, 'does it exist ?')
            return
        # get the index into the header
        i_head = self.parameter_head_index[k_index]
        self.adata[i_head] = '#\\ ' + name + ' = ' + value
        self.par.adata[k_index] = ' ' + name + ' = ' + value
        self.par.make_dictionary()

    def add_parameter(self, name, value):
        """
        add a parameter to the file, it will be saved into a new file
        with write_all(file_o, complete_header = True)
        
        """
        ns = name.strip()
        # add it to parameter section
        self.par.adata.append(ns + ' = ' + value)
        self.par.make_dictionary()
        # add it to header
        self.add_header_comment('\\ ' + ns + ' = ' + value )
        
        
        
        
        
# that's it
        
            
        


