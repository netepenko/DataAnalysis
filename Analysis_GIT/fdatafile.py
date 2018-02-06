"""

Reads a file containing columns of data and creates a dictionary according to header information
using the following format:

1. comments start with ``#`` in the **first column** : ``# My Comment``

2. header information starts with ``#!``, where ``#`` is also in the **first column**. It must precede the data.
   
3. each columns is represented by ``name[dtype, col.nr.]/``,  *dtype* (optional) is the data type and *col.nr*
   (starting at 0) is the column number

4. data are separated by white space **NOT commas!**

5. data types:

      - s : string
      - f : float
      - i : integer

6. blank lines are ignored

 **NOTE**: if data formats are entered they should be specified for all columns

Example data:: 

   #! p_miss[0]/ siglt[2]/ s01[3]/ alt[4]/
   200. 1.35e-4 -1.e-3    0.1
   220. 2.56e-4 -2.e-4    -0.1
   230. 3.47e-6 -3.e-5    1.1

The header can also contain data type information:: 

   #! p_miss[f,0]/ siglt[f,2]/ s01[f,3]/ alt[f,4]/

   200. 1.35e-4 -1.e-3    0.1
   220. 2.56e-4 -2.e-4    -0.1
   230. 3.47e-6 -3.e-5    1.1


Example that opens a file and create a dfile object::

   >>> f0 = dfile('sig_LT.dat')

Loop over content::

   >>> for l in f0:
   >>> ...pm = l['p_miss']
   >>> ...sig_lt = l['siglt']*10000.
   >>> ...print pm, sig_lt, l['alt']
   >>> # end for l

Variable names are also called keys.

---------------------------------------------------------

"""

import sys
import os
import string
import math
import re
import pdb
import numpy as np
# function to create a dictionary from 2 lists: key, values

# create a dictionary from  an array of keys and values

class fdfile:
    """
    Open a file, read and interpret the contents and return a dfile object:

    >>> df = dfile('my_datafile')
    
    """
    def __init__(self, filename, debug = False, new=False):
        self.H=re.compile("^#\!") # pattern for header
        self.C=re.compile("^#")   # pattern for comment
        # pattern for splitting header
        # currently:
        # ^#\! : matches line beginning with #!
        # \[[0-9]+\]\/: matches [0]/ ... [8796]/
        # \[ *\w+ *, *\w *\]\/ : matches [   f  , blabla   ]/
        # 
        self.S=re.compile("^#\!|\[[0-9]+\]\/|\[ *\w *, *\w+ *\]\/") 
        # pattern to find format information in the header 
        self.F=re.compile("\w+ *, *\w+") 
        # pattern to find a character
        self.Fc=re.compile("[a-zA-Z]")
        # supported formats: the output format dictionary key must match the supported formats
        self.fmt_characters = ['f','i','s']
        self.output_format= {'f':'%r ','i':'%d ','s':'%s '}
        # empty dictionary for the formats
        self.formats = {} 
        # flag for printing debugging information
        self.debug = debug
        #
        self.header = None
        self.data = []
        self.adata = []
        self.fdata=[]
        self.keys=[]
        self.new = False
        if new:
            new=True
            self.filename = filename
            self.keys.append('indx')
            self.headindex=0
            self.adata.append('#! ')
            return
        # pdb.set_trace()
        # open file
        self.filename = filename
        # open file
        # self.adata= open(self.filename).readlines()   # read all data
        # if self.debug :
        #    print "datafile --> data read !"
        # remove leading and trailing spaces
        #self.adata = map(string.strip,self.adata)
        #if self.debug :
        #    print "datafile --> spaces removed read !"
        #self.remove_blanks()
        #if self.debug :
        #     print "datafile --> blank lines removed read !"
        if (self.find_header() != 0): # find the header 
            print "cannot interpret data"
            return
        if self.debug :
            print "datafile --> create arrays !"
        self.make_array()  # read data
        if self.debug :
            print "datafile --> arrays created !"
    def __getitem__(self,i):
        # allows 'direct' access to the data
        if type(i) is int:
            # it'sm index
            return self.data[i,:]
        elif type(i) is str:
            # its a lkey return the list of data
            return self.get_data(i)
    def __len__(self):
        return len(self.data)
    def remove_blanks(self): 
        # work through the list in reverse order
        indices = range( len( self.adata ) ) # list of indices
        indices.reverse() # reverse order
        for i in indices: # loop through from the top
            if len(self.adata[i].strip()) == 0:
                del self.adata[i] 
    def find_header(self):
        # find header information : look for a line starting with #!
        # for l in self.adata:
        get_more = True
        io = open(self.filename)
        self.headindex = -1
        while(get_more):
            l_raw = io.readline()
            l = l_raw.strip()
             # add the line to the adata array
            if l_raw == '':
                get_more = False
            if l != '':
                self.adata.append(l)
                self.headindex += 1
            if (self.H.match(l) != None):
                # self.headindex=self.adata.index(l)
                self.header_line = l
                self.header=re.split(self.S,l)[1:-1] # remove first and last element
                self.keys=map(string.strip,self.header)
                self.keys.append('indx') # store the index into the original
                # handle data format
                # pdb.set_trace()
                fmt = re.findall(self.F,l)
                # no format specification, set default format
                if fmt == []:
                    for k in self.keys[:-1] : # skip index
                        self.formats[k] = 'f'
                # use format specifications
                else:
                    for i,f in enumerate(fmt):
                        fc = re.findall(self.Fc,f)[0] # use 1st char. for format
                        self.formats[self.keys[i]] = fc
                # addformat for the index
                self.formats['indx'] ='i'
                # adata array
                # do some checks
                if (len(self.keys)  != len(self.formats)):
                    print self.filename,": problem in data formats !"
                    print self.adata
                    return -1
                return 0
        print self.filename,": no header information !"
        # print 'data dump : '
        # print self.adata
        return -1

    def make_array(self):
        # use loadtxt to get the data
        self.data = np.loadtxt(self.filename)
        
    def make_attributes(self):
        for i,k in enumerate(self.keys[:-1]):
            setattr(self, k, self.get_data(k))

    def scale(self, key, factor):
        """
        multiply all values of key with a factor
        """
        k = self.key.index(key)
        self.data[:,k] *= factor
        #
            
            
    def name(self):
	"""
	print the filename associate with this instance 

	"""
	print "Input file name : ", self.filename
    def show_keys(self):
	"""
	print a list of variable names in the dictionary

	"""
	print self.keys
    def get_keys(self):
	"""

	return a list of keys

	"""
	return self.keys
    def get_header(self):
	"""

	return the header lines

	"""
        return self.adata[self.headindex]

    def get_full_header(self):
        """
        
        return all line up to the header line
        
        """
        return self.adata[:self.headindex+1]

    def show_data(self,keylist): 
	"""

	print all the data corresponding to the key list::

	   >>> df.show_data('key1:key2:key3')

	"""
	akey=keylist.split(":")
	for l in self.data:
	    ll=[]
	    for i in akey:
                format = self.output_format[ self.formats[i] ]
		ll.append( format%( l[i] ) )
	    print ll



    def get_data(self,key):
        """

	return all data for `key` .  


        >>> df.get_data('name')

        """
        i = self.keys.index(key)
        return self.data[:, i]
  
    def get_data_list(self,keylist, sel_func = None, sel_args = None): 

        """

	return all the data corresponding to the key list
	as follows::

	   >>> a = df.get_data_list('key1:key2:key3')
           
        """
        akey=keylist.split(":")
        dd = []
        for i in akey:
            dd.append(self.get_data(i))
        return dd
