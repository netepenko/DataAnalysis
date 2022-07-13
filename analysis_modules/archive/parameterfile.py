"""
Module that reads a parameter file and returns a dictionary with the
variable names and values (all as strings)

The file format is as follows:

 * comments start with ``#`` in the **first column** : ``# My Comment``

 * each line can have one or more name, value pairs, separated by semi colons

   ``name1 = value1; name2 = value2``

In the following example assume that ``value1`` is a float::

    >>> pf = parameterfile.pfile('my_file')

As an alternative the data can also be provided in a list of strings.
This is useful if parameters are stored in a datafile or are provided
from another source.

Example: the data are stored in the string array my_data::

   >>> pf = parameterfile.pfile('', data = my_data)

Accessing the values of a parameters file::

   >>> value1 = float(pf.data['name1'])

Values can also be retrieved using the get_value(name, dtype) function:

   >>> value2 = pf.get_value('name2', var_type = float)

Possible data types (dtype) are:

   - float
   - int
   - parameterfile.Bool ( converts ASCII True, False to a python bool value True, False)

If ``var_type`` keyword agrument is not used, the function tries to convert to float or bool. If both fails it
returns the string value.

Variable names are also called keys.
"""

import sys, os, string, math, re
import pdb
# function to create a dictionare from 2 lists: key, values


# create a dictionary from  an array of keys and values

class pfile:
    """
    Open a parameter file, read it and create a pfile object
    
    example::
    
    >>> pf=pfile('my_datafile')
    
    """
    def __init__(self, filename, data = None):
        """
	open a parameter file, scan for name=value pairs
	and return a dictionary with name value pairs

        """
	self.C=re.compile("^#")   # pattern for comment
	self.data = []
	self.keys=[]
	self.values=[]
        self.bool_res = {"false":False, "true":True}
	# pdb.set_trace()
        if data == None:
            # if no data are provided directly get them from the file
            self.filename = filename
            self.adata=open(filename,"r").readlines()   # read all data
        else:
            # store the data provided
            self.filename = 'no_file_provided'
            self.adata = data
	self.remove_blanks()
	self.make_dictionary()  # read data

    def Bool(self,x):
        """
        convert ascii True and False to bool values. Any other value will result in False
        example::
        
           >>> d.Bool('True')

        returns::

           >>> True

        
        """
        if x.lower() not in self.bool_res:
            # it's not a logical value, complain
            return None
        else:
            return self.bool_res[x.lower()]

    def remove_blanks(self): 
        # work through the list in reverse order
        indices = list(range( len( self.adata ))) # list of indices
	indices.reverse() # reverse order
	for i in indices: # loop through from the top
	    if len(self.adata[i].strip()) == 0:
		del self.adata[i] 
    def make_dictionary(self):
	# create an array of dictionaries
	self.keys=[]
	self.values = []
	do_add = False
	l_old = ''
	for l in self.adata:
	    l_l = l.strip()
	    if (self.C.match(l_l) == None) : # not a comment
	        # check for continuation
	        if do_add :
	            # add new line to end of last one and reset do_add flag
	            l_l = l_old[:-1] + l_l
	            do_add = False
	        # check for continuation character at the end of the line
	        do_cont = (l_l[-1] == '\\')
	        if do_cont:
	           # found a continuation flag it and save the current line
	           do_add = True
	           l_old = l_l
	        if (not do_cont):
	          # no more continuations, analyze the line
		  ll=l_l.split(';') # split along semicolons
		  for k in ll: 
		      f = k.split('=')
		      if len(f) != 2 :
			 print('problem with : ', k, ' -> skipping')
			 print('maybe leftover ; ?')
		      else :
			 self.keys.append((f[0]).strip())
			 self.values.append(f[1])
		      # end if
		  # end for k
		# end if (not do_count)
	    # end if
	# with the key value arrays make the dictionary
	self.make_dict(self.keys, self.values)
    def make_dict(self,keys,values):
	# create a dictionary from a list of keys and values
	p = []
	for j in range(len(keys)): # loop over keys
	    try:
		p.append( (keys[j].strip(),values[j].strip()) ) # an array of tuples
		self.data=dict(p) # create a dictionary
	    except:
		print('problem in data : ', keys, values)
	    #
	# end for
    def name(self):
	"""
	print the filename associate with this instance 

	"""
	print("Input file name : ", self.filename)
    def show_variable_names(self):
	"""
	print a list of variable names in the dictionary

	"""
	print(self.keys)
    def get_variable_names(self):
	"""

	return a list of keys

	"""
	return self.keys
    def get_data(self,keylist): # return data for a range of keys
	"""

	return all data according to the key list
	which has the following format:

	key1:key2:key3: ...

	"""
	akey=keylist.split(":")
	array=[]
	for i in akey:
	    array.append(self.data[i])
	return array
    def show_all_data(self):
	"""

	print all data and keys stored

	"""
	for l in self.data:
	    print(l, self.data[l])
    def get_all_data(self):
	"""

	return a list of all data stored

	"""
	return self.data
    def get_value(self, key, var_type = None):
	"""
	return a value of a given data type for key.
        
	>>> x = get_valye('my_var', var_type = float)
	
	prints an error message and returns none if it cannot convert. 

        possible data types are: float, int, parameterfile.Bool
	"""
        if var_type == None:
            try:
                var = float(self.data[key])
            except:
                var = self.Bool(self.data[key])
            if var != None:
                return var
            else:
                return self.data[key]
        elif var_type == int:
            try:
                var = int( float(self.data[key]) )
            except:
                print('cannot convert '+key+ ' to ', var_type)
                var = None
        else:
            try:
                var = var_type(self.data[key])
            except:
                print('cannot convert '+key+ ' to ', var_type)
                var = None
	return var
    
    def write_all(self,fp):
        """
        write all parameter file data into a new file
        using the provided file pointer fp which has been
        obtained from the open statment::
        
        >>> fp = open('new_file.par', 'w')
        >>> pf.write_all(fp)
        
        """
        for l in self.adata:
            fp.write(l+'\n')
        fp.close()
    # end of class datafile		   
#
