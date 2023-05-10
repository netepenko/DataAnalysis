"""
WB SQLITe data base initialization and operations to hande parameters needed for data analysis

- create('/Users/boeglinw/my_db.db'):  creates an initial data base

"""
import sqlite3 as lite
import sys
#import os

DATA_BASE_DIR = '/Users/boeglinw/Documents/boeglin.1/Fusion/Fusion_Products/DataAnalysis/'


#%%
# perpare data to be stored in data base
class db_table_data:
    def __init__(self, name, fields, types, values = None, special = None):
        self.name = name
        if values == None:
            self.values = [dict.fromkeys(fields)]
        else:
            self.values = [dict(zip(fields, v)) for v in values]  # convert array of values to array of dict
        self.field_names = self.values[0].keys()
        self.special = special
        self.types = types

    def create_table(self, db_connect):
        cur = db_connect.cursor()
        sql = f'CREATE TABLE IF NOT EXISTS {self.name} ('
        for i, s in enumerate(self.field_names):
            if i>0:
                sql += ', '
            sql += f'{s} {self.types[i]}'
        # add special commands, e.f.define primay key
        if self.special is not None:
            sql += ', '
            sql += self.special
        sql += ')'
        try:
            cur.execute(sql)
        except Exception as err:
                print(f'---> create_table problem with: {sql}, {err}')

    def insert_into(self, db_connect):
        cur = db_connect.cursor()
        for d in self.values:
            sql = f'INSERT INTO {self.name} VALUES ('
            for i,s in enumerate(self.field_names):
                if i>0:
                    sql += ', '
                sql += f'{d[s]}'
            sql += ')'
            try:
                cur.execute(sql)
            except Exception as err:
                print(f'---> insert_into problem with: {sql}, {err}')
        # all done




#%%
#creates a starting database with necessary tables and fields
def create(db_file):
    conn = lite.connect(db_file)
    # default values for creating tables:
    common_parameters_fields = ['Root_Folder']
    common_parameters_types = ['TEXT']
    common_parameters_values = ['"./"']
    
    common_parameters = db_table_data('Common_Parameters', common_parameters_fields, common_parameters_types, [common_parameters_values])
    
    
    # Shot list table
    shot_list_fields = ['Shot',         'Date', 'File_Name',' Folder', 'RP_position', 'RP_setpoint','t_offset', 'N_chan', 'Comment']
    shot_list_types =  ['INT not NULL', 'TEXT', 'TEXT',      'TEXT',   'REAL',        'REAL',         'REAL',   'INT',    'TEXT']
    shot_list_values = []
    shot_list_values.append( [29975,      '"22-Aug-2013"', '"29975_DAQ_220813-141746.hws"','"Data/"', 1.65, 0., 0., 6,   '"No comment"' ] )  # strings need to be enclosed in ""
    shot_list_values.append ([29879,      '"19-Aug-2013"', '"DAQ_190813-112521.hws"','"Data/"', 1.83, 0., 0., 6,   '"No comment"' ])   # strings need to be enclosed in ""
    shot_list_values.append([29880,      '"19-Aug-2013"', '"DAQ_190813-114059.hws"','"Data/"', 1.65, 0., 0., 6,   '"No comment"' ])   # strings need to be enclosed in "

    shot_list = db_table_data('Shot_List', shot_list_fields, shot_list_types, shot_list_values)

    # raw_fitting_table
    raw_fit_fields = ['Shot', 'Channel', 'Version',               'Comment', 'dtmin', 'dtmax', 'n_peaks_to_fit', 'poly_order', 'add_pulser', 'pulser_rate','P_amp', 'use_threshold', 'Vth', 'Vstep', 'n_sig_low', 'n_sig_high', 'n_sig_boundary', 'sig', 'min_delta_t', 'max_neg_V', 'Result_File_Name', 'Corrected_Data_File_Name', 'Input_File_Name']
    raw_fit_types = ['INT not NULL','INT not NULL','INT not NULL', 'TEXT',    'REAL',  'REAL',  'INT',            'INT',        'TEXT',       'REAL',       'REAL',  'TEXT',          'REAL','REAL',  'REAL',     'REAL',       'REAL'          ,  'REAL',  'REAL',    'REAL',           'TEXT',    'TEXT',    'TEXT']
    raw_fit_values = [29975,          0,             0,        '"No Comment"', 0.01,   0.1,     10,              10,           'True',        1000.,        1.0,    'True',          0.2,   0.2,       3.,           3.,         3.,              0.3,      1e-7,       -.3 ,        '"No File Saved"', '"No File Saved"', '" "']

    raw_fitting = db_table_data('Raw_Fitting', raw_fit_fields, raw_fit_types, [raw_fit_values], special = 'PRIMARY KEY (Shot, Channel, Version)')  # if only 1 row of values enter in backets

    # peak sampling
    peak_sampling_fields = ['Shot', 'Channel', 'Version',               'Comment', 'Vstep', 'Vth', 'Chi2', 'tmin', 'tmax', 'decay_time', 'rise_time', 'position', 'n_samp', 'n_max', 'n_below', 'n_above']
    peak_sampling_types = ['INT not NULL','INT not NULL','INT not NULL', 'TEXT',   'REAL',  'REAL', 'REAL', 'REAL', 'REAL', 'REAL',       'REAL',      'REAL',     'INT',    'INT',    'INT',     'INT']
    peak_sampling_values = [29975,          0,             0,        '"No Comment"', .1,     .3,     .1,     .1,    .15,    200e-9,      100e-9,      350e-9,    120,       20,      15,        50]


    peak_sampling_fields += ['b1',  'e1',  'b2',  'e2',  'b3',  'e3',  'b4',  'e4',  'b5',  'e5',  'b6',  'e6']
    peak_sampling_types +=  ['REAL','REAL','REAL','REAL','REAL','REAL','REAL','REAL','REAL','REAL','REAL','REAL']
    peak_sampling_values += [0.0264465750973, 0.0264511717629,0.0264640275167, 0.0264695346329, 0.0265033656474, 0.0265088727637, 0.0265219299587, 0.0265274370749, 0.0265635664222, 0.0265690735384,0.0266368021858, 0.0266423093020]

    peak_sampling_fields += ['b7',  'e7',  'b8',  'e8',  'b9',  'e9',  'b10',  'e10',  'b11',  'e11',  'b12',  'e12']
    peak_sampling_types +=  ['REAL','REAL','REAL','REAL','REAL','REAL','REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL']
    peak_sampling_values += [0.0266432752680, 0.0266487823842,0.0266662141836, 0.0266717212999, 0.0267147567486, 0.0267202638648, 0.0268340257862, 0.0268395329025, 0.0268832345346, 0.0268887416508,0.0268869429556, 0.0268924500719]

    peak_sampling =db_table_data('Peak_Sampling', peak_sampling_fields, peak_sampling_types, [peak_sampling_values], special = 'PRIMARY KEY (Shot, Channel, Version)')

    # rate plotting
    Rate_Analysis_fields = ['Shot', 'Channel', 'Version',               'Comment', 'time_slice_width', 'h_min', 'h_max' , 'h_bins', 'draw_p', 'draw_t', 'draw_pul' , 'draw_sum']
    Rate_Analysis_types = ['INT not NULL','INT not NULL','INT not NULL', 'TEXT',   'REAL',             'REAL',  'REAL',   'INT',   'TEXT',   'TEXT',   'TEXT',        'TEXT']
    Rate_Analysis_values = [29975,          0,             0,        '"No Comment"', 1.e-3,            0.,      1.4,      160,      '"True"',  '"False"',  '"False"',  '"False"']

    Rate_Analysis_fields+= ['p_min', 'p_max', 't_min', 't_max', 'pul_min', 'pul_max',  'sig_ratio', 'Result_File_Name', 'Input_File_Name']
    Rate_Analysis_types += ['REAL',  'REAL',  'REAL',  'REAL',  'REAL',    'REAL',       'REAL', 'TEXT', 'TEXT' ]
    Rate_Analysis_values += [0.6,     0.8,    0.3,      0.5,     0.9,        1.2,        100,  '"No File Saved"',  '" "']

    Rate_Analysis = db_table_data('Rate_Analysis', Rate_Analysis_fields, Rate_Analysis_types, [Rate_Analysis_values], special = 'PRIMARY KEY (Shot, Channel, Version)')


    # corrected shot list
    shot_list_corrected_fields = ['Shot',       'Channel',     'Version','      Iteration',   'File_Name', 'Folder', 'Comment']
    shot_list_corrected_types = ['INT not NULL','INT not NULL','INT not NULL', 'INT not NULL', '""',       '""',      '""']
    shot_list_corrected_values = [29975,          0,             0,              0,              '"No File"',  '"NO Folder"', '"No comment"']
    
    
    shot_list_corrected = db_table_data('Shot_List_Corrected', shot_list_corrected_fields, shot_list_corrected_types, values = [shot_list_corrected_values], special = 'PRIMARY KEY (Shot, Channel, Version, Iteration)' )

    # ocmbine rates 
    comb_rates_fields = ['Shot', 'Channels', 't_min', 't_max', 'A_min', 'A_max', 'd_time', 'view_dir', 'view_names', 'r_min', 'r_max', 'use_all_variables', 'calc_rate', 'model', 'Comment']
    comb_rates_types = [ 'INT',  'TEXT',     'REAL',  'REAL',  'REAL',  'REAL',  'REAL',   'TEXT',     'TEXT',       'REAL',  'REAL',  'TEXT',              'TEXT',      'TEXT',  'TEXT']
    comb_rates_values = [29975,  '"0,1,2,3"',  0.,       0.5,     0.,      150.e3,  5.e-3,    '"./orbit_public/NSTX_output"', '"nml_orb_NSTX-Case_3_0.3"', 0., 1.5, '"True"',  '"True"', '"Simple Gauss"', '"No Comment"']

    comb_rates =db_table_data('Combined_Rates', comb_rates_fields, comb_rates_types, [comb_rates_values])

    with conn:
        
        common_parameters.create_table(conn)
        common_parameters.insert_into(conn)
        
        shot_list.create_table(conn)
        shot_list.insert_into(conn)
        #
        raw_fitting.create_table(conn)
        raw_fitting.insert_into(conn)
        #
        peak_sampling.create_table(conn)
        peak_sampling.insert_into(conn)
        #
        Rate_Analysis.create_table(conn)
        Rate_Analysis.insert_into(conn)
        #
        comb_rates.create_table(conn)
        comb_rates.insert_into(conn)
        #
        shot_list_corrected.create_table(conn)
        shot_list_corrected.insert_into(conn)
        
    print("Created DB")
    conn.close()

#%% check if data exist
def check_condition(db_file, table, where):
    """
    Check if a record satsifying the condition where exists

    Parameters
    ----------
    db_file : str
        database file name.
    table : str
        table name.
    where : str
        SQL string for conditional selection.

    Returns
    -------
    bool
        True: a record exists
        False: no reecord exists

    """
    qline = f'select 1 from {table} where exists( select 1 from {table} where {where})'
    conn = lite.connect(DATA_BASE_DIR + db_file)
    with conn:
        cur = conn.cursor()
        try:
            cur.execute(qline)
        except Exception as e:
            print(f'probem with command: {qline}')
            print(f'Error: {e}')
            return False
        res = cur.fetchall()
    conn.close()
    if res == []:
        return False
    else:
        return True

#%% raise sys.exit with error message if data do not exist

def check_data(db_file, table, where, message = None):
    """
    Check if a record satsifying the condition where exists.
    If not raises sys.exit with custom error message

    Parameters
    ----------
    db_file : str
        database file name.
    table : str
        table name.
    where : str
        SQL string for conditional selection.
    message: str, optional
        Error message to be displayed

    Returns
    -------

    """    
    if check_condition(db_file, table, where):
        return
    else:
        if message is None:
            message = f'No data in {table} for condition {where}'
        sys.exit(message)
        

#%%retrieve data
def retrieve(db_file, params, table, where = None):
    """
    Retrieves paremeters from table in database.

    Parameters
    ----------
    db_file : str
        database file name.
    params : str
        parameter (field) name to retreive.
    table : str
        table name where the parameter is located.
    where : str
        SQL string for conditional selection.

    Returns
    -------
    
    list of values   

    Example: retrieve(db_file, 'Folder, File_Name', 'Shot_List', 'Shot = 29975')
    
    returns:
        
        [('Data/', '29975_DAQ_220813-141746.hws')]

    """

    conn = lite.connect(DATA_BASE_DIR + db_file)
    #create query line
    if where is None:
        qline='SELECT '+ params +' FROM ' + table
    else:
        qline='SELECT '+ params +' FROM ' + table + ' WHERE ' + where
    print(f'Using database file : {DATA_BASE_DIR + db_file}')
    with conn:
        cur = conn.cursor()
        try:
            cur.execute(qline)
        except Exception as e:
            print(70*'-')
            print(f'probem with command: {qline}')
            print(f'Error: {e}')
            print(70*'-')
            return []
        result = cur.fetchall()
        return result
    conn.close()

#%%write to database
def writetodb(db_file, params, table, where):
    """
    
    Write parameters to data base

    Parameters
    ----------
    db_file : str
        database file name.
    params : str
        parameter (field) name to set.
    table : str
        table containing hte parameter
    where : str
        selection criteria

    Returns
    -------
    None.

    Example:
        
        writetodb(db_file, 'Folder = "Data/"', 'Shot_list', 'Shot = 29975')
        
    """
    conn = lite.connect(DATA_BASE_DIR + db_file)
    #create query line
    qline='UPDATE '+ table +' SET ' + params + ' WHERE ' + where
    print(f'writetodb: {qline}')
    with conn:
        cur = conn.cursor()
        try:
            cur.execute(qline)
        except Exception as e:
            print(f'probem with command: {qline}')
            print(f'Error: {e}')
    conn.close()

#%% insert a new row, this produces an error if there is a uniqueness error

def insert_row_into(db_file, table, names, values):
    conn = lite.connect(DATA_BASE_DIR + db_file)
    cur = conn.cursor()
    p_1 = f'INSERT INTO  {table} ('
    p_2 = ' VALUES ('
    for i,d in enumerate(names):
        p_1 += d + ','
        p_2 += f'{values[i]}' + ','
    p1 = p_1[:-1] + ')'
    p2 = p_2[:-1] + ')'
    sql = p1 + p2
    print(sql)
    with conn:
        try:
            cur.execute(sql)
        except Exception as err:
            print(f'---> insert_into problem with: {sql}, {err}')
    # all done
    conn.close()

#%%  update_row
def update_row(db_file, table, names, values, where):
    conn = lite.connect(DATA_BASE_DIR + db_file)
    cur = conn.cursor()
    cmd0 = f'UPDATE  {table} SET ' 
    cmd1 = ','.join( [f'{nn} = {values[i]}' for i,nn in enumerate(names)] )
    cmd2 = f' WHERE {where}'
    q_line  = cmd0 + cmd1 + cmd2
    print(f'update_row: {q_line}')
    with conn:
        cur = conn.cursor()
        try:
            cur.execute(q_line)
        except Exception as e:
            print(f'probem with command: {q_line}')
            print(f'Error: {e}')
    conn.close()    
    
    
#%% some utility functions
def get_shot_list(db_file):
    sl = retrieve(db_file, 'Shot, N_chan', 'Shot_List')
    return sl
    

def get_folder(db_file, shot):
    (dd,) = retrieve(db_file, 'Folder', 'Shot_List', f'Shot = {shot}')[0]
    return dd

def set_folder(db_file, shot, dir_name):
    q_what = f'Folder = "{dir_name}"'
    q_table = 'Shot_List'
    q_where = f'Shot = {shot}'
    writetodb(db_file, q_what, q_table, q_where)    


#%% get previous shot
def prevshot(db_file, shot):
    """
    
    Retrieves previous existing shot number, takes one input - shot number, will return
    shot number which exist in database before entered input shot number.

    Parameters
    ----------
    db_file : str
        database file name.
    shot : str
        shot number

    Returns
    -------
    TYPE
        previous shot number
        
    Example: prevshot(29976)

    """
    conn = lite.connect(DATA_BASE_DIR + db_file)
    try:
        (rowid,) = retrieve(db_file, 'ROWID', 'Shot_List', f'Shot = {shot}')[0]
    except:
        print(f'--> could not find shot {shot} in {db_file}')
        return None
    print(rowid)
    #create query line
    qline='SELECT Shot FROM Shot_List WHERE ROWID = ' + str(rowid-1)
    with conn:
        cur = conn.cursor()
        try:
            cur.execute(qline)
        except Exception as e:
            print(f'probem with command: {qline}')
            print(f'Error: {e}')
        return cur.fetchall()
    conn.close()


#%% duplicate row
def duplicate_row(db_file, table, where_cp):
    """
    duplicates a row. If it is versioned it increments the version number

    Parameters
    ----------
    db_file : str
        database file name.
    table : str
        table name
    where_cp : str
        where statement to select row

    Returns
    -------
    has_version : Bool
        True if thie is a versioned row
    max_version : Int
        latest version number

    Be carful when using queries with versions

    """
    conn = lite.connect(DATA_BASE_DIR + db_file)
    with conn:
        cur = conn.cursor()
        # get table info
        t_info = cur.execute(f'PRAGMA table_info({table})').fetchall()
        field_names = [s[1] for s in t_info]
        field_types = [s[2] for s in t_info]
        # check if a version field exists'
        has_version = 'Version' in field_names
        # for finding the max row remove the Version statement from the query if it contains one
        wcp = where_cp.split(' ')
        if 'Version' in wcp:
            del wcp[wcp.index('Version')-1:wcp.index('Version')+3]
            where_cp_no_version = ' '.join(wcp)
        else:
            where_cp_no_version = where_cp
        
        # if it is a versioned row get all field names except version
        if has_version:
            # check if record exists
            if not (check_condition(db_file, table, where_cp) ):
                print(f'--> No data for condition: {where_cp} in {table},  cannot duplicate')
                return has_version, -1                    
            # get the row with the largest version number
            query = f'SELECT *, max(Version) FROM {table}' + ' WHERE ' + where_cp_no_version
            max_row = list(cur.execute(query).fetchall()[0])
            # increment the version number
            max_row[field_names.index('Version')] += 1
            q_names = '('+ ''.join([f'{x},' for x in field_names])[:-1] + ')'
            # convert values to string
            value_list = [f'"{x}",' if field_types[i] == 'TEXT' else f'{x},' for i,x in enumerate(max_row[:-1])]
            q_values = '('+ ''.join(value_list)[:-1] + ')'
            # insert new row with new version number
            q_insert = 'INSERT INTO '+ table + ' ' + q_names +' VALUES ' + q_values
            max_version = max_row[field_names.index('Version')]
            print(f'duplicate row: {q_insert}')
            try:
                cur.execute(q_insert)
            except Exception as err:
                print(f'Cannot execute {q_insert} : {err}')
                max_version = -cur.lastrowid
                return True, max_version
        else:
            max_rowid = retrieve(db_file, 'max(ROWID)', table, where_cp)[0][0]  # get maximal RWOID for this condition
            q_insert='INSERT INTO '+ table + ' SELECT * '+'FROM ' + table + ' WHERE ' + where_cp + f' AND ROWID = {max_rowid}'
            try:
                cur.execute(q_insert)
            except Exception as e:
                print(f'probem with command: {q_insert}')
                print(f'Error: {e}')            
            max_version = -cur.lastrowid
    conn.close()
    return has_version, max_version


#%%duplicate certain row in databse
def copyrow(db_file, table, where_cp, substitutions):
    """
    Creates a copy of the row in table and changes some parameter values in it.

    Parameters
    ----------
    db_file : str
        database file name.
    table : str
        table containing hte parameter.
    where_cp : str
        selection criteria.
    substitutions : str
        parameter setting command.

    Returns
    -------
    None.
    
    Example: copyrow(db_file, 'Raw_Fitting', 'Shot = 29975 AND Channel = 0', 'Channel = 1')

    """
    # check if row exists
    if not (check_condition(db_file, table, where_cp) ):
        print(f'--> No data for condition: {where_cp} in {table},  cannot copy')
        return
    # first duplicate row
    has_version, max_version = duplicate_row(db_file, table, where_cp)
    # insert new values
    if has_version:
        # q_update='UPDATE ' + table + ' SET ' + sub + where_cp + f' AND Version = {max_version}'
        writetodb(db_file, substitutions, table, where_cp + f' AND Version = {max_version}' )
        # try to reset the version
        where_new = ' AND '.join(substitutions.split(','))
        writetodb(db_file, f'Version = {max_version-1}', table, where_new + f' AND Version = {max_version}' )
    else:
        # q_update='UPDATE ' + table + ' SET ' + sub + f' WHERE ROWID = {-max_version}'
        writetodb(db_file, substitutions, table,  f' ROWID = {-max_version}')
    # print('q_update = ',  q_update)



def delete_row(db_file, table, where_del):
    """
    Delete a row in a table 

    Parameters
    ----------
    db_file : str
        database file name.
    table : str
        table where the row should be deleted.
    where_del : str
        selection criteria for row to be deleted (careful !).

    Returns
    -------
    None.

    """
    conn = lite.connect(DATA_BASE_DIR + db_file)
    
    if not (check_condition(db_file, table, where_del) ):
        print(f'--> No data for condition: {where_del} in {table}, nothing done')
        return 
    q_line = f'delete from {table} where {where_del};'
    print(f'{q_line}')
    with conn:
        cur = conn.cursor()
        try:
            cur.execute(q_line)
        except Exception as e:
            print(f'probem with command: {q_line}')
            print(f'Error: {e}')
    conn.close()   
    

#%% Tests
"""
if __name__ == "__main__":
    db_file = 'New_MainDB1.db'
    wheredb_cp = ('Shot = 29975 AND Channel = 0')
    wheredb_cp_new = ('Shot = 29980 AND Channel = 3')
    print(wheredb_cp)
    # create(db_file)
    # try:
    copyrow(db_file, 'Peak_Sampling', wheredb_cp, 'Shot = 29980, Channel = 3')
    print(retrieve(db_file, '*','Peak_Sampling', wheredb_cp))
    print(retrieve(db_file, '*','Peak_Sampling', wheredb_cp_new))
    # except:
    #    print("Didnt work")
"""


