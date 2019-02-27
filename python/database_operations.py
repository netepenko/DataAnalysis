import sqlite3 as lite
#import os


#creates a database with necessary tables and fields
def create():
    conn = lite.connect('MainDB.db')
    with conn:
        cur = conn.cursor()
        #create tables
        #######################################################################
        cur.execute('''CREATE TABLE IF NOT EXISTS\
        Shot_List (\
        
        Shot INT,\
        Date TEXT,\
        File_Name TEXT,\
        Folder TEXT,\
        t_offset REAL,\
        N_chan\
        
        )''')
        #insert rows
        cur.execute('''INSERT INTO\
        Shot_List \
        
        VALUES (\
        
        29975,\
        '22-Aug-2013',\
        '29975_DAQ_220813-141746.hws',\
        'Data/',\
        0.0,\
        6\
        
        )''')
        
        #######################################################################
        
        cur.execute('''CREATE TABLE IF NOT EXISTS\
        Raw_Fitting (\
        
        Shot INT,\
        Channel INT,\
        
        dtmin REAL, dtmax REAL,\
        
        n_peaks_to_fit INT,\
        poly_order INT,\
       
        add_pulser TEXT,\
        pulser_rate REAL,\
        P_amp REAL,\
        
        use_threshold TEXT,\
        Vth REAL,\
        Vstep REAL,\
        
        n_sig_low REAL,\
        n_sig_high REAL,\
        n_sig_boundary REAL,\
        
        sig REAL\
        
        )''')
        
        cur.execute('''INSERT INTO\
        Raw_Fitting \
        
        VALUES (
        
        29975,\
        1,\
        
        0.01, 0.1,\
        
        10,\
        10,\
        
        'True',\
        10000,\
        1.0,\
        
        'True',\
        0.2,\
        0.2,\
        
        3.,\
        3.,\
        3.,\
        
        0.3
        
        )''')
        #######################################################################
        cur.execute('''CREATE TABLE IF NOT EXISTS\
        Peak_Sampling (\
        
        Shot INT,\
        Channel INT,\
        
        decay_time REAL,\
        rise_time REAL,\
        position REAL,\
        
        n_samp INT,\
        n_max INT,\
        n_below INT,\
        n_above INT,\
        
        b1 REAL, e1 REAL,\
        b2 REAL, e2 REAL,\
        b3 REAL, e3 REAL,\
        b4 REAL, e4 REAL,\
        b5 REAL, e5 REAL,\
        b6 REAL, e6 REAL,\
        b7 REAL, e7 REAL,\
        b8 REAL, e8 REAL,\
        b9 REAL, e9 REAL,\
        b10 REAL, e10 REAL,\
        b11 REAL, e11 REAL,\
        b12 REAL, e12 REAL\
        
        )''')
        
        cur.execute('''INSERT INTO\
        Peak_Sampling \
        
        VALUES (\
        
        29975,\
        1,\
        
        200e-9,\
        100e-9,\
        350e-9,\
        
        120,\
        20,\
        15,\
        50,\
        
        0.0264465750973, 0.0264511717629,\
        0.0264640275167, 0.0264695346329,\
        0.0265033656474, 0.0265088727637,\
        0.0265219299587, 0.0265274370749,\
        0.0265635664222, 0.0265690735384,\
        0.0266368021858, 0.0266423093020,\
        0.0266432752680, 0.0266487823842,\
        0.0266662141836, 0.0266717212999,\
        0.0267147567486, 0.0267202638648,\
        0.0268340257862, 0.0268395329025,\
        0.0268832345346, 0.0268887416508,\
        0.0268869429556, 0.0268924500719\
        
        )''')
        
       
        
        #######################################################################
        cur.execute('''CREATE TABLE IF NOT EXISTS\
        Rates_Plotting (\
        
        Shot INT,\
        Channel INT,\
        
        time_slice_width REAL,\
        
        h_min REAL, h_max REAL,\
        h_bins REAL,\
        
        
        draw_p TEXT,\
        draw_t TEXT,\
        draw_sum TEXT,\
        
        p_min REAL, p_max REAL,\
        t_min REAL, t_max REAL,\
        pul_min REAL, pul_max REAL,\
        
        A_init REAL,\
        sig_init REAL,\
        sig_ratio REAL\
        
        )''')
        
        
          
        cur.execute('''INSERT INTO 
        Rates_Plotting \
        
        
        VALUES (\
        
        29975,\
        1,\
        
        1.e-3,\
        
        0, 1.4,\
        150,\
        
        'True',\
        'False',\
        'False',\
        
        0.6, 0.8,\
        0.3, 0.5,\
        0.9, 1.2,\
        
        10,\
        0.2,\
        100\
        
        )''')
        
        
    print "Created DB"
    conn.close()

#create table in main DB
def create_table():
    conn = lite.connect('MainDB.db')
    with conn:
        cur = conn.cursor()
        #create tables
        #######################################################################
        cur.execute('''CREATE TABLE IF NOT EXISTS\
        Combined_Rates (\
        
        Shot INT,\
        
        Channels TEXT,\
        
        t_min REAL,\
        t_max REAL,\
        
        A_min REAL,\
        A_max REAL,\
        
        d_time REAL,\
        
        view_dir TEXT,\
        
        view_names TEXT,\
        
        r_min REAL,\
        r_max REAL,\
        
        use_all_variables TEXT,\
        
        calc_rate TEXT,\
        
        model TEXT\
        
        
        )''')
        #insert rows
        cur.execute('''INSERT INTO\
        Combined_Rates \
        
        VALUES (\
        
        29975,\
        
        '0,1,2,3',\
        
        0.,\
        0.5,\
        
        0.,\
        150.e3,\
        
        5.e-3,\
        
        './orbit_public/NSTX_output',\
        
        'nml_orb_NSTX-Case_3_0.3',\
        
        0.,\
        1.5,\
        
        'True',\
        
        'True',\
        
        'simple_gauss'\
        
        )''')
    conn.close()
    
#retrieve data
def retrieve(params, table, where): 
    ''' Retrieves paremeters from table in database. Takes 3 str arguments, 1 - paremeters to retrieve, 2 - table containing
    that parameters, 3 - condition to specify the line in table.
    Example: retrieve('Folder, File_Name', 'Shot_List', 'Shot = 29975')
    '''
    conn = lite.connect('../MainDB.db')
    #create query line
    qline='SELECT '+ params +' FROM ' + table + ' WHERE ' + where
    
    with conn:
        cur = conn.cursor()
        cur.execute(qline)
        return cur.fetchall()[0]
    conn.close()

#write to database
def writetodb(params, table, where):
    conn = lite.connect('../MainDB.db')
    #create query line
    qline='UPDATE '+ table +' SET ' + params + ' WHERE ' + where
    
    with conn:
        cur = conn.cursor()
        cur.execute(qline)
    conn.close()

#return pevious(existing in database) shot number
def prevshot(shot):
    '''Retrieves previous existing shot number, takes one input - shot number, will return
    shot number which exist in database before entered input shot number.
    Example: prevshot(29976)
    '''
    conn = lite.connect('../MainDB.db')
    (rowid,) = retrieve('ROWID', 'Shot_List', 'Shot = ' + str(shot))
    print rowid
    #create query line
    qline='SELECT Shot FROM Shot_List WHERE ROWID = ' + str(rowid-1) 
    with conn:
        cur = conn.cursor()
        cur.execute(qline)
        return cur.fetchall()[0][0]
    conn.close()
    
#duplicate certain row in databse
def copyrow(table, where, sub):
    '''Creates copy of the raw in table and changes some values in it.
    Takes 3 str arguments, 1 - table where to perform copying, 2 - specifies parameters defining the raw to be copied,
    3 - what to change in copied raw.
    Example: copyrow('Raw_Fitting', 'Shot = 29975 AND Channel = 0', 'Channel = 1')
    '''
    conn = lite.connect('../MainDB.db')
    #create query line
    qline='INSERT INTO '+ table + ' SELECT * '+'FROM ' + table + ' WHERE ' + where
    with conn:
        cur = conn.cursor()
        cur.execute(qline)
        qline1='UPDATE ' + table + ' SET ' + sub + ' WHERE ROWID = ' + str(cur.lastrowid) 
        cur.execute(qline1)
    conn.close()

if __name__ == "__main__":
    wheredb_cp = ('Shot = 29975 AND Channel = 0')
    print wheredb_cp
    #create()
    try:
        copyrow('Peak_Sampling', wheredb_cp, 'Shot = 29976, Channel = 0')
    except: 
        print "Didnt work"