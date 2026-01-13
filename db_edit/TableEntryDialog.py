#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:54:16 2024

@author: boeglinw
"""

import sqlite3, sys

from PyQt5.QtWidgets import QDialog, QApplication

# preapres the dialog layout for the selected table
#import LayoutsArray as LA
import LayoutsArrayScrollArea as LA

from analysis_modules import database_operations as db


# format dictionary

debug = False

actions = ['edit', 'copy', 'new']

class TableEntryDialog(QDialog):
    def __init__(self, dbname, shot, channel, version, table, title = "Edit DB Row Values", action = 'edit'):
        super().__init__()
        self.db_name = dbname  
        self.table = table
        self.shot = shot
        self.channel = channel
        self.version = version
        if action in actions: 
            self.action = action
        else:
            print(f'Do not know this action: {action}')
            self.action = 'edit'
        # get information about table 
        rows = db.get_table_information(self.db_name, self.table)
        self.db_col_data = []
        self.db_col_types = []
        self.table_col_names = []
        for cid, name, v_type, notnull, dflt_value, pk  in rows:
            self.db_col_data.append((f'Col {cid} : {name} ({v_type}):', f'{cid}_{name}'))
            self.db_col_types.append(v_type)
            self.table_col_names.append(name)
            if debug:
                print(f'{name} has type {v_type}')
        self.LA_ui = LA.Ui_Dialog()
        self.LA_ui.setupUi(self, self.db_col_data, dialog_size=(450,700), title = title)  # (450,400)
        self.show_entries()
        self.LA_ui.pushButtonSubmit.clicked.connect(self.update_entries)
        self.LA_ui.pushButtonDone.clicked.connect(self.quit)        
        self.show()
            
    def show_entries(self):
        if self.action == 'new':
            for i,LE in enumerate(self.LA_ui.LE):
                self.LA_ui.LE[i].setText('')        
            return            
        # get the current DB entries
        qwhat = '*'
        rows = []
        self.qwhere = None
        if self.shot != '':
            self.qwhere = f'Shot = {self.shot}'
        if (self.shot != '') and (self.channel != '') and (self.version != ''):
            self.qwhere += f' and Channel = {self.channel} and Version = {self.version}'
        try:
            rows = db.retrieve(self.db_name, qwhat, self.table, where = self.qwhere)
        except Exception as e:
            print(f'Cannot retreive data ! {e}')
        if rows != []:
            for i,LE in enumerate(self.LA_ui.LE):
                self.LA_ui.LE[i].setText(str(rows[0][i]))        
    
    def update_entries(self):
        table_values = []
        # setore current values in a list
        for i,LE in enumerate(self.LA_ui.LE):
            if self.db_col_types[i] == 'TEXT':
                table_values.append("'" + self.LA_ui.LE[i].text() + "'")
            else:
                table_values.append(self.LA_ui.LE[i].text())
        # update the table
        if self.action == 'new':
            try:
                db.insert_row_into(self.db_name, self.table, self.table_col_names, table_values)
            except Exception as e:
                print(f'Cannot insert new row into {self.table}: {e}')
        elif self.action == 'copy':
            # set substitutions
            subs = ''
            for i, t_name in enumerate(self.table_col_names):
                subs +=  f' {t_name} = {table_values[i]},'
            qsub = subs[:-1]  # skip the last comma
            print('--------> Subs = ', qsub)
            try:
                db.copy_row(self.db_name, self.table, self.qwhere, qsub)
            except Exception as e:
                print(f'Cannot copy {self.table}: {e}')
        else :                
            try:
                db.update_row(self.db_name, self.table, self.table_col_names, table_values, self.qwhere)
            except Exception as e:
                print(f'Cannot update {self.table}: {e}')
       
    def quit(self):
        self.close()

# for testing
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = TableEntryDialog('MAST-U_full_shot_listDB_test.db', 53433, 0, 0, 'Peak_Sampling')
    w.show()
    sys.exit(app.exec_())