#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:54:16 2024

@author: boeglinw
"""

import sqlite3, sys

from PyQt5.QtWidgets import QDialog, QApplication

# prepares the dialog layout for the selected table

import LayoutHelpTextArray as HTA

from analysis_modules import database_operations as db


# format dictionary

debug = False

class TableHelpDialog(QDialog):
    def __init__(self,table, help_info = 'edit_shot_db_help.db'):
        super().__init__()
        self.db_name = help_info
        self.d_size = (650,400)
        self.table = table + '_Help'
        # get information about table 
        rows = db.get_table_information(self.db_name, self.table)
        self.db_col_data = []
        self.db_col_types = []
        self.table_col_names = []
        for cid, name, v_type, notnull, dflt_value, pk  in rows:
            self.db_col_data.append((f'Col {cid} : {name}:', f'{cid}_{name}'))
            self.db_col_types.append(v_type)
            self.table_col_names.append(name)
            if debug:
                print(f'{name} has type {v_type}')
        self.HTA_ui = HTA.Ui_Dialog()
        self.HTA_ui.setupUi(self, self.db_col_data, dialog_size=self.d_size, title = f"Help for {table}")
        self.HTA_ui.buttonBoxOk.accepted.connect(self.quit)   
        self.show_entries()
        self.show()
    
    def show_entries(self):
        # get the current DB entries
        qwhat = '*'
        rows = []
        try:
            rows = db.retrieve(self.db_name, qwhat, self.table)
        except Exception as e:
            print(f'Cannot retreinve data from {self.table}: {e}')
        if debug:
            print(rows)
        if rows != []:
            for i,LH in enumerate(self.HTA_ui.label_Help):
                self.HTA_ui.label_Help[i].setText(str(rows[0][i]))        
                        
    def quit(self):
        self.close()

# for testing
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = TableHelpDialog('MAST-U_full_shot_listDB_test.db', 'Raw_Fitting')
    w.show()
    sys.exit(app.exec_())