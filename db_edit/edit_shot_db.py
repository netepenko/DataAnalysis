#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:54:16 2024

This tool is setup to edit the shot db. You can add and delete entries.

@author: boeglinw
"""

import sqlite3, sys

from PyQt5 import QtWidgets as qtw

from PyQt5 import QtCore as qtc

import os

from main_window_ui import Ui_MainWindow

import TableEntryDialog as TED
import TableHelpDialog as THD

from analysis_modules import database_operations as db


debug = False

def MB_YesNo(q_text):
    """
    Yes/No message box

    Parameters
    ----------
    q_text : str
        question text.

    Returns
    -------
    bool
        answer (True for Yes).

    """
    q = qtw.QMessageBox()
    q.setText(q_text)
    q.setStandardButtons(qtw.QMessageBox.Yes | qtw.QMessageBox.Cancel)
    res = q.exec()
    if res == qtw.QMessageBox.Yes:
        return True
    else:
        return False
    
def MB_Error(q_text):
    q = qtw.QErrorMessage()
    q.showMessage(q_text)
    q.exec_()    

def SetComboText(CB, text):
    """
    Set text for a combobox if it exists

    Parameters
    ----------
    CB : PyQT QComboBox
        current combobox.
    text :str
        item to be pointed to.

    Returns
    -------
        None

    """
    if CB.findText(text) >= 0:
        CB.setCurrentText(text)
    else:
        print(f'SetComboText: {text} not found !')
    return

class MainWindow(qtw.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # setup UI
        self.setupUi(self)
        # Create an instance of the GUI
        self.db_file = ''
        # setup connections
        # File menu
        self.action_Open.triggered.connect(self.open_file)
        self.action_Save.triggered.connect( self.save_file)
        self.action_Close.triggered.connect( self.close_file)
        self.action_Quit.triggered.connect(self.quit_all)
        
        self.checkBoxFixShot.setCheckState(qtc.Qt.Unchecked)
        
        # make shotlist combobox searchable
        self.comboBoxShot.setEditable(True)
        self.comboBoxShot.setInsertPolicy(qtw.QComboBox.NoInsert)
        
        # combobox connections
        self.comboBoxTable.currentTextChanged.connect(self.update_scv)
        self.comboBoxShot.currentTextChanged.connect(self.update_cv)
        self.comboBoxChannel.currentTextChanged.connect(self.update_v)
        self.comboBoxVersion.currentTextChanged.connect(self.save_current_values)

        """
        self.comboBoxShot.activated.connect(self.update_cv)
        self.comboBoxChannel.activated.connect(self.update_v)
        self.comboBoxVersion.activated.connect(self.save_current_values)
        """
        
        # buttons
        self.pushButtonEdit.clicked.connect( self.edit_db_entry)
        self.pushButtonRefresh.clicked.connect( self.update_scv)
        self.pushButtonAdd.clicked.connect( self.add_db_entry)
        self.pushButtonDuplicate.clicked.connect( self.duplicate_row)
        self.pushButtonDelete.clicked.connect( self.delete_row)
        self.pushButtonTableHelp.clicked.connect( self.show_table_help)

        # dlg = THD.TableHelpDialog('MAST-U_full_shot_listDB_test.db', 'Raw_Fitting')
        # dlg.exec()   
        self.save_current_values()
        
    def save_current_values(self):
        print('save current values')
        # store the results
        self.current_table = self.comboBoxTable.currentText()
        self.current_shot =  self.comboBoxShot.currentText()
        self.current_channel =  self.comboBoxChannel.currentText()
        self.current_version = self.comboBoxVersion.currentText()

        
    def edit_db_entry(self):
        print('start edit DB entry')
        self.show_current_values()
        # make sure the information is complete
        dlg = TED.TableEntryDialog(self.db_file, \
                                   self.current_shot, \
                                       self.current_channel, \
                                           self.current_version, \
                                               self.current_table, \
                                                   title = f'Edit row for : {self.current_table}')
        dlg.exec()
        
    def add_db_entry(self):
           print('add DB entry')
           self.show_current_values()
           if self.current_table == 'Common_Parameters':
               MB_Error('Cannot add entries to Common_Parameters ! (use edit instead)')
               return
               
           # make sure the information is complete
           dlg = TED.TableEntryDialog(self.db_file, \
                                      self.current_shot, \
                                          self.current_channel, \
                                              self.current_version, \
                                                  self.current_table, \
                                                      title = f'Edit row for : {self.current_table}',\
                                                          action = 'copy')
           dlg.exec()     

    def duplicate_row(self):
        print('===========>  duplicate the current row')
        print('Table = ',self.current_table)
        print('Shot = ', self.current_shot)
        print('Channel = ', self.current_channel)
        print('Version = ', self.current_version)
        
        # save current values
        table = self.current_table
        shot = self.current_shot
        channel = self.current_channel
        version = self.current_version
       
        # setup query
        if (self.current_shot == ''):
            qwhere = 'True'
            has_version = False
        elif (self.current_channel == '') and (self.current_version == ''): 
            qwhere = f'Shot = {self.current_shot}'
        elif (self.current_version == ''):
            qwhere = f'Shot = {self.current_shot} and Channel = {self.current_channel}'
        else:
            qwhere = f'Shot = {self.current_shot} and Channel = {self.current_channel} and Version = {self.current_version}'
        if (self.current_table == 'Shot_List') :
            MB_Error(f'Cannot duplicate row for {self.current_table} use "add"')
            return
        if (self.current_table == 'Common_Parameters'):
            MB_Error(f'Cannot duplicate row for {self.current_table}')
            return       
        try:
            has_version, max_version, qwhere_no_version = db.duplicate_row(self.db_file, self.current_table, qwhere)
        except Exception as e:
            MB_Error(f'Cannot duplicate row  in {self.current_table} where {qwhere}: {e}')
            return
        self.update_scv()
        #restore previous selections
        self.set_combo_selections(table, shot, channel, version)
    

    
    def delete_row(self):
        if not MB_YesNo("Are you sure?"):
            return
        print('delete current row')
        print('===========>  delete the current row')
        print('Table = ',self.current_table)
        print('Shot = ', self.current_shot)
        print('Channel = ', self.current_channel)
        print('Version = ', self.current_version)

        # save current values
        table = self.current_table
        shot = self.current_shot
        channel = self.current_channel
        version = self.current_version
        
        # setup query
        if (self.current_shot == ''):
            qwhere = 'True'
            has_version = False
        elif (self.current_channel == '') and (self.current_version == ''): 
            qwhere = f'Shot = {self.current_shot}'
        elif (self.current_version == ''):
            qwhere = f'Shot = {self.current_shot} and Channel = {self.current_channel}'
        else:
            qwhere = f'Shot = {self.current_shot} and Channel = {self.current_channel} and Version = {self.current_version}'
        if (self.current_table == 'Common_Parameters'):
            MB_Error(f'Cannot delete row for {self.current_table}')
            return       
        try:
            db.delete_row(self.db_file, self.current_table, qwhere)
        except Exception as e:
            MB_Error(f'Cannot delete row in {self.current_table} where {qwhere}: {e}')
            return        
        self.update_scv()
        #restore previous selections
        self.set_combo_selections(table, shot, channel, version)


        
    def show_table_help(self):
        if self.db_file is None:
            print('No data base, nothing to show !')
            return
        if debug:
            print('show table help')
        # dlg = THD.TableHelpDialog(self.db_file, 'Raw_Fitting')
        dlg = THD.TableHelpDialog(self.current_table, help_info = 'edit_shot_db_help.db')
        dlg.exec()        
        
    def skip_db_edit(self):
        print('Selected to skip editing any of this')
        
    def open_file(self):
        filename, _ = qtw.QFileDialog.getOpenFileName(
            self,
            'Select a data base file to open…',
            qtc.QDir.currentPath(),
            'datafile Files (*.db) ;; All Files (*)'
        )
        if filename:
            loc_directory, f_name = os.path.split(filename)
            #self.labelDBname.setText(db.DATA_BASE_DIR)
            rel_path = os.path.relpath(loc_directory)
            self.statusBar().showMessage(f"DB dir: {rel_path}/") 
            #
            self.current_dir = loc_directory
            self.db_file = f_name
            qtc.QDir.setCurrent(loc_directory)
            self.setWindowTitle(f_name)
            # update current data base directory
            db.DATA_BASE_DIR = self.current_dir + '/'
            # set title text
            print(f'db.DATA_BASE_DIR = {db.DATA_BASE_DIR}', rel_path)
            # setup table list
            #self.setup_combo_tables()
            self.setup_table_list()
            
    def setup_table_list(self):
        try:
            table_list = db.get_list_of_tables(self.db_file)
            # set the content of the table combo box
            i_t = 0
            for t_name in table_list:
                if t_name.lower().find('_help') >= 0 :
                    continue
                else:
                    self.comboBoxTable.addItem(t_name, i_t)
                    i_t += 1
        except Exception as e:
            print(f'Cannot get table list : {e}')
            return
        
    def update_scv(self):
        print(f'===> update_scv: db table trigger, update all, shot = {self.current_shot}')
        print( '===> update_scv: setup shot combo')
        if self.checkBoxFixShot.isChecked():
            print(f'===> update_scv: shot number {self.current_shot} is fixed, no changein shot list')
        else:
            self.setup_shot_list()
        print('setup channel combo')
        self.setup_channel_list()
        print('setup version combo')
        self.setup_version_list()
        self.save_current_values()

    def update_cv(self):
        if debug:
            print('---> Shot combo triggered, update channel, version ')
            print('setup channel combo')
        self.setup_channel_list()
        if debug:
            print('setup version combo')
        self.setup_version_list()
        self.save_current_values()        

    def update_v(self):
        if debug:
            print('--> Channel combo triggered, updateversion ')
            print('setup version combo')
        self.setup_version_list()
        self.save_current_values()
        
        
        

        
    def show_current_values(self):
        # make sure comboboxes show the current values
        print('===========>  Current Values')
        print('Table = ',self.current_table)
        print('Shot = ', self.current_shot)
        print('Channel = ', self.current_channel)
        print('Version = ', self.current_version)

    
    def set_combo_selections(self, table, shot, channel, version): 
        # set the current selections
        print(f'Set CB texts: {table} {shot} {channel} {version}')
        if table is not None:
            SetComboText(self.comboBoxTable, table)
        if shot is not None:
            SetComboText(self.comboBoxShot, shot)
        if channel is not None:
            SetComboText(self.comboBoxChannel, channel)
        if version is not None:
            SetComboText(self.comboBoxVersion, version)
        
        
    def setup_shot_list(self):
        # get selected table
        current_table = self.comboBoxTable.currentText()
        # clear the shitlist content
        self.comboBoxShot.clear()
        print(f'====> setup_shot_list : current_table = {current_table}')
        try:
            shots = [r[0] for r in db.retrieve(self.db_file, 'Shot', current_table, distinct = True)]
        except Exception as e:
            print(f'cannot get shot list {e}')
            return
        if shots == []:
            # nothing returned, nothing to do
            return
        for i, s_name in enumerate(sorted(shots)):
            self.comboBoxShot.addItem(str(s_name), i)
        

    def setup_channel_list(self):
        # get selected table
        current_table = self.comboBoxTable.currentText()
        current_shot = self.comboBoxShot.currentText()
        print(f'---> setup_channel_list: current_table = {current_table}, current_shot = {current_shot}')
        if current_shot == '':
            qwhere = None
        else:
            qwhere = f"Shot = {current_shot}"
        self.comboBoxChannel.clear()
        try:
            chs = [r[0] for r in db.retrieve(self.db_file, 'Channel', current_table, qwhere, distinct = True)]
            if debug:
                print('channel list :', chs)
        except Exception as e:
            print(f'cannot get channel list {e}')
            return
        for i, c_name in enumerate(chs):
            self.comboBoxChannel.addItem(str(c_name), i)

    
    def setup_version_list(self):
        print('update version list called')
        # get selected table
        current_table = self.comboBoxTable.currentText()
        current_shot = self.comboBoxShot.currentText()
        current_channel = self.comboBoxChannel.currentText()
        print(f'++++> setup_version_list : current_table = {current_table}, current_shot = {current_shot}, current_channel = {current_channel}')
        if (current_shot == ''):
            qwhere = None
        elif (current_channel == ''):
            qwhere = f"Shot = {current_shot}"
        else:
            qwhere = f"Shot = {current_shot} and Channel = {current_channel}"
        self.comboBoxVersion.clear()
        try:
            vers = [r[0] for r in db.retrieve(self.db_file, 'Version', current_table, qwhere, distinct = True)]
            if debug:
                print('version list :', vers)
        except Exception as e:
            print(f'cannot get version list {e}')
            return
        for i, v_name in enumerate(vers):
            self.comboBoxVersion.addItem(str(v_name), i)


    def setup_rowid_list(self):
        print('update rowid list called')
        # get selected table
        current_table = self.comboBoxTable.currentText()
        current_shot = self.comboBoxShot.currentText()
        current_channel = self.comboBoxChannel.currentText()
        print(f'current_table = {current_table}, current_shot = {current_shot}, current_channel = {current_channel}')
        if (current_shot == ''):
            qwhere = None
        elif (current_channel == ''):
            qwhere = f"Shot = {current_shot}"
        else:
            qwhere = f"Shot = {current_shot} and Channel = {current_channel}"
        self.comboBoxVersion.clear()
        try:
            vers = [r[0] for r in db.retrieve(self.db_file, 'Version', current_table, qwhere, distinct = True)]
            if debug:
                print('version list :', vers)
        except Exception as e:
            print(f'cannot get version list {e}')
            return
        for i, v_name in enumerate(vers):
            self.comboBoxVersion.addItem(str(v_name), i)


         
    def save_file(self):
        print('save file selected')

    def close_file(self):
        print('close file selected')


    def quit_all(self):
        print('quit everything')
        self.close()
        qtw.QApplication.quit()
 
    
# for testing
if __name__=="__main__":
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
