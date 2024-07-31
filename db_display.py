import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from PyQt5 import QtSql as qts

import numpy as np

"""
class CoffeeForm(qtw.QWidget):
    # Form to display/edit all info about a coffee

    def __init__(self, roasts):
        super().__init__()
        self.setLayout(qtw.QFormLayout())

        self.coffee_brand = qtw.QLineEdit()
        self.layout().addRow('Brand: ', self.coffee_brand)
        self.coffee_name = qtw.QLineEdit()
        self.layout().addRow('Name: ', self.coffee_name)
        self.roast = qtw.QComboBox()
        self.roast.addItems(roasts)
        self.layout().addRow('Roast: ', self.roast)
        self.reviews = qtw.QTableWidget(columnCount=3)
        self.reviews.horizontalHeader().setSectionResizeMode(
            2, qtw.QHeaderView.Stretch)
        self.layout().addRow(self.reviews)

    def show_coffee(self, coffee_data, reviews):
        self.coffee_brand.setText(coffee_data.get('coffee_brand'))
        self.coffee_name.setText(coffee_data.get('coffee_name'))
        self.roast.setCurrentIndex(coffee_data.get('roast_id'))
        self.reviews.clear()
        self.reviews.setHorizontalHeaderLabels(
            ['Reviewer', 'Date', 'Review'])
        self.reviews.setRowCount(len(reviews))
        for i, review in enumerate(reviews):
            for j, value in enumerate(review):
                self.reviews.setItem(i, j, qtw.QTableWidgetItem(value))

"""
class MainWindow(qtw.QMainWindow):

    def __init__(self):
        """MainWindow constructor.

        Code in this method should define window properties,
        create backend resources, etc.
        """
        super().__init__()
        # Code starts here
        self.stack = qtw.QStackedWidget()
        self.setCentralWidget(self.stack)

        # Connect to the database
        self.db = qts.QSqlDatabase.addDatabase('QSQLITE')
        self.db.setDatabaseName('../MAST_data/full_shot_listDB.db')
        if not self.db.open():
            error = self.db.lastError().text()
            qtw.QMessageBox.critical(
                None, 'DB Connection Error',
                'Could not open database file: 'f'{error}')
            sys.exit(1)

        # Check for existing tables
        tables = self.db.tables()

        # Make a query
        query = self.db.exec('SELECT count(*) FROM Shot_List')
        query.next()
        count = query.value(0)
        print(f'There are {count} shots in the database.')

        # Retreive the shots table
        query = self.db.exec('SELECT * FROM Shot_list ORDER BY Shot')
        shot_info = []
        while query.next():
            shot_info.append(query.value(1))

        # Retreive the coffees table using a QSqlQueryModel
        shots = qts.QSqlQueryModel()
        shots.setQuery(
            "SELECT * "
            "FROM Shot_List ORDER BY Shot")
        self.shot_list = qtw.QTableView()
        self.shot_list.setModel(shots)
        self.stack.addWidget(self.shot_list)
        self.stack.setCurrentWidget(self.shot_list)
        
        # Navigation between stacked widgets
        navigation = self.addToolBar("Navigation")
        navigation.addAction(
            "Back to list",
            lambda: self.stack.setCurrentWidget(self.shot_list))

        
        # Code ends here
        self.show()

    def get_id_for_row(self, index):
        index = index.siblingAtColumn(0)
        shot_id = self.shot_list.model().data(index)
        return shot_id
    """
    def show_coffee(self, coffee_id):
        # get the basic coffee information
        query1 = qts.QSqlQuery(self.db)
        query1.prepare('SELECT * FROM coffees WHERE id=:id')
        query1.bindValue(':id', coffee_id)
        query1.exec()
        query1.next()
        coffee = {
            'id': query1.value(0),
            'coffee_brand': query1.value(1),
            'coffee_name': query1.value(2),
            'roast_id': query1.value(3)
        }
        # get the reviews
        query2 = qts.QSqlQuery()
        query2.prepare('SELECT * FROM reviews WHERE coffee_id=:id')
        query2.bindValue(':id', coffee_id)
        query2.exec()
        reviews = []
        while query2.next():
            reviews.append((
                query2.value('reviewer'),
                query2.value('review_date'),
                query2.value('review')
            ))

        self.coffee_form.show_coffee(coffee, reviews)
        self.stack.setCurrentWidget(self.coffee_form)
    """

if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    # it's required to save a reference to MainWindow.
    # if it goes out of scope, it will be destroyed.
    mw = MainWindow()
    sys.exit(app.exec())
