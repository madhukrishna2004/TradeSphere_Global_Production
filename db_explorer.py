import os
from dotenv import load_dotenv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QComboBox, QTableView, QLabel, QMessageBox, QTextEdit
)
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from sqlalchemy import create_engine, inspect, text
import pandas as pd

#(method) def __init__(self: Self@DBExplorer) -> None(class) QWidget(method) def setWindowTitle(a0: str) -> None
class DBExplorer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PostgreSQL DB Admin | TradeSphere Global")
        self.setGeometry(100, 100, 1000, 700)
        self.init_ui()
        self.connect_db()

    def init_ui(self):
        layout = QVBoxLayout()

        self.status_label = QLabel("🔄 Connecting to database...")
        layout.addWidget(self.status_label)

        self.table_selector = QComboBox()
        self.table_selector.currentTextChanged.connect(self.load_table_data)
        layout.addWidget(self.table_selector)

        self.data_view = QTableView()
        layout.addWidget(self.data_view)

        self.query_edit = QTextEdit()
        self.query_edit.setPlaceholderText("Write any SQL query here...")
        self.query_edit.setFixedHeight(100)
        layout.addWidget(self.query_edit)

        run_query_btn = QPushButton("▶️ Run SQL Query (Admin Access)")
        run_query_btn.clicked.connect(self.run_custom_query)
        layout.addWidget(run_query_btn)

        refresh_btn = QPushButton("🔄 Refresh Tables")
        refresh_btn.clicked.connect(self.connect_db)
        layout.addWidget(refresh_btn)

        self.setLayout(layout)

    def connect_db(self):
        try:
            load_dotenv()
            self.db_url = os.getenv("DATABASE_URL")
            if not self.db_url:
                raise Exception("DATABASE_URL not found in .env file.")

            self.engine = create_engine(self.db_url)
            self.connection = self.engine.connect()
            self.inspector = inspect(self.engine)

            tables = self.inspector.get_table_names()
            self.table_selector.clear()
            self.table_selector.addItems(tables)

            self.status_label.setText(f"✅ Connected to: {self.engine.url.database}")
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", str(e))
            self.status_label.setText("❌ Failed to connect")

    def load_table_data(self, table_name):
        if not table_name:
            return
        try:
            query = f"SELECT * FROM {table_name} LIMIT 100"
            df = pd.read_sql(query, self.engine)
            self.show_data(df)
            self.status_label.setText(f"📊 Showing: {table_name} ({len(df)} rows)")
        except Exception as e:
            QMessageBox.warning(self, "Data Load Error", str(e))
            self.status_label.setText("⚠️ Error loading table")

    def run_custom_query(self):
        sql = self.query_edit.toPlainText().strip()
        if not sql:
            QMessageBox.warning(self, "Empty Query", "Please enter a SQL query.")
            return

        try:
            # Detect SELECT to fetch rows
            if sql.lower().startswith("select"):
                df = pd.read_sql(sql, self.engine)
                self.show_data(df)
                self.status_label.setText(f"✅ SELECT executed — {len(df)} rows shown")
            else:
                # Run write/DDL queries
                with self.engine.begin() as conn:
                    result = conn.execute(text(sql))
                    affected = result.rowcount
                self.data_view.setModel(None)
                self.status_label.setText(f"✅ Query executed. Rows affected: {affected}")
                QMessageBox.information(self, "Success", f"Query executed successfully.\nRows affected: {affected}")
                self.connect_db()  # Refresh table list in case of DROP/ALTER
        except Exception as e:
            QMessageBox.critical(self, "Query Error", str(e))
            self.status_label.setText("❌ Query Failed")

    def show_data(self, df):
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(df.columns.tolist())

        for row in df.itertuples(index=False):
            items = [QStandardItem(str(field)) for field in row]
            model.appendRow(items)

        self.data_view.setModel(model)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = DBExplorer()
    window.show()
    sys.exit(app.exec_())
