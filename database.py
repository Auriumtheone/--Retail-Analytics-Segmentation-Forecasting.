

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.orm import declarative_base

# Sukuriame bazinį modelį
Base = declarative_base()

# Klientų lentelė (Customer_ID ir Country iš CSV)
class Customer(Base):
    __tablename__ = 'customers'
    Customer_ID = Column(Integer, primary_key=True, autoincrement=True)
    Country = Column(String(50))
    Segment_ID = Column(Integer, ForeignKey('customer_segments.Segment_ID'))
    Created_At = Column(DateTime)

    segment = relationship("CustomerSegment", back_populates="customers")

# Segmentų lentelė klientų grupavimui
class CustomerSegment(Base):
    __tablename__ = 'customer_segments'
    Segment_ID = Column(Integer, primary_key=True, autoincrement=True)
    Segment_Name = Column(String(50))
    Segment_Description = Column(String(200))

    customers = relationship("Customer", back_populates="segment")

# Pardavimų lentelė (Invoice, InvoiceDate, Quantity, Price, Revenue iš CSV)
class Transaction(Base):
    __tablename__ = 'transactions'
    Invoice = Column(String(20), primary_key=True)
    InvoiceDate = Column(DateTime)
    StockCode = Column(String(20), ForeignKey('products.StockCode'))
    Quantity = Column(Integer)
    Price = Column(Float)
    Revenue = Column(Float)
    Customer_ID = Column(Integer, ForeignKey('customers.Customer_ID'))

# Produktų katalogo lentelė (StockCode, Description, Price iš CSV)
class Product(Base):
    __tablename__ = 'products'
    StockCode = Column(String(20), primary_key=True)
    Description = Column(String(200))
    Price = Column(Float)

# Nauji klientai iš web sąsajos
class NewCustomer(Base):
    __tablename__ = 'new_customers'
    New_Customer_ID = Column(Integer, primary_key=True, autoincrement=True)
    Name = Column(String(100))
    Country = Column(String(50))
    Registration_Date = Column(DateTime)

# Modelio prognozių lentelė
class Prediction(Base):
    __tablename__ = 'predictions'
    Prediction_ID = Column(Integer, primary_key=True, autoincrement=True)
    Customer_ID = Column(Integer, ForeignKey('customers.Customer_ID'))
    Predicted_Segment = Column(String(50))
    Predicted_Sales = Column(Float)
    Model_Version = Column(String(10))

# Sukuriame MySQL ryšį (pritaikytą tavo prisijungimo būdą)
engine = create_engine('mysql+mysqlconnector://ORACLETM:akle7@localhost/retail_analysis_db')
Base.metadata.create_all(engine)

# Sukuriame sesiją darbui su duomenimis
Session = sessionmaker(bind=engine)
session = Session()