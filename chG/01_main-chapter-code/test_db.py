import os
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

conninfo = os.getenv("DATABASE_URL")
print("DATABASE_URL:", conninfo)

dl = SQLAlchemyDataLayer(conninfo=conninfo)
print("Connected successfully!")
