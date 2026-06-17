import asyncio
import asyncpg

async def test_connection():
    conn = await asyncpg.connect(
        user="admin",
        password="admin",  # or the password you set
        database="my_chainlit_db",
        host="localhost"
    )
    print("Connected successfully!")
    await conn.close()

asyncio.run(test_connection())
