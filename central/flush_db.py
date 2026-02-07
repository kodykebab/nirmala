import asyncio
from redis_client import redis_client
from neo4j_client import neo4j_client

async def flush_all():
    print("Flushing Redis...")
    try:
        await redis_client.flushall()
        print("Redis flushed.")
    except Exception as e:
        print(f"Error flushing Redis: {e}")

    print("Flushing Neo4j...")
    try:
        async with neo4j_client.driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
        print("Neo4j flushed.")
    except Exception as e:
        print(f"Error flushing Neo4j: {e}")

    await redis_client.close()
    await neo4j_client.close()

if __name__ == "__main__":
    asyncio.run(flush_all())
