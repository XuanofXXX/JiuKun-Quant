import asyncio

async def fetch_data(index, sleep):
    print(f'start fetching {index}')
    await asyncio.sleep(sleep)
    print(f'end fetching {index}')
    
async def main():
    res = []
    for i in range(3):
        res.append(await fetch_data(i,i))
    # a = await fetch_data(1,1)
    # b = await fetch_data(2,2)
    return res

asyncio.run(main())