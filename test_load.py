import sys
sys.path.insert(0, '')
import asyncio
import api_server

async def main():
    try:
        api_server._load_models()
        print("Success")
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
