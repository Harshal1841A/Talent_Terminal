import asyncio
import json
import logging
import sys

from api_server import rank, RankRequest, _load_models

logging.basicConfig(level=logging.INFO)

def run():
    print("Loading models...")
    _load_models()
    
    with open("job_desc.txt", "r", encoding="utf-8") as f:
        jd_text = f.read()

    print("Running rank()...")
    req = RankRequest(jd_text=jd_text, top_n=10)
    try:
        res = rank(req)
        print("Success!", len(res["results"]))
    except Exception as e:
        print("Exception:", e)

if __name__ == "__main__":
    run()
