import json
import os
import requests


""" 
# for https://serper.dev/
def tool_serp(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        "X-API-KEY": os.environ.get("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    res = response.json()
    # print(res)
    res = res["organic"]
    transformed_res = []

    for item in res:
        transformed_res.append(
            {
                "title": item["title"],
                "url": item["link"],
                "summary": item["snippet"],
                "images": [],
            }
        )

    transformed_res = {"references": transformed_res, "text": ""}
    return transformed_res 
"""

# for https://apyhub.com/utility/serp-rank
def tool_serp(query):
    payload = json.dumps({"keyword": query})
    headers = {
        "apy-token": os.environ.get("APYHUB_API_KEY"),
        "Content-Type": "application/json",
    }
    response = requests.post(
        "https://api.apyhub.com/extract/serp/rank", headers=headers, data=payload
    )
    res = response.json()
    res = res["data"]
    transformed_res = []
    for item in res:
        transformed_res.append(
            {
                "title": item["title"],
                "url": item["url"],
                "summary": item["description"],
                "images": [],
            }
        )
    transformed_res = {"references": transformed_res, "text": ""}
    return transformed_res
