#!/usr/bin/env python3
"""
stats about Nginx logs in mongo
"""
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs_collection = client.logs.nginx

    total_logs = logs_collection.count_documents({})
    print("{} logs".format(total_logs))

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods:")
    for method in methods:
        count = logs_collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, count))

    status_check_count = logs_collection.count_documents({
        "method": "GET",
        "path": "/status"
    })
    print("{} status check".format(status_check_count))
