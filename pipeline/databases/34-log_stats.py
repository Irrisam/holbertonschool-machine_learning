#!/usr/bin/env python3
"""
Script that provides some stats about Nginx logs stored in MongoDB
"""
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    nginx_collection = client.logs.nginx
    
    # Total number of logs
    total_logs = nginx_collection.count({})
    print("{} logs".format(total_logs))
    
    # Methods stats
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        count = nginx_collection.count({"method": method})
        print("\tmethod {}: {}".format(method, count))
    
    # GET requests to /status
    status_count = nginx_collection.count({"method": "GET", "path": "/status"})
    print("{} status check".format(status_count))