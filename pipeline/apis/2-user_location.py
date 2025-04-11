#!/usr/bin/env python3
"""
Script to fetch and print the location of a GitHub user using the GitHub API.
"""

import requests
import sys
from datetime import datetime


def get_user_location(api_url):
    """ Fetches the location of a GitHub user from the provided API URL."""
    try:
        r = requests.get(api_url, timeout=5)
        data = r.json()
        if r.status_code == 404:
            return 'Not found'
        if r.status_code == 403:
            reset_time = int(r.headers.get('X-Ratelimit-Reset', 0))
            reset_in = (datetime.fromtimestamp(reset_time) -
                        datetime.now()).seconds // 60
            return f'Reset in {reset_in} min'
        return data.get('location', 'Location not available')
    except requests.RequestException as e:
        return f'Error: {e}'


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API URL>")
        sys.exit(1)

    api_url = sys.argv[1]
    print(get_user_location(api_url))
