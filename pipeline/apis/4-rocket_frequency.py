#!/usr/bin/env python3

from collections import Counter
import requests


def get_launches_per_rocket():
    """
    Fetch all SpaceX launches and count the number of launches per rocket.
    Returns a sorted list of (rocket_name, launch_count) tuples.
    """
    # Fetch all launches from the SpaceX API
    response = requests.get(
        "https://api.spacexdata.com/v4/launches", timeout=20)
    launches = response.json()

    # Dictionary to store rocket ID to name mapping
    rocket_names = {}
    # List to store rocket IDs for each launch
    rocket_ids = []

    # Collect rocket IDs and their names
    for launch in launches:
        rocket_id = launch['rocket']
        rocket_ids.append(rocket_id)

        if rocket_id not in rocket_names:
            rocket_response = requests.get(
                f"https://api.spacexdata.com/v4/rockets/{rocket_id}",
                timeout=20)
            rocket_data = rocket_response.json()
            rocket_names[rocket_id] = rocket_data['name']

    rocket_counts = Counter(rocket_ids)

    result = [(rocket_names[rocket_id], count)
              for rocket_id, count in rocket_counts.items()]

    result.sort(key=lambda x: (-x[1], x[0]))

    return result


def display_launches_per_rocket(rockets_data):
    """
    Display the number of launches per rocket in the specified format.
    """
    for rocket_name, count in rockets_data:
        print(f"{rocket_name}: {count}")


if __name__ == '__main__':
    rockets_data = get_launches_per_rocket()
    display_launches_per_rocket(rockets_data)
