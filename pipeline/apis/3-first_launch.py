#!/usr/bin/env python3
"""
First launch
"""
import requests


def get_json(url):
    '''
    Get JSON data from a URL'''
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print("Request failed: {}".format(e))
        return None


def get_up_launch():
    ''' Get the next launch from SpaceX API'''
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    launches = get_json(url)
    if launches is None:
        return None
    dates = []

    next_launch = min(launches, key=lambda launch: launch['date_unix'])
    return next_launch


def get_name_rocket(rocket_id):
    ''' Get the name of the rocket from SpaceX API'''
    url = "https://api.spacexdata.com/v4/rockets/{}".format(rocket_id)
    rocket_info = get_json(url)
    if rocket_info is None:
        return None

    return rocket_info['name']


def get_launchpad_info(launchpad_id):
    ''' Get the name and location of the launchpad from SpaceX API'''
    url = "https://api.spacexdata.com/v4/launchpads/{}".format(launchpad_id)
    launchpad_info = get_json(url)
    if launchpad_info is None:
        return None, None

    return launchpad_info['name'], launchpad_info['locality']


if __name__ == "__main__":
    next_launch = get_up_launch()
    if next_launch is None:
        print("Failed to retrieve the next launch.")
        exit(1)

    name = next_launch['name']
    date = next_launch['date_local']

    rocket_name = get_name_rocket(next_launch['rocket'])
    if rocket_name is None:
        print("Failed to retrieve rocket information.")
        exit(1)

    launchpad_name, launchpad_loc = (
        get_launchpad_info(next_launch['launchpad']))
    if launchpad_name is None or launchpad_loc is None:
        print("Failed to retrieve launchpad information.")
        exit(1)

    print("{} ({}) {} - {} ({})".format(name,
                                        date,
                                        rocket_name,
                                        launchpad_name,
                                        launchpad_loc))
