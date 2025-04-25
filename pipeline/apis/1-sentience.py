#!/usr/bin/env python3
"""
star wars swapi project on apis
"""
import requests


def sentientPlanets():
    """
    name of the starships that are large enough
    :param passengerCount: int
        The number of passengers to check for.
    :return: list
        A list of starship names that can carry the given number of passengers.
    """

    url = "https://swapi-api.hbtn.io/api/species/"
    sentients = []
    while url:
        r = requests.get(url, timeout=25)
        data = r.json()
        for species in data["results"]:
            if species['designation'] == 'sentient':
                if species["homeworld"] is None:
                    continue
                world_url = species["homeworld"]
                world_r = requests.get(world_url, timeout=25)
                world_data = world_r.json()
                sentients.append(str(world_data["name"]))
        url = data["next"]
    return sentients
