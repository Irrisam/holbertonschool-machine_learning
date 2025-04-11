#!/usr/bin/env python3
"""
star wars swapi project on apis
"""
import requests
import json


def sentientPlanets():
    """
    name of the starships that are large enough
    :param passengerCount: int
        The number of passengers to check for.
    :return: list
        A list of starship names that can carry the given number of passengers.
    """

    url = "https://swapi.dev/api/species/"
    sentients = []
    while url:
        r = requests.get(url, timeout=5)
        data = r.json()
        for species in data["results"]:
            if species['designation'] == 'sentient':
                sentients.append(species["name"])
        url = data["next"]
    return sentients
