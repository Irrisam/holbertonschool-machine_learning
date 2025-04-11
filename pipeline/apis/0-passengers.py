#!/usr/bin/env python3
"""
star wars swapi project on apis
"""
import requests


def availableShips(passengerCount):
    """
    name of the starships that are large enough
    :param passengerCount: int
        The number of passengers to check for.
    :return: list
        A list of starship names that can carry the given number of passengers.
    """

    url = "https://swapi.dev/api/starships/"
    ships = []
    passengerCount = float(passengerCount)
    while url:
        r = requests.get(url, timeout=5)
        data = r.json()
        for ship in data["results"]:
            passengers = ship["passengers"].replace(',', '')
            if passengers.isdigit():
                if int(passengers) >= passengerCount:
                    ships.append(ship["name"])
        url = data["next"]
    return ships
