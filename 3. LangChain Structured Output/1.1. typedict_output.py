# TypedDict is a way to define dictionary in python where you specify what keys and values should exist. It helps ensure that your dictionary follows a specific structure.

# Why use TypedDict?
# It tells python what keys and values should be in the dictionary. This helps catch errors early and makes your code easier to understand.
# It does not validate the data at runtime, but it helps during development by providing type hints.

from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

person1: Person = { 'name': 'Saqib', 'age': 20}

print(person1)