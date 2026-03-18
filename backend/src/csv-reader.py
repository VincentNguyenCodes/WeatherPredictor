import csv


def read_temperature_max(filename, year, day_of_year):
    with open(filename, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["year"]) == year and int(row["day_of_year"]) == day_of_year:
                return float(row["tmax"]) if row["tmax"] != "" else None
    return None


def read_temperature_min(filename, year, day_of_year):
    with open(filename, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["year"]) == year and int(row["day_of_year"]) == day_of_year:
                return float(row["tmin"]) if row["tmin"] != "" else None
    return None


def read_temperature_precipitation(filename, year, day_of_year):
    with open(filename, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["year"]) == year and int(row["day_of_year"]) == day_of_year:
                return float(row["precip"]) if row["precip"] != "" else None
    return None
