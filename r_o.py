from geopy.distance import geodesic
import datetime

# Function to calculate travel time based on distance and average speed
def calculate_travel_time(distance_km, average_speed_kmph):
    # Calculate time in hours (time = distance / speed)
    time_hours = distance_km / average_speed_kmph

    return time_hours

def main():
    # Input latitude and longitude for current location
    current_latitude = float(input("Enter latitude of current location: "))
    current_longitude = float(input("Enter longitude of current location: "))
    current_location = (current_latitude, current_longitude)

    # Input latitude and longitude for destination
    dest_latitude = float(input("Enter latitude of destination: "))
    dest_longitude = float(input("Enter longitude of destination: "))
    destination_location = (dest_latitude, dest_longitude)

    # Input average speed (in km/h)
    average_speed_kmph = float(input("Enter average speed (km/h): "))

    # Calculate geodesic distance between current and destination locations
    distance_km = geodesic(current_location, destination_location).kilometers

    # Calculate travel time
    travel_time = calculate_travel_time(distance_km, average_speed_kmph)

    print(f"Distance between locations: {distance_km:.2f} km")
    print(f"Estimated travel time: {travel_time:.2f} hours")

if __name__ == "__main__":
    main()
