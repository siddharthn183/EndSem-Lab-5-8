import numpy as np
import matplotlib.pyplot as plt

cities = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}


np.random.seed(42)  # for reproducibility
city_coordinates = {city: np.random.rand(2) * 50 for city in cities}

def calculate_distances(cities):
    num_cities = len(cities)
    distances = np.zeros((num_cities, num_cities))
    for i, city_i in enumerate(cities):
        for j, city_j in enumerate(cities):
            if i != j:
                distances[i][j] = np.sqrt(np.sum((city_coordinates[city_i] - city_coordinates[city_j]) ** 2))
    return distances

class HopfieldTSP:
    def __init__(self, cities, distances):
        self.cities = cities
        self.num_cities = len(cities)
        self.distances = distances
        self.weights = np.zeros((self.num_cities, self.num_cities))
        self.initialize_weights()

    def initialize_weights(self):
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    self.weights[i][j] = -self.distances[i][j]

    def energy(self, tour):
        total_distance = 0
        for i in range(self.num_cities - 1):
            total_distance += self.distances[tour[i]][tour[i+1]]
        total_distance += self.distances[tour[-1]][tour[0]]  # Return to the starting city
        return total_distance

    def solve(self, iterations=100):
        tour = np.random.permutation(self.num_cities)
        for _ in range(iterations):
            i, j = np.random.choice(self.num_cities, 2, replace=False)
            delta_energy = 2 * (
                self.weights[i].dot(tour) * tour[i] +
                self.weights[j].dot(tour) * tour[j]
            )
            if delta_energy < 0:
                tour[i], tour[j] = tour[j], tour[i]
        return tour

distances = calculate_distances(cities)

hopfield_tsp = HopfieldTSP(list(cities), distances)

solution_tour = hopfield_tsp.solve()

print("Optimal Tour:")
for i, city_index in enumerate(solution_tour):
    if i == 0:
        print(f"Start from {list(cities)[city_index]}")
    else:
        print(f"Go to {list(cities)[city_index]}")
print(f"Return to {list(cities)[solution_tour[0]]}")


total_distance = hopfield_tsp.energy(solution_tour)


print("Total Distance Traveled:", total_distance)

plt.figure(figsize=(8, 6))
for city in cities:
    plt.scatter(city_coordinates[city][0], city_coordinates[city][1], color='blue')
    plt.text(city_coordinates[city][0], city_coordinates[city][1], city, ha='center', va='center', fontsize=12)
for i in range(len(solution_tour) - 1):
    plt.plot([city_coordinates[list(cities)[solution_tour[i]]][0], city_coordinates[list(cities)[solution_tour[i+1]]][0]],
             [city_coordinates[list(cities)[solution_tour[i]]][1], city_coordinates[list(cities)[solution_tour[i+1]]][1]],
             color='red')
plt.plot([city_coordinates[list(cities)[solution_tour[-1]]][0], city_coordinates[list(cities)[solution_tour[0]]][0]],
         [city_coordinates[list(cities)[solution_tour[-1]]][1], city_coordinates[list(cities)[solution_tour[0]]][1]],
         color='red')
plt.title("Traveling Salesman Problem Solution")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.show()
