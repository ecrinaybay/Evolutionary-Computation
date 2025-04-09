import numpy as np
import matplotlib.pyplot as plt

# Hedef fonksiyon: Paraboloid
def objective_function(x):
    return np.sum(x**2)

# Parçacık sınıfı
class Particle:
    def __init__(self, dimension, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dimension)  # Başlangıç pozisyonu
        self.velocity = np.random.uniform(-1, 1, dimension)  # Başlangıç hızı
        self.best_position = self.position.copy()  # Bireysel en iyi pozisyon
        self.best_value = objective_function(self.position)  # Bireysel en iyi değer
        self.value = self.best_value

# PSO sınıfı
class PSO:
    def __init__(self, objective_function, dimension, bounds, num_particles, max_iter):
        self.obj_func = objective_function
        self.dimension = dimension
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter

        self.particles = [Particle(dimension, bounds) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.best_values = []  # Her iterasyondaki en iyi değerleri saklamak için

    def optimize(self):
        w = 0.5  # Atalet ağırlığı
        c1 = 2   # Bireysel hızlanma katsayısı
        c2 = 2   # Küresel hızlanma katsayısı

        for t in range(self.max_iter):
            for particle in self.particles:
                # Hedef fonksiyonu değerlendir
                particle.value = self.obj_func(particle.position)

                # Bireysel en iyiyi güncelle
                if particle.value < particle.best_value:
                    particle.best_value = particle.value
                    particle.best_position = particle.position.copy()

                # Küresel en iyiyi güncelle
                if particle.value < self.global_best_value:
                    self.global_best_value = particle.value
                    self.global_best_position = particle.position.copy()

            # Her parçacığın hızını ve pozisyonunu güncelle
            for particle in self.particles:
                r1 = np.random.random(self.dimension)
                r2 = np.random.random(self.dimension)

                cognitive_component = c1 * r1 * (particle.best_position - particle.position)
                social_component = c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = w * particle.velocity + cognitive_component + social_component
                particle.position += particle.velocity

                # Pozisyonları sınırlar içinde tut
                particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])

            # En iyi değeri kaydet
            self.best_values.append(self.global_best_value)

            # Durum raporu
            print(f"Iterasyon {t+1}/{self.max_iter}, En İyi Değer: {self.global_best_value:.4f}")

    def plot_convergence(self):
        # Grafik çizimi
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_values, marker='o', color='b', label='En İyi Değer')
        plt.title("PSO İyileşme Süreci")
        plt.xlabel("İterasyon")
        plt.ylabel("En İyi Değer")
        plt.legend()
        plt.grid()
        plt.show()

# Parametreler
dimension = 2           # Problem boyutu
bounds = [-10, 10]      # Sınırlar
num_particles = 30      # Parçacık sayısı
max_iter = 50           # Maksimum iterasyon

# PSO çalıştır
pso = PSO(objective_function, dimension, bounds, num_particles, max_iter)
pso.optimize()
pso.plot_convergence()

print(f"En İyi Çözüm: {pso.global_best_position}")
print(f"En İyi Değer: {pso.global_best_value}")


