import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.animation as animation

G = 6.67430e-11  
c = 3e8  

def simulate_space_time(grid_size, mass, position):
    """Simulates space-time curvature using simplified equations."""
    x, y = np.meshgrid(np.linspace(-10, 10, grid_size), np.linspace(-10, 10, grid_size))
    r = np.sqrt((x - position[0])**2 + (y - position[1])**2)
    curvature = -G * mass / (r * c**2 + 1e-5)
    return curvature

def generate_dataset(samples, grid_size):
    """Generates a dataset for training the AI model."""
    data = []
    labels = []
    for _ in range(samples):
        mass = np.random.uniform(1e28, 1e32)  
        position = (np.random.uniform(-10, 10), np.random.uniform(-10, 10)) 
        curvature = simulate_space_time(grid_size, mass, position).flatten()  
        data.append([mass, *position])
        labels.append(curvature)
    return np.array(data), np.array(labels)

samples = 1000
grid_size = 20


data, labels = generate_dataset(samples, grid_size)

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

def build_model(grid_size):
    """Creates a neural network for predicting space-time curvature."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(grid_size**2) 
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model(grid_size)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

def gravitational_wave_simulation(grid_size, time_steps):
    """Simulates gravitational waves from two orbiting masses."""
    x, y = np.meshgrid(np.linspace(-10, 10, grid_size), np.linspace(-10, 10, grid_size))
    r1 = np.sqrt((x + 5)**2 + y**2)
    r2 = np.sqrt((x - 5)**2 + y**2)

    wave_frames = []
    for t in range(time_steps):
        wave1 = np.sin(2 * np.pi * r1 - 0.1 * t)
        wave2 = np.sin(2 * np.pi * r2 - 0.1 * t)
        wave_frames.append(wave1 + wave2)
    return wave_frames

time_steps = 100
frames = gravitational_wave_simulation(grid_size=100, time_steps=time_steps)

def plot_gravitational_wave_animation(frames):
    """Plots an animation of gravitational waves."""
    fig, ax = plt.subplots()
    wave_plot = ax.imshow(frames[0], extent=(-10, 10, -10, 10), cmap='coolwarm', animated=True)

    def update(frame):
        wave_plot.set_data(frame)
        return wave_plot,

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    plt.show()

plot_gravitational_wave_animation(frames)

def predict_and_plot_curvature(model, scaler, grid_size, mass, position):
    """Predicts and plots space-time curvature for a given mass and position."""
    sample_data = scaler.transform([[mass, *position]])
    predicted_curvature = model.predict(sample_data).reshape((grid_size, grid_size))

    plt.figure(figsize=(8, 6))
    plt.imshow(predicted_curvature, extent=(-10, 10, -10, 10), cmap='plasma')
    plt.colorbar(label="Predicted Space-Time Curvature")
    plt.title("Predicted Space-Time Curvature")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

sample_mass = 1e30
sample_position = (5, -5)
predict_and_plot_curvature(model, scaler, grid_size, sample_mass, sample_position)



