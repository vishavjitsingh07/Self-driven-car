## Self-Driving Car using NEAT and Pygame

![Demo](https://www.youtube.com/watch?v=OEOeT3jXs28)

[Watch the Demo on YouTube](https://www.youtube.com/watch?v=OEOeT3jXs28)

This is a simple self-driving car simulation using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm and Pygame. The car learns to navigate a track using LiDAR sensors to detect the environment and neural networks to make decisions about its direction.

## Prerequisites

Before running this simulation, make sure you have the following dependencies installed:

- Python (3.6 or higher)
- pygame
- neat-python

You can install these dependencies using pip:

```bash
pip install requirements.txt
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/worldwinner-vishav/Self-driven-car.git
```

2. Navigate to the project directory:

```bash
cd Self-driven-car
```

3. Run the simulation:

```bash
python main.py
```

The self-driving car simulation will start, and you can watch the cars learn to navigate the track.

## Configuration

The behavior of the self-driving cars can be configured by modifying the `config.txt` file. You can adjust parameters like population size, mutation rates, and neural network architecture in this file.

## How it Works

- The self-driving cars are controlled by neural networks that take input from LiDAR sensors, which measure the distance to obstacles in different directions.
- The cars start with random neural network weights and attempt to learn the best strategy for navigating the track.
- The NEAT algorithm is used to evolve and improve the neural networks over generations.

## Credits

This project is based on the NEAT Python library and Pygame. The original source code can be found here:

- [NEAT Python](https://neat-python.readthedocs.io/)
- [Pygame](https://www.pygame.org/)

Feel free to reach out if you have any questions or issues. Happy coding!
