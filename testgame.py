import pygame
import random
import math
import numpy as np
import time
import csv
from pygame.locals import *

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
GRAVITY = 0.25
FLAP_STRENGTH = -5
PIPE_WIDTH = 50
PIPE_GAP = 150
PIPE_FREQUENCY = 1500  # milliseconds

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption('Flappy Bird with Neural Network using tanh')
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 20)

class Bird:
    def __init__(self, brain=None):
        self.x = 100
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.radius = 15
        self.fitness = 0
        self.dead = False
        
        if brain:
            self.brain = brain
        else:
            self.input_weights = np.random.randn(3, 6) * 0.5
            self.hidden_weights = np.random.randn(6, 1) * 0.5
            self.hidden_weights[0] = 1.0
    
    def think(self, pipes):
        closest_pipe = None
        for pipe in pipes:
            if pipe.x + PIPE_WIDTH > self.x:
                closest_pipe = pipe
                break
        
        if closest_pipe:
            input1 = (closest_pipe.x - self.x) / SCREEN_WIDTH
            input2 = (self.y - closest_pipe.gap_y) / SCREEN_HEIGHT
            input3 = self.velocity / 10.0
            
            inputs = np.array([input1, input2, input3])
            hidden = np.tanh(np.dot(inputs, self.input_weights))
            output = np.tanh(np.dot(hidden, self.hidden_weights))
            
            if output > 0.5:
                self.flap()
        else:
            if self.velocity > 2:
                self.flap()
    
    def flap(self):
        self.velocity = FLAP_STRENGTH
    
    def update(self):
        if not self.dead:
            self.velocity += GRAVITY
            self.y += self.velocity
            self.fitness += 1
            
            if self.y < 0:
                self.y = 0
                self.velocity = 0
            elif self.y > SCREEN_HEIGHT:
                self.dead = True
    
    def draw(self):
        color = BLUE if not self.dead else RED
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)
    
    def mutate(self):
        input_mutation_rate = 0.2
        hidden_mutation_rate = 0.1
        
        mask = np.random.rand(*self.input_weights.shape) < input_mutation_rate
        self.input_weights += np.random.randn(*self.input_weights.shape) * mask
        
        mask = np.random.rand(*self.hidden_weights.shape) < hidden_mutation_rate
        self.hidden_weights += np.random.randn(*self.hidden_weights.shape) * mask

class Pipe:
    def __init__(self):
        self.x = SCREEN_WIDTH
        self.gap_y = random.randint(150, SCREEN_HEIGHT - 150)
        self.passed = False
    
    def update(self):
        self.x -= 2
        
    def draw(self):
        pygame.draw.rect(screen, GREEN, (self.x, 0, PIPE_WIDTH, self.gap_y - PIPE_GAP // 2))
        pygame.draw.rect(screen, GREEN, (self.x, self.gap_y + PIPE_GAP // 2, PIPE_WIDTH, SCREEN_HEIGHT - (self.gap_y + PIPE_GAP // 2)))
    
    def collide(self, bird):
        if bird.x + bird.radius > self.x and bird.x - bird.radius < self.x + PIPE_WIDTH:
            if bird.y - bird.radius < self.gap_y - PIPE_GAP // 2 or bird.y + bird.radius > self.gap_y + PIPE_GAP // 2:
                return True
        return False

def create_new_generation(previous_birds):
    sorted_birds = sorted(previous_birds, key=lambda x: x.fitness, reverse=True)
    total_fitness = sum(bird.fitness for bird in sorted_birds)
    
    new_birds = []
    population_size = 20
    
    if sorted_birds:
        elite = Bird()
        elite.input_weights = sorted_birds[0].input_weights.copy()
        elite.hidden_weights = sorted_birds[0].hidden_weights.copy()
        new_birds.append(elite)
    
    while len(new_birds) < population_size:
        candidates = random.sample(sorted_birds[:10], min(3, len(sorted_birds)))
        parent1 = max(candidates, key=lambda x: x.fitness)
        
        candidates = random.sample(sorted_birds[:10], min(3, len(sorted_birds)))
        parent2 = max(candidates, key=lambda x: x.fitness)
        
        child = Bird()
        mask = np.random.rand(*child.input_weights.shape) < 0.5
        child.input_weights = parent1.input_weights * mask + parent2.input_weights * (1 - mask)
        
        mask = np.random.rand(*child.hidden_weights.shape) < 0.5
        child.hidden_weights = parent1.hidden_weights * mask + parent2.hidden_weights * (1 - mask)
        
        child.mutate()
        new_birds.append(child)
    
    return new_birds[:population_size]

def main():
    birds = [Bird() for _ in range(20)]
    pipes = []
    last_pipe = pygame.time.get_ticks()
    generation = 1

    # Logging setup
    start_time = time.time()
    last_logged_time = start_time
    csv_data = [("Timestamp", "Generation", "Max Fitness", "Best Fitness")]

    for bird in birds:
        bird.flap()

    running = True
    best_fitness = 0

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if birds and not birds[0].dead:
                        birds[0].flap()

        current_time = pygame.time.get_ticks()
        if current_time - last_pipe > PIPE_FREQUENCY:
            pipes.append(Pipe())
            last_pipe = current_time
        
        for pipe in pipes[:]:
            pipe.update()
            if pipe.x + PIPE_WIDTH < 0:
                pipes.remove(pipe)
        
        all_dead = True
        for bird in birds:
            if not bird.dead:
                all_dead = False
                bird.think(pipes)
                bird.update()
                for pipe in pipes:
                    if pipe.collide(bird):
                        bird.dead = True
                        break
        
        if all_dead:
            birds = create_new_generation(birds)
            pipes = []
            last_pipe = pygame.time.get_ticks()
            generation += 1
            for bird in birds:
                bird.flap()
        
        screen.fill(BLACK)
        for pipe in pipes:
            pipe.draw()
        for bird in birds:
            bird.draw()
        
        alive_count = sum(1 for bird in birds if not bird.dead)
        max_fitness = max(bird.fitness for bird in birds) if birds else 0
        best_fitness = max(best_fitness, max_fitness)
        info_text = f"Generation: {generation} | Alive: {alive_count}/{len(birds)} | Max Fitness: {max_fitness} | Best:{best_fitness} tanh activation function"
        text_surface = font.render(info_text, True, WHITE)
        screen.blit(text_surface, (10, 10))
        pygame.display.update()
        clock.tick(60)

        # Log data every 60 seconds
        current_real_time = time.time()
        if current_real_time - last_logged_time >= 60:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_real_time))
            csv_data.append((timestamp, generation, max_fitness, best_fitness))
            last_logged_time = current_real_time

    # Save CSV when quitting
    with open("flappy_nn_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    pygame.quit()

if __name__ == "__main__":
    main()
