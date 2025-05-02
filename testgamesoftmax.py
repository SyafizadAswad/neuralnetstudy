import pygame
import random
import math
import numpy as np
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
RED = (255, 0, 0, 20)
BLUE = (0, 0, 255)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption('Flappy Bird with Neural Network')
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 20)

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

class Bird:
    def __init__(self, brain=None):
        self.x = 100
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.radius = 15
        self.fitness = 0
        self.dead = False
        
        # Neural network (3 inputs, 6 hidden, 2 outputs)
        if brain:
            self.brain = brain
        else:
            # Input weights (3 inputs to 6 hidden nodes)
            self.input_weights = np.random.randn(3, 6) * 0.5  # Smaller initial weights
            # Hidden layer weights (6 hidden to 2 outputs)
            self.hidden_weights = np.random.randn(6, 2) * 0.5
            # Bias to make initial flapping more likely
            self.hidden_weights[0] = 1.0  # Encourage initial flapping

    def think(self, pipes):
        # Find the closest pipe
        closest_pipe = None
        for pipe in pipes:
            if pipe.x + PIPE_WIDTH > self.x:
                closest_pipe = pipe
                break
        
        if closest_pipe:
            # Inputs: 
            # 1. horizontal distance to pipe (normalized)
            # 2. vertical distance to gap center (normalized)
            # 3. bird's current velocity
            input1 = (closest_pipe.x - self.x) / SCREEN_WIDTH
            input2 = (self.y - closest_pipe.gap_y) / SCREEN_HEIGHT
            input3 = self.velocity / 10.0  # Normalized velocity
            
            inputs = np.array([input1, input2, input3])
            
            # Forward propagation with tanh for hidden, and softmax for output
            hidden = np.tanh(np.dot(inputs, self.input_weights))
            output = softmax(np.dot(hidden, self.hidden_weights))

            # Flap if the 'flap' output is greater than the 'no flap' output
            if output[1] > output[0]:
                self.flap()
        else:
            # If no pipes, flap when falling too fast
            if self.velocity > 2:
                self.flap()
    
    def flap(self):
        self.velocity = FLAP_STRENGTH
    
    def update(self):
        if not self.dead:
            self.velocity += GRAVITY
            self.y += self.velocity
            self.fitness += 1
            
            # Check boundaries
            if self.y < 0:
                self.y = 0
                self.velocity = 0
            elif self.y > SCREEN_HEIGHT:
                self.dead = True
    
    def draw(self):
        color = BLUE if not self.dead else RED
        if self.dead: #make red translucent
            alpha = 128  # Adjust alpha (0-255, where 0 is fully transparent, 255 is fully opaque)
            surface = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA) #create surface with alpha
            pygame.draw.circle(surface, (color[0], color[1], color[2], alpha), (self.radius, self.radius), self.radius)
            screen.blit(surface, (int(self.x) - self.radius, int(self.y) - self.radius))
        else: #normal blue bird.
            pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)

    def mutate(self):
        # Mutate the neural network weights with different rates
        input_mutation_rate = 0.2
        hidden_mutation_rate = 0.1
        
        # Mutate with chance to keep some weights unchanged
        mask = np.random.rand(*self.input_weights.shape) < input_mutation_rate
        self.input_weights += np.random.randn(*self.input_weights.shape) * mask
        
        mask = np.random.rand(*self.hidden_weights.shape) < hidden_mutation_rate
        self.hidden_weights += np.random.randn(*self.hidden_weights.shape) * mask

class Pipe:
    def __init__(self):
        self.x = SCREEN_WIDTH
        self.gap_y = random.randint(150, SCREEN_HEIGHT - 150)  # More centered gaps
        self.passed = False
    
    def update(self):
        self.x -= 2
        
    def draw(self):
        # Top pipe
        pygame.draw.rect(screen, GREEN, (self.x, 0, PIPE_WIDTH, self.gap_y - PIPE_GAP // 2))
        # Bottom pipe
        pygame.draw.rect(screen, GREEN, (self.x, self.gap_y + PIPE_GAP // 2, PIPE_WIDTH, SCREEN_HEIGHT - (self.gap_y + PIPE_GAP // 2)))
    
    def collide(self, bird):
        if bird.x + bird.radius > self.x and bird.x - bird.radius < self.x + PIPE_WIDTH:
            if bird.y - bird.radius < self.gap_y - PIPE_GAP // 2 or bird.y + bird.radius > self.gap_y + PIPE_GAP // 2:
                return True
        return False

def create_new_generation(previous_birds):
    # Sort birds by fitness
    sorted_birds = sorted(previous_birds, key=lambda x: x.fitness, reverse=True)
    
    # Calculate total fitness for selection probabilities
    total_fitness = sum(bird.fitness for bird in sorted_birds)
    
    new_birds = []
    population_size = 20
    
    # Always keep the best performer unchanged (elitism)
    if sorted_birds:
        elite = Bird()
        elite.input_weights = sorted_birds[0].input_weights.copy()
        elite.hidden_weights = sorted_birds[0].hidden_weights.copy()
        new_birds.append(elite)
    
    # Create offspring through selection and mutation
    while len(new_birds) < population_size:
        # Select parents with probability proportional to fitness
        parent1 = None
        parent2 = None
        
        # Tournament selection
        candidates = random.sample(sorted_birds[:10], min(3, len(sorted_birds)))
        parent1 = max(candidates, key=lambda x: x.fitness)
        
        candidates = random.sample(sorted_birds[:10], min(3, len(sorted_birds)))
        parent2 = max(candidates, key=lambda x: x.fitness)
        
        # Create child with crossover
        child = Bird()
        
        # 50% chance to get weights from either parent for each weight
        mask = np.random.rand(*child.input_weights.shape) < 0.5
        child.input_weights = parent1.input_weights * mask + parent2.input_weights * (1 - mask)
        
        mask = np.random.rand(*child.hidden_weights.shape) < 0.5
        child.hidden_weights = parent1.hidden_weights * mask + parent2.hidden_weights * (1 - mask)
        
        # Mutate the child
        child.mutate()
        new_birds.append(child)
    
    return new_birds[:population_size]  # Ensure we don't exceed population size

def main():
    birds = [Bird() for _ in range(20)]  # Initial population
    pipes = []
    last_pipe = pygame.time.get_ticks()
    generation = 1
    
    # Make first bird flap immediately
    for bird in birds:
        bird.flap()
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    # Manual flap for testing
                    if birds and not birds[0].dead:
                        birds[0].flap()
        
        # Add new pipes
        current_time = pygame.time.get_ticks()
        if current_time - last_pipe > PIPE_FREQUENCY:
            pipes.append(Pipe())
            last_pipe = current_time
        
        # Update pipes
        for pipe in pipes[:]:
            pipe.update()
            if pipe.x + PIPE_WIDTH < 0:
                pipes.remove(pipe)
        
        # Update birds
        all_dead = True
        for bird in birds:
            if not bird.dead:
                all_dead = False
                bird.think(pipes)
                bird.update()
                
                # Check for collisions
                for pipe in pipes:
                    if pipe.collide(bird):
                        bird.dead = True
                        break
        
        # If all birds are dead, create new generation
        if all_dead:
            birds = create_new_generation(birds)
            pipes = []
            last_pipe = pygame.time.get_ticks()
            generation += 1
            
            # Make new birds flap immediately
            for bird in birds:
                bird.flap()
        
        # Draw everything
        screen.fill(BLACK)
        
        for pipe in pipes:
            pipe.draw()
        
        for bird in birds:
            bird.draw()
        
        # Display info
        alive_count = sum(1 for bird in birds if not bird.dead)
        max_fitness = max(bird.fitness for bird in birds) if birds else 0
        if 'best_fitness' not in locals():
            best_fitness = 0

        best_fitness = max(best_fitness, max_fitness)
        info_text = f"Generation: {generation} | Alive: {alive_count}/{len(birds)} | Max Fitness: {max_fitness} | Best:{best_fitness}"
        text_surface = font.render(info_text, True, WHITE)
        screen.blit(text_surface, (10, 10))
        
        pygame.display.update()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()