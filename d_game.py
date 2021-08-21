import pygame
import random
import neat
import os
import math


WIN_WIDTH = 800
WIN_HEIGHT = 600

pygame.init()
pygame.font.init()
myfont = pygame.font.SysFont('Helvetica Neue', 30)

win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
color = (227, 227, 227)
win.fill(color)
pygame.display.set_caption("2048")

clock = pygame.time.Clock()

DinoRun1 = pygame.image.load(os.path.join("image", "DinoRun1.png"))
DinoRun2 = pygame.image.load(os.path.join("image", "DinoRun2.png"))
DinoJump = pygame.image.load(os.path.join("image", "DinoJump.png"))
BigCactus1 = pygame.image.load(os.path.join("image", "LargeCactus1.png"))
BigCactus2 = pygame.image.load(os.path.join("image", "LargeCactus2.png"))
BigCactus3 = pygame.image.load(os.path.join("image", "LargeCactus3.png"))
SmallCactus1 = pygame.image.load(os.path.join("image", "SmallCactus1.png"))
SmallCactus2 = pygame.image.load(os.path.join("image", "SmallCactus2.png"))
SmallCactus3 = pygame.image.load(os.path.join("image", "SmallCactus3.png"))
base = pygame.image.load(os.path.join("image", "Track.png"))

Dinoimg = [DinoRun1, DinoRun2, DinoJump]

Cactus_img = [BigCactus1, BigCactus2, BigCactus3,
              SmallCactus1, SmallCactus2, SmallCactus3]


class Dinosaur:

    ANIMATION_TIME = 2

    def __init__(self):
        self.x = 100
        self.y = WIN_HEIGHT-100
        self.width = DinoRun1.get_width()
        self.height = DinoRun1.get_height()
        self.img = Dinoimg[0]
        self.image_count = 0
        self.v = 10
        self.m = 1
        self.isjumping = False

    def jump(self):

        if self.isjumping == False:

            self.isjumping = True

        if self.isjumping:
            F = (1/2)*self.m*(self.v**2)

            self.y -= F

            self.v -= 1

            if self.v < 0:

                self.m = -1

            if self.v <= -11:
                self.isjumping = False

                self.v = 10
                self.m = 1

    def draw(self):

        

        if self.image_count == 0:
            self.img = Dinoimg[1]
            self.image_count += 1
        else:
            self.img = Dinoimg[0]
            self.image_count = 0

        if self.isjumping:
            self.jump()
            self.img = Dinoimg[2]

        win.blit(self.img, (self.x, self.y))

    def get_mask(self):

        return pygame.mask.from_surface(self.img)


class Cactus:

    VEL = 15

    def __init__(self, x):

        self.x = x

        r = random.randint(0, 5)
        self.img = Cactus_img[r]
        self.y = WIN_HEIGHT-self.img.get_height()-5
        self.passed = False

    def move(self):
        self.x -= self.VEL

    def draw(self):

        win.blit(self.img, (self.x, self.y))

    def collide(self, dino):

        dino_mask = dino.get_mask()

        cactusmask = pygame.mask.from_surface(self.img)

        offset = (self.x - dino.x, self.y - round(dino.y))

        c_point = dino_mask.overlap(cactusmask, offset)

        if c_point:
            return True

        return False


class Track:

    VEL = 15
    WIDTH = base.get_width()
    

    def __init__(self):

        self.y = WIN_HEIGHT-base.get_height()
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):

        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self):
        
        win.blit(base, (self.x1, self.y))
        win.blit(base, (self.x2, self.y))        

def distance(pos_a, pos_b):
    dx = pos_a[0]-pos_b[0]
    dy = pos_a[1]-pos_b[1]
    return math.sqrt(dx**2+dy**2)

def draw_everything(dinos,cactuses,track,score, gen):
    win.fill(color)

    for cactus in cactuses:
        cactus.draw()

    track.draw()
    for dino in dinos:
        dino.draw()

    text1 = myfont.render("Score: " + str(score) , True, (83,83,83))
    text2 = myfont.render("Generation: " + str(gen) , True, (83,83,83))
    text3 = myfont.render("Alive: " + str(len(dinos)) , True, (83,83,83))

    win.blit(text1 ,(WIN_WIDTH - text1.get_width() - 10 ,10) )
    win.blit(text2 ,(WIN_WIDTH - text2.get_width() - 10 ,30) )
    win.blit(text3 ,(WIN_WIDTH - text3.get_width() - 10 ,50) )

    pygame.display.update() 


gen = 0
def eval_genomes(genomes, config):

    global gen
    gen += 1
    nets = []
    ge = []
    dinos = []

    for genome_id,genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome,config)
        nets.append(net)
        dinos.append(Dinosaur())
        genome.fitness = 0
        ge.append(genome)
    
    
    cactuses = [Cactus(700)]
    track = Track()
    score = 0

    running = True
    while running:
        clock.tick(30)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()
                break

        cac_ind = 0
        if len(dinos) > 0:
            if len(cactuses) > 1 and dinos[0].x > cactuses[0].x + cactuses[0].img.get_width():
                cac_ind = 1        
        else:
            running = False
            break


        for x, dino in enumerate(dinos):
            
            ge[x].fitness += 0.1

            output = nets[x].activate((dino.y,distance((dino.x,dino.y),(cactuses[cac_ind].x,cactuses[cac_ind].y))))

            if output[0] > 0.5:
                dino.jump()


        add_cactus = False    
        rem = []

        for cactus in cactuses:
            for x, dino in enumerate(dinos):

                if cactus.collide(dino):
                    ge[x].fitness -= 5
                    dinos.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not cactus.passed and cactus.x < dino.x:
                    cactus.passed = True
                    add_cactus = True    


            if cactus.x + cactus.img.get_width() < 0:
                rem.append(cactus)
           
            cactus.move()

        if add_cactus:
            score += 1
            for g in ge:
                g.fitness += 5
            cactuses.append(Cactus(700))

        for r in rem:
            cactuses.remove(r)

        track.move()
        draw_everything(dinos, cactuses, track ,score, gen)


def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    

    
    winner = p.run(eval_genomes, 50)

    
    print('\nBest genome:\n{!s}'.format(winner))        
    
if __name__ == '__main__':
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-dino.txt')
    run(config_path)
