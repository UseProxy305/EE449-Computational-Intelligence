import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from math import floor
from copy import deepcopy
import time


#consider 3 cases for both x and y
#this function returns True if the circle is in the image
#returns False otherwise
def BoundaryCheck(x, y, radi):
	#x is on left of frame
	if (x < 0):
		#y is on bottom of frame
		if (y < 0):
			if (x + radi >= 0 and y + radi >= 0):
				return True
		#y is on the frame
		elif (0 <= y <= 180):
			if (x + radi >= 0):
				return True
		#y is on top of frame
		else:
			if (x + radi >= 0 and y - radi <= 180):
				return True
	#x is on frame			
	elif (0 <= x <= 180):
		#y is on bottom of frame
		if (y < 0):
			if (y + radi >= 0):
				return True
		#y is on the frame
		elif (0 <= y <= 180):
				return True
		#y is on top of frame
		else:
			if (y - radi <= 180):
				return True
	#x is on right of frame
	else:
		#y is on bottom of frame
		if (y < 0):
			if (x - radi >= 180 and y + radi >= 0):
				return True
		#y is on the frame
		elif (0 <= y <= 180):
			if (x - radi >= 180):
				return True
		#y is on top of frame
		else:
			if (x - radi >= 180 and y - radi <= 180):
				return True
	return False


class Gene:
	#constructor
	def	__init__(self, x=0, y=0, radi=0, BLUE=0, GREEN=0, RED=0, ALPHA=0):
		#x and y = ±10 + width,heigth
		self.x 		= random.randint(-10,width+10)
		self.y 		= random.randint(-10,height+10)
		#radi is in between 1 and 30
		self.radi 	= random.randint(1,30)
		# if the circle is not in the image, reinitialize the coordinates and the radius
		while(not BoundaryCheck(self.x,self.y,self.radi)):
			self.x 		= random.randint(-10,width+10)
			self.y 		= random.randint(-10,height+10)
			self.radi 	= random.randint(1,30)
		#random number between 0,255 for the colors
		self.BLUE	= random.randint(0,255)
		self.GREEN	= random.randint(0,255)
		self.RED	= random.randint(0,255)
		#random number between 0 and 1 for the alpha value
		self.ALPHA	= random.random()
		
	def mutateGene(self, mutation_type="guided"):
		#initial values for coordinates will be needed later
		orig_x = self.x
		orig_y = self.y
		#if guided
		if (mutation_type == "guided"):
			#implement what is in the homework manual
			self.x 		= random.randint(orig_x - 45, orig_x + 45)
			self.y 		= random.randint(orig_y - 45, orig_y + 45)
			#if the radius is below 10, select a new radius between 1 and 10
			if (self.radi < 10):
				self.radi 	= random.randint(1, self.radi + 10)
			#if the radius is above 10, select a new radius between ±10 + radius
			else:
				self.radi 	= random.randint(self.radi - 10, self.radi + 10)
			#if the circle is not in the image, reinitialize the coordinates and the radius
			while(not BoundaryCheck(self.x,self.y,self.radi)):
				self.x 		= random.randint(orig_x - 45, orig_x + 45)
				self.y 		= random.randint(orig_y - 45, orig_y + 45)	
				if (self.radi < 10):
					self.radi 	= random.randint(1, self.radi + 10)
				else:
					self.radi 	= random.randint(self.radi - 10, self.radi + 10)
			#if the any of the color is below 64, select between 0, color + 64
			#if the any of the color is above 191, select between color - 64, 255
			#else select between ±64 + color
			if (self.BLUE - 64 < 0):
				self.BLUE = random.randint(0, self.BLUE + 64)
			elif (self.BLUE + 64 > 255):
				self.BLUE = random.randint(self.BLUE - 64, 255)
			else:
				self.BLUE	= random.randint(self.BLUE - 64, self.BLUE + 64)
			if (self.GREEN - 64 < 0):
				self.GREEN = random.randint(0, self.GREEN + 64)
			elif (self.GREEN + 64 > 255):
				self.GREEN = random.randint(self.GREEN - 64, 255)
			else:
				self.GREEN	= random.randint(self.GREEN - 64, self.GREEN + 64)
			if (self.RED - 64 < 0):
				self.RED = random.randint(0, self.RED + 64)
			elif (self.RED + 64 > 255):
				self.RED = random.randint(self.RED - 64, 255)
			else:
				self.RED = random.randint(self.RED - 64, self.RED + 64)
			#same story applies also for the alpha
			if (self.ALPHA - 0.25 < 0):
				self.ALPHA = random.uniform(0, self.ALPHA + 0.25)
			elif (self.ALPHA + 0.25 > 1):
				self.ALPHA = random.uniform(self.ALPHA - 0.25, 1)
			else:
				self.ALPHA	= random.uniform(self.ALPHA - 0.25, self.ALPHA + 0.25)
		#else if unguided
		elif (mutation_type == "unguided"):
			self.x 		= random.randint(-10,width+10)
			self.y 		= random.randint(-10,height+10)
			self.radi 	= random.randint(1,30)
			while(not BoundaryCheck(self.x,self.y,self.radi)):
				self.x 		= random.randint(-10,width+10)
				self.y 		= random.randint(-10,height+10)
				self.radi 	= random.randint(1,30)
			self.BLUE	= random.randint(0,255)
			self.GREEN	= random.randint(0,255)
			self.RED	= random.randint(0,255)
			self.ALPHA	= random.random()

class Individual:
	#constructor
	def	__init__(self, num_genes=50):
		self.fitness = None
		self.num_genes = num_genes
		#initialize #num_genes genes
		self.chromosome = [Gene() for i in range(0,num_genes)]
		
	#while random number is below mutation_prob, mutate a random gene
	def mutateIndividual(self, mutation_prob = 0.2, mutation_type = "guided"):
		while (random.random() < mutation_prob):
			randomGene = random.randint(0, self.num_genes - 1)
			self.chromosome[randomGene].mutateGene(mutation_type)
			self.fitness = None
	
	#evaluate an individual as described in the homework manual
	def evalIndividual(self, painting):		
		img = np.full((180, 180, 3), (255, 255, 255), np.uint8)
		for gene in self.chromosome:
			overlay = deepcopy(img)
			cv2.circle(overlay, (gene.x, gene.y), gene.radi, (gene.BLUE, gene.GREEN, gene.RED), thickness = -1)
			cv2.addWeighted(overlay, gene.ALPHA, img, 1 - gene.ALPHA, 0.0, img)
		"""
		fitness = 0
		for i in range(0,3):
			for j in range(0,180):
				for k in range(0,180):
					fitness = fitness + (int(painting[j][k][i]) - int(img[j][k][i])) * (int(painting[j][k][i]) - int(img[j][k][i]))
		"""
		#convert to integer to avoid overflow
		self.fitness = -np.sum(np.square(np.subtract(painting.astype(int), img.astype(int))))
		
	#this method sorts genes according to their radiuses for the image plotting
	def sortGenes(self):
		self.chromosome.sort(key = lambda x: x.radi, reverse=True)
		
	#this method returns the image
	def bestImage(self):
		img = np.full((180, 180, 3), (255, 255, 255), np.uint8)
		self.sortGenes()
		for gene in self.chromosome:
			overlay = deepcopy(img)
			cv2.circle(overlay, (gene.x, gene.y), gene.radi, (gene.BLUE, gene.GREEN, gene.RED), thickness = -1)
			cv2.addWeighted(overlay, gene.ALPHA, img, 1 - gene.ALPHA, 0.0, img)
		return img
			

class Population:
	#constructor
	def	__init__(self, num_inds = 20, num_genes = 50, tm_size = 5, frac_elites = 0.2, frac_parents = 0.6, mutation_prob = 0.2, mutation_type = "guided"):
		self.num_inds = num_inds
		self.num_genes = num_genes
		self.tm_size = tm_size
		self.frac_elites = frac_elites
		self.frac_parents = frac_parents
		self.mutation_prob = mutation_prob
		self.mutation_type = mutation_type
		#initialize #num_inds individuals
		self.individuals = [Individual(num_genes=num_genes) for i in range(0, num_inds)]
		#list for storing elites
		self.elites = []
		#list for storing winners
		self.winners = []
		#list for storing children
		self.children = []
		#list for storing every individual except elites
		self.excludeElites = []
	
	#this method is used for evaluation of the population	
	def evalPopulation(self, painting):
		for ind in self.individuals:
			ind.evalIndividual(painting)
			#print(ind.fitness)
			
	def selection(self):
		#clear the previous generation winners and elites first
		self.winners.clear()
		self.elites.clear()
		#sort individuals according to their fitness values
		self.individuals = sorted(self.individuals, key = lambda x: x.fitness, reverse = True)
		#find the elites
		for i in range(floor(self.num_inds * self.frac_elites)):
			temp = deepcopy(self.individuals[i])
			self.elites.append(temp)
			#if elites are not going to be included in the tournaments enable the below line
			#self.elites.append(self.individuals.pop(0))

		#apply tournament and select winners
		for i in range(floor(self.num_inds * self.frac_parents)):
			self.winners.append(self.pop_tmsel())
		#remove elites from the individuals list
		for i in range(len(self.elites)):
			self.individuals.pop(len(self.individuals) - 1)

		#returns the best individual
		return self.elites[0]
		
	#this method is the same as in the lecture notes
	def pop_tmsel(self):
		#select random individual
		index = random.randrange(0, len(self.individuals))
		best = self.individuals[index]
		x = self.tm_size
		#tournament until tournament size is reached
		while (x > 0):
			#decrement the tournament size
			x = x - 1
			#select a random individual
			temp_index = random.randrange(0, len(self.individuals))
			ind = self.individuals[temp_index]
			#if the new individual beats the old individual, make it winner
			if (ind.fitness > best.fitness):
				best = ind
				index = temp_index
		#remove the winner from the individuals list
		self.individuals.pop(index)
		#return it
		return best
	
	def crossover(self):
		#first clear the previos generation lists
		self.children.clear()
		self.excludeElites.clear()
		#make crossovers until every individual is done
		while(len(self.winners)):
			#if there are only 1 individual is remaining, let it free without crossover
			if (len(self.winners) == 1):
				self.children.append(self.winners.pop(0))
			#if there are at least 2 indiivduals remaining
			else:
				#randomly choose two individuals
				parent1 = self.winners.pop(random.randrange(0, len(self.winners)))
				parent2 = self.winners.pop(random.randrange(0, len(self.winners)))
				#create two random children
				children1 = Individual(self.num_genes)
				children2 = Individual(self.num_genes)
				#crossover
				for j in range(0, self.num_genes):
					if (random.random() < 0.5):
						children1.chromosome[j] = parent1.chromosome[j]
						children2.chromosome[j] = parent2.chromosome[j]
					else:
						children1.chromosome[j] = parent2.chromosome[j]
						children2.chromosome[j] = parent1.chromosome[j]
				#append these children to the children list
				self.children.append(children1)
				self.children.append(children2)
		#determine the excludeElites list
		self.excludeElites = self.individuals + self.children
		
	#this method is used to mutate the population
	def mutatePopulation(self):
		#every individual except elites will be mutated
		for ind in self.excludeElites:
			ind.mutateIndividual(mutation_prob = self.mutation_prob, mutation_type = self.mutation_type)
		#update individuals
		temp = self.elites + self.excludeElites
		self.individuals = deepcopy(temp)
		
	#this method is used to sort every individual according to their fitness value
	def sortPopulation(self):
		self.individuals.chromosome = sorted(self.individuals, key = lambda x: x.fitness, reverse = True)
	
	#this method is used to sort every individual according to their radius value
	def sortPopRadi(self):
		for ind in self.individuals:
			ind.chromosome.sort(key=lambda x: x.radi, reverse=True)
			

#this is the beginning of the main part

#read the painting
painting = cv2.imread("inputs/example.png") #180x180
#extract the height, width, channels informations
height, width, channels = painting.shape
#determine number of generations, I choose 10k
num_generations = 10000
#this list is used to store fitness values of a single generation
all_fitness = []
num_gen_inc = num_generations + 1
x = 1

#loop until every case is covered
while (x != 21):
	temp_generations = 1
	prev_fitness = -9999999999
	all_fitness.clear()
	#num_inds = 5
	if (x == 0):
		num_inds 		= 5
		num_genes 		= 50
		tm_size			= 5
		frac_elites		= 0.2
		frac_parents	= 0.6
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "num_inds5"
	#num_inds = 10
	elif (x == 1):
		num_inds 		= 10
		num_genes 		= 50
		tm_size			= 5
		frac_elites		= 0.2
		frac_parents	= 0.6
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "num_inds10"
	#num_inds = 20
	elif (x == 2):
		num_inds 		= 20
		num_genes 		= 50
		tm_size			= 5
		frac_elites		= 0.2
		frac_parents	= 0.6
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "num_inds20"
	#num_inds = 50
	elif (x == 3):
		num_inds 		= 50
		num_genes 		= 50
		tm_size			= 5
		frac_elites		= 0.2
		frac_parents	= 0.6
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "num_inds50"
	#num_inds = 75
	elif (x == 4):
		num_inds 		= 75
		num_genes 		= 50
		tm_size			= 5
		frac_elites		= 0.2
		frac_parents	= 0.6
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "num_inds75"
	#num_genes = 10
	elif (x == 5):
		num_inds 		= 20
		num_genes 		= 10
		tm_size			= 5
		frac_elites		= 0.2
		frac_parents	= 0.6
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "num_genes10"
	#num_genes = 25
	elif (x == 6):
		num_inds 		= 20
		num_genes 		= 25
		tm_size			= 5
		frac_elites		= 0.2
		frac_parents	= 0.6
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "num_genes25"
	#num_genes = 100
	elif (x == 7):
		num_inds 		= 20
		num_genes 		= 100
		tm_size			= 5
		frac_elites		= 0.2
		frac_parents	= 0.6
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "num_genes100"
	#num_genes = 150
	elif (x == 8):
		num_inds 		= 20
		num_genes 		= 150
		tm_size			= 5
		frac_elites		= 0.2
		frac_parents	= 0.6
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "num_genes150"
	#tm_size = 2
	elif (x == 9):
		num_inds 		= 20
		num_genes 		= 50
		tm_size			= 2
		frac_elites		= 0.2
		frac_parents	= 0.6
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "tm_size2"
	#tm_size = 10
	elif (x == 10):
		num_inds 		= 20
		num_genes 		= 50
		tm_size			= 10
		frac_elites		= 0.2
		frac_parents	= 0.6
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "tm_size10"
	#tm_size = 20
	elif (x == 11):
		num_inds 		= 20
		num_genes 		= 50
		tm_size			= 20
		frac_elites		= 0.2
		frac_parents	= 0.6
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "tm_size20"
	#frac_elites = 0.05
	elif (x == 12):
		num_inds 		= 20
		num_genes 		= 50
		tm_size			= 5
		frac_elites		= 0.05
		frac_parents	= 0.6
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "frac_elites0.05"
	#frac_elites = 0.4
	elif (x == 13):
		num_inds 		= 20
		num_genes 		= 50
		tm_size			= 5
		frac_elites		= 0.4
		frac_parents	= 0.6
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "frac_elites0.4"
	#frac_parents = 0.2
	elif (x == 14):
		num_inds 		= 20
		num_genes 		= 50
		tm_size			= 5
		frac_elites		= 0.2
		frac_parents	= 0.2
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "frac_parents0.2"
	#frac_parents = 0.4
	elif (x == 15):
		num_inds 		= 20
		num_genes 		= 50
		tm_size			= 5
		frac_elites		= 0.2
		frac_parents	= 0.4
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "frac_parents0.4"
	#frac_parents = 0.8
	elif (x == 16):
		num_inds 		= 20
		num_genes 		= 50
		tm_size			= 5
		frac_elites		= 0.2
		frac_parents	= 0.8
		mutation_prob	= 0.2
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "frac_parents0.8"
	#mutation_prob = 0.1
	elif (x == 17):
		num_inds 		= 20
		num_genes 		= 50
		tm_size			= 5
		frac_elites		= 0.2
		frac_parents	= 0.6
		mutation_prob	= 0.1
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "mutation_prob0.1"
	#mutation_prob = 0.5
	elif (x == 18):
		num_inds 		= 20
		num_genes 		= 50
		tm_size			= 5
		frac_elites		= 0.2
		frac_parents	= 0.6
		mutation_prob	= 0.5
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "mutation_prob0.5"
	#mutation_prob = 0.8
	elif (x == 19):
		num_inds 		= 20
		num_genes 		= 50
		tm_size			= 5
		frac_elites		= 0.2
		frac_parents	= 0.6
		mutation_prob	= 0.8
		mutation_type	= "guided"
		num_generations = 10000
		dir_name = "mutation_prob0.8"
	#mutation_type = unguided
	elif (x == 20):
		num_inds 		= 20
		num_genes 		= 50
		tm_size			= 5
		frac_elites		= 0.2
		frac_parents	= 0.6
		mutation_prob	= 0.2
		mutation_type	= "unguided"
		num_generations = 10000
		dir_name = "unguided"

	print(dir_name)
	#create population
	pop = Population(num_inds=num_inds, num_genes=num_genes, tm_size=tm_size, frac_elites=frac_elites, frac_parents=frac_parents, mutation_prob=mutation_prob, mutation_type=mutation_type)
	
	while (temp_generations != num_gen_inc):

		#sort the population according to their radiuses
		pop.sortPopRadi()
		#evaluate the population
		pop.evalPopulation(painting)


		#select winners and elites
		best = pop.selection()
		#print current number of generation and the best fitness value in this generation
		print(temp_generations, best.fitness)
		#if the best value is lower than the previous generation best value, it means an error
		if (prev_fitness > best.fitness):
			print("ERROR")
			break
		#append the best fitness value of this generation to fitness list
		all_fitness = all_fitness + [best.fitness]
		#make the new fitness the previous, for the next loop
		prev_best = best.fitness

		#apply crossover
		pop.crossover()
		#mutate the population except elites
		pop.mutatePopulation()

		#if the current number of generations is the a multiple of 1000, plot the image of the best individual, and save it
		if (temp_generations % 1000 == 0):
			cv2.imwrite("Images/" + dir_name + "/GN" + str(temp_generations) + "_NI" + str(num_inds) + "_NG" + str(num_genes) + 	"_TS" + str(tm_size) + "_FE" + str(frac_elites) + "_FP" + str(frac_parents) + "_MP" + str(mutation_prob) + "_" + mutation_type + ".png", best.bestImage())
		
		#increment the current generation number
		temp_generations = temp_generations + 1
	
	
	#plot fitness figure from gen. 1 to gen. 10000
	plt.figure()
	plt.plot(all_fitness)
	plt.savefig("Fitness/" + dir_name + "/1_to_10000" + str(temp_generations) + "_NI" + str(num_inds) + "_NG" + str(num_genes) + "_TS" + str(tm_size) + "_FE" + str(frac_elites) + "_FP" + str(frac_parents) + "_MP" + str(mutation_prob) + "_" + mutation_type + ".png")

	#plot fitness figure from gen. 1000 to gen. 10000
	plt.figure()
	plt.plot(all_fitness[999:])
	plt.savefig("Fitness/" + dir_name + "/1000_to_10000" + str(temp_generations) + "_NI" + str(num_inds) + "_NG" + str(num_genes) + "_TS" + str(tm_size) + "_FE" + str(frac_elites) + "_FP" + str(frac_parents) + "_MP" + str(mutation_prob) + "_" + mutation_type + ".png")
	
	#try the next case
	x = x + 1
