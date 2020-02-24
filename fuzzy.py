#  Copyright © 2018 Siddharth Mukkamala. Terms of use below. 

#  Despite the copyright, the following Software is made available to anyone interested free of charge; you may copy, distribute or modify the source code
#  as you like, provided you adhere to the following conditions:
#  1) Any distribution of a copy of this software must be made free of charge; you may not sell a copy of the source code/software.
#  2) If you modify the source code, you may not publish or distribute the resulting derivative work without explicit permission from the author.*
#  3) Understand that the software comes without any warranty.
#

#  Contact Information: E-mail: siddharth.world@gmail.com
#  NOTE: This software is an implementation created from scratch by the Author with influences from the abstract works of Dr. Detlef Nauck and Dr. Rudolf Kruse.
#  NOTE 2: Additionally, the software as it exists at this point in time (April 2018) is in its beta version, and I am hoping to improve upon 
#  the architecture and optimization in the foreseeable future.

#  * I can assure you the permission will be granted in writing promptly, I just want to stay informed about the potential benefits of Neuro - Fuzzy classifiers. 

# ~ ==============================================================================================================================================================================================


# Fuzzy set implementation with a triangular membership function

# Imports

import matplotlib.pyplot as plt
import json

# Classes

# Simple Fuzzy Set Class with a triangular membership function. The parameters of the membership function can be changed
# to allow learning 

# Allows shouldering for the rightmost and leftmost fuzzy sets - this is to ensure we dont get a membership fucntion of 0 for linguistic variable 'Large', when value is on the higer extreme of fuzzy set 'Large'
class FuzzySet:
	linguistic_term = ''
	a = 0
	b = 0
	c = 0
	l_shouldered = False
	r_shouldered = False

	def __init__(self, linguistic_term, parameters_lst, l_shouldered = False, r_shouldered = False):
		self.linguistic_term = linguistic_term
		self.a = parameters_lst[0]
		self.b = parameters_lst[1]
		self.c = parameters_lst[2]
		if (l_shouldered == True):
			self.l_shouldered = True

		if (r_shouldered == True):
			self.r_shouldered = True
		

	# R -> [0,1]
	def membership_degree(self,x):
		if (self.l_shouldered == True):
			if x < self.b:
				return 1	
			if self.b <= x and x <= self.c:
				return (self.c - x)/(self.c - self.b)
			else:
				return 0
		
		if (self.r_shouldered == True):

			if self.a <= x and x < self.b:
				return (x - self.a)/(self.b - self.a)
			if self.b <= x:
				return 1
			else:
				return 0
		else:

			if self.a <= x and x < self.b:
				return (x - self.a)/(self.b - self.a)
			if self.b <= x and x <= self.c:
				return (self.c - x)/(self.c - self.b)
			else:
				return 0

	# Returns assigned Linguistic Term
	def name(self):
		return self.linguistic_term

	def change_a(self, val):
		self.a += val

	def change_b(self, val):
		self.b += val

	def change_c(self, val):
		self.c += val




# User defining a number of fuzzy sets partitioning the domain of an input feature  

class FuzzySets:
	lst_fuzzysets = []

	def __init__(self, lst_linguistic_terms, lst_parameters_lst):
		temp_lst = []
		# self.lst_fuzzysets = []
		for i in range(len(lst_linguistic_terms)):
			# print("ling term: ",lst_linguistic_terms[i] )
			# print("membership function parameters: ", lst_parameters_lst[i])
			if i == 0:
				temp_lst.append(FuzzySet(lst_linguistic_terms[i], lst_parameters_lst[i], l_shouldered = True))
			elif i == len(lst_linguistic_terms) - 1:
				temp_lst.append(FuzzySet(lst_linguistic_terms[i], lst_parameters_lst[i], r_shouldered = True))
			else:

				temp_lst.append(FuzzySet(lst_linguistic_terms[i], lst_parameters_lst[i]))
			# print("size of fuzzysets: ", len(self.lst_fuzzysets) )
		self.lst_fuzzysets = temp_lst

	# A graphical illustration of the fuzzy sets of the domain.
	# Maybe change to save into a folder. HAVE TO CHANGE THIS! LEFT SHOULDER not working

	def plot_sets(self):
		for i in range(len(self.lst_fuzzysets)):
			# print(self.lst_fuzzysets[i].a)
			# print(self.lst_fuzzysets[i].b)
			# print(self.lst_fuzzysets[i].c)
			print(self.lst_fuzzysets[i].name())

			if self.lst_fuzzysets[i].l_shouldered == True:
				plt.plot([self.lst_fuzzysets[i].a,self.lst_fuzzysets[i].b,self.lst_fuzzysets[i].c], [1,1,0])

			elif self.lst_fuzzysets[i].r_shouldered == True:
				plt.plot([self.lst_fuzzysets[i].a,self.lst_fuzzysets[i].b,self.lst_fuzzysets[i].c], [0,1,1])
			else:

				plt.plot([self.lst_fuzzysets[i].a,self.lst_fuzzysets[i].b,self.lst_fuzzysets[i].c], [0,1,0])
		# 	plt.plot([1,2,3], [1,1,0])
		# 	plt.plot([3,4,5], [0,1,0])
		# 	plt.plot([5,6,7], [0,1,1])
		plt.show()


	def return_fuzzysets(self):
		return self.lst_fuzzysets

	def return_fuzzysets_count(self):
		return len(self.lst_fuzzysets)


class Fuzzy_domain:
	lst_Fuzzy_features = []

	def __init__(self, lst_fuzzyfeatures):
		self.lst_Fuzzy_features = lst_fuzzyfeatures

	def return_fuzzy_domain(self):
		return self.lst_Fuzzy_features

	def num_features(self):
		return len(self.lst_Fuzzy_features)






# Creating a Rule class that allows us to retrieve the linguistic variables that are the antecedents
class Rule:
	# Variables required for a rule
	# List of antecedent term fuzzy sets
	lst_fuzzysets = []
	# List of weights for the rule from the first layer
	fuzzy_weights = []
	# Output node - the node that categorizes the training sample into a cluster/category, 0 being first category of node array and etc. (the following of THEN)
	# This rule says we are going to categorize the antecedents into the cluster node (int between 0 to N-1, where N is the number of output nodes/clusters)
	output_node = 0


	def __init__ (self, lst_fuzzysets, weights, output_node):
		self.lst_fuzzysets = lst_fuzzysets
		self.fuzzy_weights = weights
		self.output_node = output_node

	def rule_ling_terms(self):
		return self.lst_fuzzysets

	def readable_Rule(self):
		rule_str = "If " 

		#['setosa', 'versicolor', 'virginica']
		# Change here to add what you want to call categories
		if self.output_node == 0:
			category = "Setosa"
		elif self.output_node == 1:
			category = "versicolor"
		else:
			category = "virginica"

		for fuzzyset, weight in zip(self.lst_fuzzysets, self.fuzzy_weights):
			# rule_str =  rule_str + "AND " + term + " is " + str(round(weight,2)) + " "
			rule_str =  rule_str + "AND " + fuzzyset.name() +  " "
		rule_str = rule_str + "THEN " + category
		return rule_str

class sym_rule:
	# Antecedent is a list of fuzzysets described by linguistic term, each linguistic term is referring to one feature 
	lst_fuzzysets = ''
	m = [] # contains list of m's whichs are dicts of the symbolic variables (to express as fuzzy sets)
	output_node = 0


	def __init__ (self, antecedent, m, output_node):
		self.lst_fuzzysets = antecedent
		self.m = m 
		self.output_node = output_node

	# returns the list of fuzzy sets 
	def antecedent(self):
		return self.antecedent

	def change_m(self,m):
		self.m = m

	def output(self):
		return self.output

	def redable(self):
		rule_str = "If " 

		#['setosa', 'versicolor', 'virginica']
		# Change here to add what you want to call categories
		if self.output_node == 0:
			category = "Setosa"
		elif self.output_node == 1:
			category = "versicolor"
		else:
			category = "virginica"

		for fuzzyset in self.lst_fuzzysets:
			# rule_str =  rule_str + "AND " + term + " is " + str(round(weight,2)) + " "
			rule_str =  rule_str + "AND " + fuzzyset.name() +  " "
		rule_str = rule_str + " AND m " + json.dumps(self.m) + " "
		rule_str = rule_str + "THEN " + str(self.output_node)
		
		return rule_str





