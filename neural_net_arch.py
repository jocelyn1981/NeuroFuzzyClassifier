# Neuro-Fuzzy Architecture
import fuzzy
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy

class Neuro_Fuzzy_Network:
	# Variables needed
	num_features = 0
	max_rules = 0
	num_outputs = 0
	# Contains all fuzzy sets needed
	fuzzy_domain = 0
	rules = []
	best_rules = []
	learning_rate = 0
	sym_rules = []
	sym_domain = 0

	# Neural Net variables



	def __init__(self, fuzzy_domain, max_rules, num_outputs, learning_rate, sym_domain):
		# Fuzzy_domain is an instance of the class Fuzzy_domain 
		self.num_features = fuzzy_domain.num_features()
		self.fuzzy_domain = fuzzy_domain
		self.max_rules = max_rules
		self.num_outputs = num_outputs
		self.learning_rate = learning_rate
		self.sym_domain = sym_domain

	# takes a list of symbol variables and returns a dict for the network to handle 

	def sym_domain_handler(self,sym_domain):
		dict_domain = []
		for symbol_lst in sym_domain:
			dict_temp = {}
			# dict_temp = Counter()
			counter = 0
			for sym_var in symbol_lst:
				dict_temp[sym_var] = counter
			dict_domain.append(dict_temp)

		return dict_domain

		

	def return_num_outputs(self):
		return self.num_outputs

	def return_num_features(self):
		return self.num_features

	def return_max_rules(self):
		return self.max_rules

	def return_rules(self):
		return self.rules


		# Rule -> No return
	def add_rule(self, Rule):
		# Needs updating
		self.rules.append(Rule)


	# Helper function that takes in a single data point and a FuzzySets instance, returns a tuple of the linguist term that datapoint
	# belong to as well as the degree of memberhsip (the higest degree of membership is chosen with its linguistic variable)
	# Called by feedforward, FIRST STEP IN FEEDFORWARD MECHANISM 

	def ling_mem(self, value, fuzzysets):
		fuzzy_lst = fuzzysets.return_fuzzysets()

		ling_val = ""
		max_val = 0

		for fuzzyset in fuzzy_lst:
			
			mem_val = fuzzyset.membership_degree(value)
			if mem_val > max_val:
				max_val = mem_val
				ling_val = fuzzyset
			


		return ling_val, max_val 

	def does_rule_exist(self, ling_terms):
		for rule in self.rules:
			if ling_terms == rule.rule_ling_terms():
				# print(rule.readable_Rule())
				return True

		return False


	# Function rule_base creates a set of rules from the data, from each training example

	def rule_base(self, input_vec, output_val):
		# WHERE output_val is a value that is a number between 0 and N-1 , relating to the position of the 1 value in the output array 
		# Hence output_val denotes which caetgory the training example belongs to!!!!!

		# For example, if output array is [0,0,0,1,0], then output_val has 5 possible values. and the value is 3 and is between [0,4]


		# when an input_vec passes, we store the highest membership function as well as the ling terms in order, relating to the input parameters
		ling_terms = []
		mem_vals = []


		# iterate over the input array and return a linguistic term and the highest membership func value
		# for each row value in input_vec, return a tuple of the linguist term and highest membership function
		for value, fuzzysets in zip(input_vec, self.fuzzy_domain.return_fuzzy_domain()):
			ling, mem_val = self.ling_mem(value, fuzzysets)
			ling_terms.append(ling)
			mem_vals.append(mem_val)

		# Find if rule with the same linguistic variables, IN THE SAME ORDER exists

		# Comapres each rules ling_terms and their order with ling_term of the datapoint
		if not self.does_rule_exist(ling_terms):
			# print("rule doesnt exist")
			# Create a rule with the ling_terms and the mem_vals which will act as weights 
			new_rule = fuzzy.Rule(ling_terms, mem_vals, output_val)

			#Add the newly created rule to rules by using function add_rule(rule)
			self.add_rule(new_rule)


	def feedforward(self, X, y, return_rule_activations = False, use_best_rulebase = False):
		# we feed forward each pattern one at a time

		# X represents the pattern and y represents the classification 
		# Initializing a list of activations
		if use_best_rulebase == True:
			rule_activations = [0] * len(self.best_rules)
			counter = 0
			for rule in self.best_rules:
				# print("new rule", counter)
				min_activation = float("inf")
				for val, fuzzyset in zip(X,rule.lst_fuzzysets):
					temp_val = fuzzyset.membership_degree(val)
					# print("temp val is : ", temp_val)
					if temp_val < min_activation:
						min_activation = temp_val
					rule_activations[counter] = min_activation 
				counter += 1
			output_activations= [0]* self.num_outputs

			# create a list of output nodes the rules are connected to
			outputs_nodes = [rule.output_node for rule in self.best_rules]
			# Zip activations and outputs 
			for activation, node in zip(rule_activations, outputs_nodes):
				if activation > output_activations[node]:
					output_activations[node] = activation

			# print("OUTPUTS ARE: ", output_activations)
			# Return node is where the neuro fuzzy system has chosen to classify the input!

			return_node = output_activations.index(max(output_activations))	
			if return_rule_activations == True:
				return return_node, np.asarray(rule_activations).reshape((len(self.best_rules),1))
				
			return return_node

		else:

			rule_activations = [0] * len(self.rules)
			counter = 0
			for rule in self.rules:
				# print("new rule", counter)
				min_activation = float("inf")
				for val, fuzzyset in zip(X,rule.lst_fuzzysets):
					temp_val = fuzzyset.membership_degree(val)
					# print("temp val is : ", temp_val)
					if temp_val < min_activation:
						min_activation = temp_val
					rule_activations[counter] = min_activation 
				counter += 1


		

			output_activations= [0]* self.num_outputs

			# create a list of output nodes the rules are connected to
			outputs_nodes = [rule.output_node for rule in self.rules]
			# Zip activations and outputs 
			for activation, node in zip(rule_activations, outputs_nodes):
				if activation > output_activations[node]:
					output_activations[node] = activation

			# print("OUTPUTS ARE: ", output_activations)
			# Return node is where the neuro fuzzy system has chosen to classify the input!

			return_node = output_activations.index(max(output_activations))	
			if return_rule_activations == True:
				return return_node, np.asarray(rule_activations).reshape((len(self.rules),1))
				
			return return_node

	# For a single symbolic variable

	def init_sym_rulebase(self, num_categorical_variables):
		num_outputs = self.num_outputs

		# Number of vectors to be created related to number of categorical variables, and initialize to zeros
		
		# Create an empty m with the number of possible category values for the feature. For example, if we have v5 = (Image, Text, Video)
		# then we have to create an m with 3 zeros which will contain the frequencies of symbolic variables 

		# Have to allow the following code to work on multiple symbolic variables 



		for rule in self.rules:
			for i in range(num_outputs):
				self.sym_rules.append(fuzzy.sym_rule(rule.lst_fuzzysets, self.sym_domain_handler(self.sym_domain), i))

		# Each m now is a list of dicts that are the fuzzy sets of the symbolic variables 

	def populate_sym_rulebase(self, x, sym_val, y):
		# This function essentitally takes in each example in the training data and figures out which of the rules it belongs to 

		# ... have to figure out how that works...

		# Have to figure out which index in m represents which symbolic variable 


		# Notes for implementation:
		# Take the x and apply the fuzzy sets of each rule, the rule can be simple numerical rules used or rule bases,

			# Activations , create a zero array for all the activations
		activations = np.zeros(len(self.rules))
		counter = 0
		for rule in self.rules:
			# compute activation for the example , p, with which this function is called by

			# We have a list of 4 data points in p and 4 fuzzysets of each rule, zip them
			temp_val = np.inf
			for datapoint, fuzzyset in zip(x, rule.lst_fuzzysets):
				# print(datapoint)
				val = fuzzyset.membership_degree(datapoint)
				if val < temp_val:
					temp_val = val 

			activations[counter] = temp_val
			counter += 1

		# print(activations)

		# With the activations, return the indices where the value is NOT zero, this means that there were rules that worked well with this
		non_zeros_indices = np.nonzero(activations)[0]

		#isolate these rule indices, then map them to the sym_rules, since there are len( num_classes*rules ) = len( sym*rules )
		# Create a list of indices with the additional + 2 other indices since the sym_rules are duplicates of the rules
				# Since we are just recording the indices of the activations, we dont care about order.

				# Initialize empty numpy array
		non_zeros_indices_sym = np.array([], dtype = int)

		for index in non_zeros_indices:
			# record next 2 indices 
			index = index * self.num_outputs
			a = index + 1
			b = index + 2
			non_zeros_indices_sym = np.append(non_zeros_indices_sym,[index,a,b])

			# non_zeros_indices_sym now contains all the indices of the symbolic rules which have activations and need to count
			# occurances of symbolic variables now			

		# Since we can access the symrules using recorded indexes we have to update m accordingly, uisng a pre-defined formula for the position of sym variables in m

		for index in non_zeros_indices_sym:		

			sym_rule = self.sym_rules[index]
			# print("the sym rule's readbale rule is: ", sym_rule.redable())
			# HERE, iterate through list using a counter and update each dict indepentadly for multiple symbolic variables !!!!! USE COUNTER!!
			for i in range(len(self.sym_domain)):

				if sym_rule.output_node == y:

					sym_rule.m[i][sym_val[i]] += 1

		return 

	def normalise(self, m):
		for sub_m in m:


			target = 1
			raw = sum(sub_m.values())
			if raw != 0:
				factor = target/raw
			
				for key, value in sub_m.items():
					sub_m[key]= value*factor

	def remove_zero_value_rules(self):

		# The following function will return TRUE if any of the dicts in the list arn't empty. 
		# It achieves this by looping through each dict and returns if any one of the dicts's values sum up to greater than 1

		def determine(rule):
			for sub_m in rule.m:
				if sum(sub_m.values()) > 0:
					return True

			return False 
		self.sym_rules[:] = [sym_rule for sym_rule in self.sym_rules if determine(sym_rule)]
		# somelist[:] = [tup for tup in somelist if determine(tup)]

	def remove_contradictions(self):

		def compare(current_sr, sr):
			
			lst = list(current_sr.m.values()) 
			lst2 = list(sr.m.values())
			temp = []
			for l, l2 in zip(lst,lst2):
				temp.append(min(l,l2))
			return max(temp)
 

		# Step 1, take a rule and look at the next (num_outputs - 1 ) rules to see if there is a match, and whether a contradiction exists
		for i in range(len(self.sym_rules) - (self.num_outputs - 1) ):
			current_sym_rule = self.sym_rules[i]

			# take the ith rule and check the next (num_outputs -1 ) rules
			for j in range(1, self.num_outputs):
				# use compare to return if a contradiction exists and print for now
				print(compare(current_sym_rule, self.sym_rules[i+j]))


		# This activations function returns the list of rule performances 
	def activations(self, X_matrix, train_sym ,y_vals):
		# Development for the performance measure of a rule. The research paper numeric + symbolic data does not adequately show us how to caluclate the 
		# output of a rule when the numeric antecedent is not 1, i.e. the degree of fulfillment of the numeric part. 
		# In this case, I will use either the min or max function. Both have their merits and will be discussed in the manual.

		rule_performances = [0] * len(self.sym_rules)


		rule_activations = [0] * len(self.sym_rules)
		for x,sym,y in zip(X_matrix, train_sym, y_vals):

			# THIS PART OUTPUTS ACTIVATIONS FOR THE NUMERIC VARIABLES
			counter = 0
			for rule in self.sym_rules:
				# print("new rule", counter)
				min_activation = float("inf")
				for val, fuzzyset in zip(x,rule.lst_fuzzysets):
					temp_val = fuzzyset.membership_degree(val)
					# print("temp val is : ", temp_val)
					if temp_val < min_activation:
						min_activation = temp_val
					rule_activations[counter] = min_activation 
				counter += 1

			# THE BELOW code has to be changed to allow to handle more than 1 symbolic variable 
			# Get a list of fuzzyvalues from m (from each rule) related to the sym_val of the current training example,
			# rule_vals_for_sym = [rule.m[sym] for rule in self.sym_rules]

			rule_vals_for_sym = []
			for rule in self.sym_rules:

				# The following code finds the min value among the symbolic variables, similar to how we found the min value for the numerical variables
				min_val = np.inf 
				for i in range(len(sym)):
					# print(i)
					cur_val = rule.m[i][sym[i]]
					# print(cur_val)
					if cur_val < min_val:
						min_val = cur_val

				# Now that we have the min value among the symbolic variables for the given example, we append it to rule_vals_for_sym
				rule_vals_for_sym.append(min_val)

			# compare activations and rule_vals_for_sym and take the min value of the smallest numerical fuzzy value and the smallest symbolic fuzzy value (personal choice, check back to offer another option)
			rule_performance_val = [ min(val1,val2) for val1, val2 in zip(rule_activations, rule_vals_for_sym)]

			# if rule performace val for the rule is not zero, then add it to 'global' performance val by checking if ouput of training 
			# example is the same as the rule consequent 
			# counter = 0
			for i in range(len(self.sym_rules)):
				if self.sym_rules[i].output_node == y:
					rule_performances[i] += rule_performance_val[i]
				else:
					rule_performances[i] -= rule_performance_val[i]

		return rule_performances

	def best_sym_rules(self, X_matrix, train_sym, y_vals):
		max_rules_number = self.max_rules
		rule_performances = self.activations(X_matrix, train_sym, y_vals)
		# print(rule_performances)
		array = np.asarray(rule_performances)
		array = array.argsort()

		for i in range(1, max_rules_number + 1):
			# print(array[-i])
			print(self.sym_rules[array[-i]].redable())


	def redable_rule(self, categories,sym_categories ,sym_rule):
		counter = 0
		antecedents = sym_rule.lst_fuzzysets

		formatted_str = ""

		for i in range(len(categories)):
			if i == 0:
				formatted_str +=  categories[i] + " is " + antecedents[i].name()
			else:
				formatted_str += " AND " + categories[i] + " is " +antecedents[i].name()

		# Handle formatting of symbolic part of rule
		for j in range(len(sym_categories)):
			formatted_str += "  AND " + sym_categories[j] + "  is  " + json.dumps(sym_rule.m[j])


		return formatted_str + " THEN " + str(sym_rule.output_node) 




