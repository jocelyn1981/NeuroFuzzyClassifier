# Func about fuzzyfying when given a max and min value:


def fuzzyfy(min_val, max_val, num_sets, fuzzy_percent):
	# min is min value of the feature 
	# max is max value of the feature
	# num_sets is the number of fuzzy sets the feature domain is split into
	# A value between [0,1], that refers to the percentage of fuzzification required 


	# calulate difference and the increment
	dif = (max_val - min_val)/num_sets
	fuzzy_sets = []

	for i in range(num_sets):
		val = min_val + i*dif
		fuzzy_sets.append([val, (val + dif)])

	fuzzy_val = fuzzy_percent* dif 

	for i in range(num_sets):
		if i == 0:
			fuzzy_sets[i][1] += fuzzy_val

		elif i == (num_sets -1):
			fuzzy_sets[i][0] -= fuzzy_val
		else:
			fuzzy_sets[i][0] -= (fuzzy_val/2)
			fuzzy_sets[i][1] += (fuzzy_val/2)

	parameters = []

	for fuzzy_set in fuzzy_sets:
		a = fuzzy_set[0]
		c = fuzzy_set[1]
		b = a +(c-a)/2
		parameters.append([a,b,c])




	return parameters

# print(fuzzyfy(0,10,5,0.2))


