def normalise_tangible(matrix):
	"""
		This function is used to determine the normalised matrix
	"""
	trans_mat = find_transpose(matrix)
	
	for i in range(len(trans_mat)):
		maxItem = max(trans_mat[i])
		minItem = min(trans_mat[i])
		for j in  range(len(trans_mat[i])):
			if(posneg[i]):
				#find the normal value of a positive criteria
				trans_mat [i][j] = (trans_mat [i][j] - minItem) / (maxItem - minItem)
			else:
				#find the normal value of a negative criteria
				trans_mat [i][j] = (maxItem - trans_mat [i][j]) / (maxItem - minItem)
	return find_transpose(trans_mat)

def normalise_intangible(matrix):
	"""
		This function is used to determine the normalised matrix
	"""
	trans_mat = find_transpose(matrix)
	
	for i in range(len(trans_mat)):
		total = sum(trans_mat[i])
		for j in  range(len(trans_mat[i])):
			trans_mat [i][j] = trans_mat [i][j]/total
	return find_transpose(trans_mat)


def find_transpose(matrix):
	"""
		This function is used to find the transpose of a given matrix
	"""
	transposed = [[0 for x in range(attributesNum)] for i in range(attributesNum)]
	for i in range(len(matrix[0])):
		for j in range(len(matrix)):
			transposed[i][j]=matrix[j][i]
	return transposed


def find_weights(matrix):
	"""
		This function is used to find the weights of attributes as we get from the comparison matrices 
	"""
	
	weight_vector = [[0] for k in range(len(matrix))]
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			weight_vector[i][0]+=matrix[i][j]
		weight_vector[i][0]=weight_vector[i][0]/len(matrix)
	return weight_vector

def find_CI(matrix,weight):
	"""
		This function is used to find the consistency index of the given matrix
	"""
	result=multiply_matrices(matrix,weight)
	
	RI = [0 ,0.58, 0.90 ,1.12 ,1.24 ,1.32, 1.41, 1.45, 1.51]
	sum = 0
	
	for i in range(len(result)):
		sum += result [i][0] / weight[i][0]
	
	lam = sum / len(result)
	CI = (lam - len(result)) / (len(result) - 1)
	CI_by_RI = CI / RI [len(result) - 2]
	
	return CI_by_RI


def multiply_matrices(X,Y):
	result = [[0 for y in range(len(Y[0]))] for k in range(len(X))]
	for i in range(len(X)):
	   # iterate through columns of Y
	   for j in range(len(Y[0])):
	       # iterate through rows of Y
	       for k in range(len(Y)):
	           result[i][j] += X[i][k] * Y[k][j]
	
	return result


#takes the required attributes
attributesNum = int(input())

attributes = []
posneg = []
for i in range(attributesNum):
	attributes.append(raw_input())
	posneg.append(int(input()))

print("The attributes are : ")
print(attributes)

alternativesNum = int(input())

A = [[0 for x in range(attributesNum)] for i in range(attributesNum)]

#take the matrix as input
for i in range(attributesNum):
	for j in range(attributesNum):
		if(i == j):
			A [i][j] = 1.0
		elif(i < j):
			A [i][j] = float(input())
		elif(i > j):
			A [i][j] = 1.0/A[j][i]

print("\n")

print("Attribute comparison matrix : ")

print('\n'.join(['\t'.join([('{:15}').format(item) for item in row]) 
      for row in A]))

normalised = normalise_intangible(A)
print("\n")

print("Normalised attribute comparison matrix : ")

print('\n'.join(['\t'.join([('{:15}').format(item) for item in row]) 
      for row in normalised]))

normal_weights = find_weights(normalised)
print("\n")
print("The weights for A : ")
print(normal_weights)

ci = find_CI(normalised, normal_weights)
print("\n")
print("CI for A : "+str(ci))

altAttMatrices=[]

for x in range(attributesNum):
	B = [[0 for y in range(alternativesNum)] for k in range(alternativesNum)]
	#print("Enter the comparison matrix for attribute "+str(x+1) +":")
	for i in range(alternativesNum):
		for j in range(alternativesNum):
			if(i == j):
				B [i][j] = 1.0
			elif(i < j):
				B [i][j] = float(input())
			elif(i > j):
				B [i][j] = 1.0/B[j][i]
	altAttMatrices.append(B)

for i in range(len(altAttMatrices)):
	print("Attribute specific score matrix for attribute "+str(i+1))
	print('\n'.join(['\t'.join([('{:15}').format(item) for item in row]) 
      for row in altAttMatrices[i]]))

score_matrix=[]

for i in range(len(altAttMatrices)):
	score_matrix.append(find_weights(normalise_intangible(altAttMatrices[i])))

print("\n")

print("Final Score matrix : ")

print('\n'.join(['\t'.join([('{:7.6}').format(item) for item in row]) 
      for row in find_transpose(score_matrix)]))
