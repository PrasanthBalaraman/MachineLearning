# A dictionary of movie critics and their ratings of a small 
# set of movies 
critics={'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, 'The Night Listener': 3.0}, 
'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0, 'You, Me and Dupree': 3.5}, 
'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0, 'Superman Returns': 3.5, 'The Night Listener': 4.0}, 
'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'The Night Listener': 4.5, 'Superman Returns': 4.0, 'You, Me and Dupree': 2.5}, 
'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0, 'You, Me and Dupree': 2.0}, 
'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5}, 
'Toby': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0,'Superman Returns':4.0}}

from math import sqrt

def sim_distance(prefs, person1, person2):
	similar_items = {}
	for item in prefs[person1]:
		if item in prefs[person2]:
			similar_items[item] = 1
	
	if len(similar_items)==0:
		return 0
		
	sum_of_squares = sum([pow(prefs[person1][item]-prefs[person2][item], 2)
						for item in prefs[person1] if item in prefs[person2]])
	
	return 1/(1+sum_of_squares)

def sim_pearson(prefs, person1, person2):
	similar_items = {}
	for item in prefs[person1]:
		if item in prefs[person2]:
			similar_items[item] = 1
	
	n = len(similar_items)
	
	if n == 0:
		return 0
	
	# add up all preferences 
	sum1 = sum([prefs[person1][item] for item in similar_items])
	sum2 = sum([prefs[person2][item] for item in similar_items])

	# sum of squares 
	sum1Sq = sum([pow(prefs[person1][item], 2) for item in similar_items])
	sum2Sq = sum([pow(prefs[person2][item], 2) for item in similar_items])

	# sum of product of preferences 
	pSum = sum([prefs[person1][item]*prefs[person2][item] for item in similar_items])

	# calculate pearson score 
	num = pSum-(sum1*sum2/n)
	den = sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
	if den==0:
		return 0
	
	return num/den

# print(sim_distance(critics, 'Lisa Rose', 'Gene Seymour'))
# print(sim_pearson(critics, 'Lisa Rose', 'Gene Seymour'))

def topMatches(prefs, person, n=5, similarity=sim_pearson):
	scores = [(similarity(prefs, person, other), other)
			for other in prefs if other!=person]
	scores.sort()
	scores.reverse()
	return scores[:n]

# print(topMatches(critics, 'Toby'))

# get recommendations for a person using a weighted average of every other user's rankings 
def getRecommendations(prefs, person, similarity=sim_pearson):
	totals={}
	simSums = {}
	for other in prefs:
		if other==person:
			continue
		sim=similarity(prefs, person, other)

		if sim<=0:
			continue
		for item in prefs[other]:
			if item not in prefs[person] or prefs[person][item]==0:
				totals.setdefault(item, 0)
				totals[item]+=prefs[other][item]*sim
				simSums.setdefault(item, 0)
				simSums[item]+=sim
	
	rankings=[(total/simSums[item], item) for item, total in totals.items()]
	rankings.sort()
	rankings.reverse()
	return rankings

# print(getRecommendations(critics, 'Toby'))
# print(getRecommendations(critics, 'Toby', similarity=sim_distance))

