id = "PassengerId"
target = "Survived"
features = ["Pclass",
			"Age",
			"SibSp",
			"Parch",
			"Fare",
			"Rank",
			"Sex",
			"Embarked",
			"Ticket"]
samplingsNum = 10
samplingsRate = 0.4

titlesEncoding = {'Col' : 0,
					'Miss' : 1,
					'Lady' : 2,
					'Rev' : 3,
					'the Countess' : 4,
					'Capt' : 5,
					'Sir' : 6,
					'Mme' : 7,
					'Dr' : 8,
					'Master' : 9,
					'Don' : 10,
					'Ms' : 11,
					'Mlle' : 12,
					'Major' : 13,
					'Jonkheer' : 14,
					'Mr' : 15,
					'Mrs' : 16,
					'Dona' : 17,}

sexEncoding = {'female' : 0,
				'male' : 1}

citiesEncoding = {'Q' : 0,
					'C' : 1,
					'S' : 2}

bootstrapSampling = False