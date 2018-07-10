- Tableau worksheet is too large to be put in github, please find it in the public link:- 
https://public.tableau.com/profile/nutthika.rattanakornkul#!/vizhome/Crime_Airbnb_Chicago/Story1
- For Web application, to run it you need following tools:-
	- Visual Studio 2017
	- Install python and sklearn packages
- For prediction part, there is 2 files:-
	- classification.py : this one is the command we used to train and select the features of the model. However, the training dataset is not included here, because it is too big, which is not allowed in github.
	- genThreshold: this one is used to calculate the Threshold (average) line that is shown as the grey line in the web applicatio in prediction line graph.
	- mlp.pk1 is the model serialised in file. This one is used in the web application.	
	- scaler.phl is the normaliser generated from training data. This one is also used in web app for normalising user input to the same scale as training set before feeding to the model.
