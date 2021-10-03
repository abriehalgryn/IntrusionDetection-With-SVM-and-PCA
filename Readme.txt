the GUI is easy (gui.py), you unpout the Data, pca and svm config settings that you want to train you system on and train the system with the button.
the test button will show the acuracy and scores of the data
this model (after training) can be easily saved for later use so that you dont have to waste time on training it



the graph program: (generate scores.py)
this uses "scores.txt" which is some of the data that i generated using that for loop i was talking about last week, the things you can display is:
changes that the following values have on the code:

"n_eigenvectors": "Number of Eigenvectors"
"normalize_method": "Normalization method"
"kernel": "Kernel"
"category_classifications": "Category classifications"
"whiten_eigenvectors": "Whitened Eigenvectors"
"gamma": "Gamma"
"C": "C"


to display the graph for the required testing param, you have to change line 14 TEST_PARAMETER = "input testing param"
	- the correct testing param has to be in "" and can be found in the above dictionary. e.g. TEST_PARAMETER = "normalize_method"
this should now work when you run the code

PERCENTAGE line 10:
change this value from 0.01 (1%) to 1 (100%) to only display the top x percent of data on the graph