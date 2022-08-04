# beerRecommenderSystems
Recommender systems for the beer review dataset from Beeradvocate using KNN, SVD and Pearson Correlation

The dataset
We have chosen a beer review dataset from Beeradvocate. The data was collected over a period of more than ten years up to November 2011, and includes approximately 1.5 million reviews. Each review consists of an overall rating as well as ratings under the following headings: appearance, aroma, palate, and taste. Reviews also include the username of the reviewer and product information including beer id, name, style, abv, and brewery name and id. The dataset was downloaded as a csv file.


Importing and cleaning the data

We used the “Pandas” library in python to read in the csv file and store the data in a data frame. Error handling was implemented here to ensure the file had a .csv extension and was found in the current working directory. After experiencing difficulties due to the magnitude of the data, we decided to take a smaller sample of reviews. Therefore, rather than checking all data in the data frame which felt unnecessary, we chose to ensure the data frame contained the four columns intended for use in our recommender systems, discussed in further detail later.

Rather than taking a random sample, we decided to filter the data to exclude beers with few reviews. Having tested a number of thresholds to remove an adequate number of rows, we settled on a lower limit of 500. Any reviews pertaining to beers which had been reviewed less than 500 times were removed from our dataset. To further cut the data, we set an upper limit of 2000, meaning any reviews pertaining to beers which had been reviewed more than 2000 times were removed. In this way, we have removed the most popular beers, which users are likely already aware of, as well as unknown or lesser-known beers, for which recommendations would be less accurate as less historical data is available. We cleaned the data further by removing reviews by users who had submitted less than 100 reviews. This was done for the same reason as for the lower limit on beers. Having cleaned the data in this way, we were left with ~265,000 reviews of 562 different beers from 1360 different users.

It should be noted that a number of techniques were discussed for cleaning the data, including random sampling or taking the first x entries. However, we were worried that cleaning the data in this way could potentially leave us with a dataset of various different beers which had only been reviewed a small number of times, or various different users who had only reviewed a small number of beers, providing little historical data on which to test our recommender system methods. We also had concerns around which value of x should be chosen if we were to work with the first x rows. We had discussed using 10,000 reviews, which seems like a lot, but in reality, may have only accounted for 10 different beers depending on the split of data.

When building our recommender systems we decided to focus solely on the overall rating of each beer, as opposed to having five ratings for each review. Therefore, the appearance, aroma, palate, and taste scores were also removed from the data. To implement the filtering methods outlined in Section 3, we deduced that only four of the original columns were required. Thus, the cleaned data contains the following fields:  
review_overall  
review_profilename  
beer_name 
beer_beerid 


Recommender Systems Implemented

All the recommender systems implemented are collaborative due to the nature of our dataset. With little information to distinguish users and items, we weren’t able to develop profiles describing the characteristics of reviewers or beers and hence we were unable to develop a useful content-based filter.

We researched the possibility of a content-based filter using beer_style as a category on which to compare items and users, but we found this was still too broad and might not provide meaningful recommendations.

Before implementing our recommender systems we ask the user to enter a beer name, upon which recommendations will be based. Error handling was implemented here to ensure the name entered was found in the list of defined beer names in the data. If the user enters an invalid name, they will be asked once again to enter a beer name. Three invalid attempts prompts the printing of all beers in the system in alphabetical order. It is hoped that this will allow the user to find the item of interest, which they may then copy and paste if necessary. The user is then asked how many recommendations they would like to see. Error handling is introduced here to ensure an integer value between one and twenty is entered.

After the recommender systems have performed their function, the user is given the option to seek recommendations for another beer or exit. In this case, if the user enters ‘N’, ‘n’, ‘No’, or ‘no’, the programme is terminated. Any other input is taken as ‘yes’, and the user is invited to restart the process by entering another beer name.
Recommender System 1: k-Nearest Neighbours
Our first collaborative filter was implemented using a nearest neighbours approach. The KNN algorithm assumes that similar things are near to each other, and thus finds the k most similar items to a particular item (in our case the beer entered by the user) based on a particular distance metric. We have chosen cosine similarity as our distance metric. Cosine similarity measures the angle between two vectors as opposed to the distance between two points. This allowed similar vectors pointing in the same direction as our initial vector to be found. KNN is a supervised machine learning algorithm in that it is trained using labelled data in order to handle unlabeled data appropriately. The nearest neighbours algorithm is beneficial in its simplistic approach but may struggle with larger datasets. That being said, the algorithm performs reasonably well on our reduced dataset.

Implementation:
The pivot function was used on the cleaned data to form a pivot table. The beer names were used as the index, and the reviewer usernames were used as the columns. The ratings made up the values of the table. Since most users only recommended a small proportion of the beers, the pivot table was sparse. Empty values were filled with zeros to allow for matrix operations to be performed. The csr_matrix function, imported from the spicy.sparse library was used to compress the matrix to allow for efficient row slicing and matrix-vector products. Having imported the NearestNeighbors module from sklearn.neighbors, we defined our kNN model using cosine similarity as our distance metric and a brute-force search algorithm. We determined the algorithm by letting it equal to auto. Auto assigns the best-suited algorithm for the data, which in our case was brute force. With a larger dataset, a different algorithm would likely be assigned. 

We fitted this model using our sparse matrix of ratings as training data. We then used the kneighbors() function to find the k nearest neighbours (where k is given by the user) of the user-inputted beer and returned these values as well as the cosine distances between them and the starting point (the user-input beer).  
Recommender System 2: Single Value Decomposition (SVD)
Singular value decomposition is a linear algebra method which can be used to decompose a utility matrix into three compressed matrices. For recommender systems, the compressed matrices can be used to make recommendations without referring back to the original matrix. 

SVD factorises matrices in the form:

Where: A is the original matrix.  U and V are orthogonal matrices with orthonormal eigenvectors. In our case U will hold information about the users and V will hold information about beers. S is a diagonal matrix. SVD gives us these matrices, which can be used to make accurate recommendations. 

We can find a truncated SVD of A by setting all but the first k largest singular values equal to zero and using only the first k columns of U and V. This allows us to represent the large number of reviews in a much smaller number of components. 

Implementation:
The ratings were put into a pivot table - this will act as our utility matrix. User name was used as the index and beer names as the columns. The ratings made up the values, and any absent values were replaced with 0 to allow for matrix operations. This gave a 1360x562 data frame. Since we are recommending beers, we can use the similarity between users to decide what beers to recommend. As truncated SVD is performed on columns, we transposed the matrix so that beers were maintained in the rows.

Truncated SVD imported from sklearn.decomposition was then used to compress values in the transposed ratings pivot down into a smaller number of latent variables. Truncated singular value decomposition (SVD) does not centre the data before computing the singular value decomposition. This means it can work with scipy.sparse matrices efficiently. The more latent variables used, the more accurate the estimated matrix will be; however, they also increase the processing time. A function was created to determine the exact amount of components that would be required to account for 90% of the variance of the original matrix. The random state was arbitrarily set to 1. This was done to make the result repeatable, i.e. the same beer would give the same results each time it was entered. The result was a 562 beers by 334 latent variable matrix.    

This matrix was passed into the Pearson r correlation coefficient imported from the NumPy library, which was used to estimate a covariance matrix. This gave a 562x562 matrix with a correlation for each set of beers. The column containing the user inputted beer was then extracted. This column contained the correlation between the user's chosen beer and every other beer in our dataset. This was sorted by correlation in descending order, and the top N results were printed out to the user. 
Recommender System 3: Pearson Correlation
Pearson’s correlation coefficient (PCC) measures the association between two variables.  Based on the method of covariance, PCC tells us not only about the magnitude of the association (the correlation), but also about the direction of the relationship. In theory, the closer the PCC is to 1, the more perfectly correlated the two variables are. Likewise, if the PCC is near 0, the correlation is said to be very low or non-existent. With such a sparse table of values, however, it is near impossible to reach correlations close to one. In our recommender system, we found that most values in the top recommendations ranged roughly from 0.2 to 0.4 - we therefore work under the assumption that,as these beers are the most highly correlated with the beer in question, there is some positive association.

Implementation:
Pearson Correlation could be implemented in one of two ways, both of which have been tried and tested in the code and provide the same results. The first method used was more manual, taking inspiration from the PCC formula:

https://www.google.com/search?q=pearson+correlation+coefficient+formula+copy+and+paste&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjEj9q-y635AhWVWcAKHYpSBDgQ_AUoAXoECAEQAw&biw=1440&bih=789&dpr=1#imgrc=zt5HAfyDcekFtM

Firstly, a function was created to take two parameters and calculate the Pearson correlation between them using the formula above. A second function was then created, taking arguments of (1) the beer name upon which recommendations are to be based, (2) a pivot table of rating values, and (3) the number of recommendations to be returned (n). This function loops through all beer names in the system, and calculates the Pearson correlation between these beers and the user-entered beer. The function skips over the user-entered beer when found in the list, as this would automatically return a value of 1, and is not useful to the user. Any non-null correlations and their respective beer names are stored in an array, which is then sorted in decreasing order of correlation. The top n recommendations are returned.

The second method uses the inbuilt corrwith() function to find Pearson correlation between the ratings of the user-entered beer and all beers in the system. The beer names and correlations are stored in a dataframe, and near perfect correlations are removed to ensure we are not recommending the beer entered by the user, which would add no real value. The beers are then sorted in descending order of correlation value, and the top 10 values are recommended.


Difficulties Encountered

Data cleaning:
As the original dataset had over 1.5 million reviews processing this data was slow. Cleaning the data was an essential step to allow for recommendations to be produced in an appropriate time frame. We wanted to ensure the cold start problem and data scarcity didn’t have an overly large impact on our recommender systems e.g. by disregarding beers with less than 50 reviews as opposed to 500. However, this may introduce some bias when assessing the accuracy of our systems. Also, to minimise processing time a large amount of useful data had to be disregarded.

SVD variance:
Choosing the components for SVD posed a few issues. From research, we found that the higher the number of components, the more variance of the original matrix had been included. However, additional components raised the processing time. We were unsure how to determine if we accounted for enough variance. We decided to add a function that would determine how many components would be required to cover a certain percentage of the variance. For our dataset, we found 90% variance still gave a reasonable run time. So the number of components required for 90% variance were used for the truncated SVD. For larger datasets, a lower percentage of variance could be chosen to allow for better processing time. 

Pivot tables:
We’ve used two pivot tables in our code, both of which use ratings as their values. One uses beer name as the index and reviewer name as the columns, while the other uses reviewer name as the index and beer name as the columns The use of two separate pivot tables arose from uncertainty as to efficiencies. We could have used one pivot table, but in doing so would have needed to transpose the values a number of times and/or loop through it to find the relevant values. We were unsure whether it would be more efficient to have two pivot tables, or to have one pivot table which would then require manipulation. In the end we settled on two pivots, but this could be changed if necessary.
