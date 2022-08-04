# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:00:26 2020

@author: calummcgrath
"""

# load python libraries
import os
import pandas as pd
import numpy as np

#Get current working directory
currentDir= os.getcwd()

# set file name, in this case the beer reviews dataset
file_name = 'beer_reviews.csv'

# define boolean file_found and set as false 
file_found = False

# if file extension is csv, read data and check it includes the necessary columns
# otherwise print error error message to let user know the file should be a csv
if file_name.endswith('.csv'):

    # Append names of files you want to read to the current directory
    filesDir = os.path.join(currentDir, file_name)
    
    # if file found in given directory, read csv file into variable 'data' 
    # if data contains the column headings required for analysis set file_found boolean to true
    # if file not found or all relevant columns not included print error message
    try:
        data = pd.read_csv(filesDir)
        if set(['review_overall', 'review_profilename', 'beer_name', 'beer_beerid']).issubset(data.columns):
            file_found = True
        else:
            print("This dataset does not include the required columns for analysis. \nEnsure beer reviews dataset is entered.")
    except FileNotFoundError:
        print("Wrong file or file path. \nPlease ensure file name is correct and file is stored in the current working directory.")
else:
    print("File should be of type csv")
   
# if the data has been read in correctly, continue into data cleaning and recommender systems implementation
# otherwise file stops running
if file_found != False:

    # subset data so we have 4 columns rather than 13
    # overall rating, username of reviewer, name of beer, id of beer
    ratings = data[['review_overall', 'review_profilename', 'beer_name', 'beer_beerid']]
    ratings.head()
    # count number of times each beer is reviewed and sort in descending order
    ratingCount = ratings.groupby('beer_beerid')['review_overall'].count().sort_values(ascending=False)
    ratingCount.head()
    # merge ratingCount to ratings using beer id as comparison standard 
    ratingCountFull = ratings.merge(ratingCount, left_on = 'beer_beerid', right_on = 'beer_beerid', how='left', suffixes=('', '_count_per_beer'))
    
    # set lower limit of 500
    popularity_lower_threshold = 500
    # set upper limit of 2000
    popularity_upper_threshold = 2000
    
    # store data on beers with 500 or more reviews
    ratingsPopularBeers1 = ratingCountFull.query('review_overall_count_per_beer >= @popularity_lower_threshold')
    # filter this data further to remove beers with 2000 or more ratings
    ratingsPopularBeers = ratingsPopularBeers1.query('review_overall_count_per_beer < @popularity_upper_threshold')
    
    # count number of reviews for each user and sort in descending order
    userCount = ratingsPopularBeers.groupby('review_profilename')['review_overall'].count().sort_values(ascending=False)
    # merge userCount to ratingsPopularBeers using reviewer name as comparison standard
    userCountFull = ratingsPopularBeers.merge(userCount, left_on = 'review_profilename', right_on = 'review_profilename', how='left', suffixes=('', '_count_per_user'))
    
    # set lower limit of 100
    user_threshold = 100
    # get all reviews by users who have reviewed at least 100 beers
    ratingUserPopular = userCountFull.query('review_overall_count_per_user >=@user_threshold') 
    
    # store review values in a pivot table with beer names as index and user profile names as columns
    ratingsPivot = ratingUserPopular.pivot_table(index=['beer_name'],columns=['review_profilename'],values='review_overall',fill_value=0)
    
    # initialise varibale quit to false
    # this boolean will be used in determining whether the user wants recommendations for another beer or if they don't and the programme should terminate
    quit = False 
    
    # while the user still wants beer recommendations, let them enter a beer name and run recommender systems
    while quit == False:
        
        # initialise count variable to keep track of number of times user entered invalid input
        count = 0
        
        # while loop for user input beer name
        # if name entered not found in beers listed in pivot table, show error message and try again
        # if invalid beer name entered 3+ times, show list of available beers
        # otherwise break the loop
        while True:
            user_input = input("Enter a beer name to see recommendations:")
            if not user_input in ratingsPivot.index:
                print("Sorry, we couldn't find this beer in our system")
                count = count + 1
                if count > 2:
                    print(list(ratingsPivot.index))
                    print("\nAbove is a list of available beers in alphabetical order. \nBeer name should be entered exactly as listed.")
                continue
            else:
                break
            
        #print("\nBeer selected: " + user_input)
        
        # while loop for user input number of recommendations to show
        # check input is an integer
            # if it is, make sure it is between 1 and 20 (20 arbitrarily selected so console isn't flooded with recommendations)
            # if not between 1 and 20, print an error message and let user re-enter
        # if value is not an integer print error message and let them re-enter
        while True:
            num_recommendations = input ("How many recommedations would you like to see?")
    
            try:
               num_recommendations = int(num_recommendations)
               if num_recommendations > 0 and num_recommendations < 21:
                   print("\nFinding the top", num_recommendations, "recommendations for " + user_input + "...")
                   break
               else:
                   print("Number of recommendations should be between 1 and 20")
            except ValueError:
               print("Please enter a number in integer form")
               continue
        
        
        ################################################
        ################################################
        ################    KNN    #####################
        ################################################
        ################################################
        
        ## if we use profile name as index and beer name as columns we don't need for loop
        ## but apparently we still need the csr matrix to have beer name in index and profile name in columns
        ## don't really understand why
        ## otherwise when we run distances, indices..
        ## ValueError: Incompatible dimension for X and Y matrices: X.shape[1] == 1360 while Y.shape[1] == 562
        ## not sure if it's more efficient to pivot twice or pivot once but then need to loop through pivot to find index number
        
        # import compressed sparse row matrix from scipy.sparse
        from scipy.sparse import csr_matrix
        # convert pivot table values to sparse matrix
        ratingsMat = csr_matrix(ratingsPivot.values)
        
        # import nearest neighbors package from sklearn.neighbors
        from sklearn.neighbors import NearestNeighbors
        
        # define knn model with cosine similarity and brute force search algorithm
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        # fit the model on sparse matrix
        a1 = model_knn.fit(ratingsMat)
        #print(a1)
        
        
        # loop through each row in pivot table until the user-entered beer name is found
        # break the loop at this point
        # row number found will be used later
        for row in range(1,ratingsPivot.shape[0]):
            if(ratingsPivot.index[row]==user_input):
                break
        
        # run knn model on user_input beer name to find the user entered number of nearest neighbours
        # (index of and distance to each of these neighbours)
        distances, indices = model_knn.kneighbors(ratingsPivot.iloc[row,:].values.reshape(1,-1),n_neighbors=num_recommendations+1)
        
        #print("\nNearest Neighbours Recommendations\n")
        # loop through distances to nearest neighbours 
        # output user entered beer name, nearest neighbour beer names, and distance to from user entered beer to these nearest neighbours
        for i in range (0, len(distances.flatten('C'))):
            if i ==0:
                 print('\nNearest Neighbour Recommendations for {0}:\n'.format(ratingsPivot.index[row]))
            else:
                 print('{0}: {1}, with distance of {2}:'.format(i, ratingsPivot.index[indices.flatten('C')[i]], distances.flatten('C')[i]))
                 
                 
                   
        ################################################
        ################################################
        ################    SVD    #####################
        ################################################
        ################################################ 
        
        from sklearn.decomposition import TruncatedSVD    
        from sklearn.preprocessing import StandardScaler
        
        # making pivot table which will act as our utility matrix. empty values are filled with 0 to allow matrix operations 
        ratingsPivot1 = ratingUserPopular.pivot_table(index=['review_profilename'],columns=['beer_name'],values='review_overall',fill_value=0)
        
        #########################################################
        ##### finding optimal number of components for SVD#####
        ########################################################
        #standardizing the matrix
        Y = StandardScaler().fit_transform(ratingsPivot)
        # Making the sparse matrix
        Y_sparse = csr_matrix(Y)
        # Create and run an Truncated SVD with one less than number of features
        tsvd = TruncatedSVD(n_components=Y_sparse.shape[1]-1)
        Y_tsvd = tsvd.fit(Y)
        # List of explained variances
        tsvd_var_ratios = tsvd.explained_variance_ratio_
        
        # function to determin the n_components for svd that covers 90% of the variance 
        def select_n_components(var_ratio, goal_var: float) -> int:
            total_variance = 0.0             
            n_comp = 0            
            # Loops through adding the variance explained by each extra component
            for explained_variance in var_ratio:        
                total_variance += explained_variance        
                n_comp += 1        
                if total_variance >= goal_var:
                    # Breaks once required variance is explained 
                    break            
            # Returns the number of components
            return n_comp
                
        # Running function for number of components
        n_comp = select_n_components(tsvd_var_ratios, 0.90)
                 
         
        #transpose matrix so movies are represented by rows and users by columns 
        #HX = ratingsPivot1.values.T
        #using truncated SVD to condense all of 1360 user reviews in 12 latent variables . Random_state set to 1 to make results repeatable 
        SVD = TruncatedSVD(n_components=n_comp, random_state=1)
        #preforming SVD on X; the transposed ratings 
        resultant_matrix = SVD.fit_transform(ratingsPivot)
        #making a matrix of correlation between all beers
        corr_mat = np.corrcoef(resultant_matrix)
        
        #taking the column containing the users inputted beer
        beer_names= ratingsPivot1.columns
        beer_list = list(beer_names)
        inputBeer = beer_list.index(user_input)
        corInput = corr_mat[inputBeer]
        #combing the user inputted beer column and the correlations 
        recommendations = pd.DataFrame(corInput, beer_names, columns=['Correlation'])
        #only including correlations less than 1 so the user inputed beer is not recommended 
        recommendations = recommendations[recommendations.Correlation<0.999999]
        #taking the n top correlated beers
        recommendations = recommendations.sort_values('Correlation', ascending=False).head(num_recommendations)
        
        print("\nSVD Recommendations\n")
        print(recommendations)
        
        
        
        ################################################
        ################################################
        ##############    Pearson    ###################
        ################################################
        ################################################
        
        ################################################
        # Correlation Coefficienct calculated manually #
        ################################################
        
        ratings = data[['beer_beerid', 'review_profilename', 'review_overall']]
        ratings1 = ratings.head(n=5000)
        
        # function to calculate pearson correlation between two values
        def pearson(s1, s2):
            s1_c = s1 - s1.mean()
            s2_c = s2 - s2.mean()
            return np.sum(s1_c * s2_c) / (np.sqrt(np.sum(s1_c ** 2)) * np.sqrt(np.sum(s2_c ** 2)))
        
        # function to return recommendations given a beer name, pivot table of ratings, and number of recommendations to be shown
        # creates empty array of reviews
        # loops through beers in system - if beer name is not user input, use previously defined function to calculate pearson correlation between the two
        # append beer names and correlation between that beer and user input beer (provided correlation is not null)
        # sort in decreasing order of correlation and return top num recommendations
        def get_recs(beer_name, ratingsPivot, num):
            
            reviews = []
            for name in ratingsPivot.columns:
                if name == beer_name:
                    continue
                cor = pearson(ratingsPivot[beer_name], ratingsPivot[name])
                if np.isnan(cor):
                    continue
                else:
                    reviews.append((name, cor))
                    
                    
            reviews.sort(key=lambda tup: tup[1], reverse=True)
            return reviews[:num]
              
        # use function above to get user entered number of recommendations for user input beer        
        recs = get_recs(user_input , ratingsPivot1, num_recommendations)
        # store recommendations in data frame with two columns to print neatly
        recs = pd.DataFrame(recs, columns=['Beer','Correlation'])        
        
        print("\nPearson Correlation Method 1 Recommendations\n")
        print(recs)
        
        
        ################################################
        ###### Using inbuilt corrwith() function #######
        ################################################
        
        # Also pearson but using inbuilt function - same outcome
        
        #mypivot = ratingUserPopular.pivot_table(index=['review_profilename'],columns=['beer_name'],values='review_overall',fill_value=0)
        
        # get ratings associated with user input beer
        input_beer_ratings = ratingsPivot1[user_input]
        
        # find correlation between this beer and other beers using these ratings
        beers_like_input = ratingsPivot1.corrwith(input_beer_ratings)
        
        # store beers and pearson correlation in a dataframe
        corr_inputBeer = pd.DataFrame(beers_like_input, columns=['Correlation'])
        # remove any results with correlation > 0.99999 i.e. beers that are perfectly correlated, this is likely the beer input by the user, which we don't want to recommend
        corr_inputBeer = corr_inputBeer[corr_inputBeer.Correlation<0.999999]
        
        # sort the correlation values in descending order and retain only the top user entered number of recommendations
        corr_inputBeer = corr_inputBeer.sort_values('Correlation', ascending=False).head(num_recommendations)
        
        print("\nPearson Correlation Method 2 Recommendations\n")
        # print top N recommended beers and their correlation with user input beer
        print(corr_inputBeer)
        
        
        # ask user if they would like to find recommendations based on another beer
        # if they enter some variation of 'no', exit the loop, terminating the programme
        # otherwise while loop will continue to run and ask user to enter a new beer
        user_again = input("Would you like to enter a new beer? (Y/N)")
        if user_again == "N" or user_again == "n" or user_again == "No" or user_again == "no":
            quit = True
            
        
        