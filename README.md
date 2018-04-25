# Machine Learning with python

I want to do my first steps with AI...

    ... it's the well known Iris flower data set. The Hello World of AI - they say ;)

Following this tutorial: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
I decided to document a bit the topics handled. It should at first be a help for me. But if it helps someone else I'm happy too.

## Used Libraries
pandas: Library for Data-Analysis
    Reading datasets from url or file and analyze the data with different tools which comes with panda

matplotlib: Library for visualization of data
    Make diagrams from data offered. For example from a dataset from pandas

sklearn: AI-Library
    Offers ready to go algorithms for machine learning in form of data-mining and analysing tools

## Used Algorithms

## What makes iristutorial.py
First I do some normal python imports. Then I'm taking a finished dataset offered by the tutorial with data for the iris flower. The first four values are width and length of sepal and petal (Google it ;)... it's beautiful). Then I'm testing the pandas-library. This Library makes it really easy to evaluate datasets. I did not read a lot about how it works but it has started very good and I need to hold it in my brain for maths and physics. After the data-analysation I tried to visualize the data with the Library matplotlib. Same here; really cool but not read a lot about till know. Maybe later. After the visualization we're slowly coming to the AI stuff. First I create two datasets of my basic-dataset. One holds 20% of Data and the other 80%. The 20% I'm going to use to check the results of my trained AI. The 80%-Part is the training-data for my AI. After splitting it up I'm running tests with several algorithms listet above (doc coming later). With the help of the accuracy I can find out which algorithm is the best.