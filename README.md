# IGB-Greyhound-Position-Prediction
This objective of this project is to predict the position of the greyhound based on the other fields and estimated time taken by the greyhound to finish the race.
The workflow of the project is as follows:
1. **Acquire data:**
Make http requests to the website, to download the html data
2. **Data Cleaning:**
Data is found under various tags in the HTML file. So the data is extracted from the file and cleaned. This finally gives us the data we can work on.
3. **Greyhound Position Prediction:**
We are predicting the top 3 Greyhound in the race using other columns. This problem is a classification problem. The prediction is done using ANN and CNN. The ANN was giving an accuracy of 87% and CNN 76%.
4. **Estimating the race finish time of Greyhound:**
We are estimating the race finish time of Greyhound using ANN.

## Technologies Used

### Libraries:
* pandas
* urllib3
* Keras
* Numpy
* sklearn
* matplotlib.pyplot

### Requirements:
* Python 3.6

Data Source:
https://www.igb.ie/
