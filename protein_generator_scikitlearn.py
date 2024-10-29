import random as rand 
import pandas as pd
from sklearn.metrics import  ConfusionMatrixDisplay, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix

#set seed for reproducibility
rand.seed(99)

# Generate synthetic protein sequences based off of standard 20 AA's
def generate_random_protein_seq(length):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # Standard 20 amino acids
    return ''.join(rand.choice(amino_acids) for _ in range(length))


# Generate a list of protein sequences
protein_seqs = [generate_random_protein_seq(3) for _ in range(5)]

# Patient Dataset Generator Function

def gen_patient_dataset():
    healthySick = ["PD", "PRCR"]
    data=[]
    prognosis= rand.choices(healthySick, k=1)
    
    for _ in range(10):
        prognosis = rand.choice(healthySick)  # Randomly assign a prognosis
        randID= rand.randint(1, 100)

        # Create a dictionary for the patient record
        patient_record = {
            'ID': randID,
            'Pathology': prognosis,
           protein_seqs[0] : rand.randint(1, 1000),
           protein_seqs[1] : rand.randint(1, 1000),
           protein_seqs[2] : rand.randint(1, 1000)
        }
        
        data.append(patient_record)  # Append to the dataset
    
    # Convert the list of patient records to a DataFrame (optional)
    patient_df = pd.DataFrame(data)
    return patient_df


# Generate the dataset
patient_dataset = gen_patient_dataset()

# Print patient Df for reference 
print(patient_dataset)

# Encode the Pathology column into binary form 
label_encoder = LabelEncoder()
patient_dataset['Pathology'] = label_encoder.fit_transform(patient_dataset['Pathology'])


Pathos = patient_dataset["Pathology"]
Counts = patient_dataset.drop(columns = ["ID", "Pathology"])
# Drop ID column for logistic regression input
patient_dataset.drop(columns = ["ID"], inplace = True)


# Create logistic regression classifier
# set X and Y Axes
X, y = Counts, Pathos   

# Initialize model
model = LogisticRegression(random_state=0).fit(X, y)

# Acquire predictions and decode them
preds = model.predict(X.iloc[:2, :])
predictions = label_encoder.inverse_transform(preds)

# Print Predictions & Accuracu
print("Predictions for the first two rows: ", predictions)

accuracy = model.score(X, y)

print("Model Accuracy: ",accuracy)

#Create Scatter Matrix
scatter_matrix(X, alpha=0.7, figsize=(10,10), diagonal='hist')
plt.suptitle("Scatter Matrix of Protein Sequence Counts")
plt.show()

def ROC_curve_create(model, X, y):
    #Function to plot ROC curve
    
    # Predict the probabilities of the positive class
    y_prob = model.predict_proba(X)[:, 1]  # Probability for class 1

    # Compute ROC curve and AUC
    # fpr = false positive rate, tpr = true positive rate
    fpr, tpr, thresholds = roc_curve(y, y_prob)  
    # Calculate the area under the curve (AUC)
    roc_auc = auc(fpr, tpr)  

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

ROC_curve_create(model, X, y)

def Confusion_matrix_create(model, X, y, display_labels= ["PD", "PRCR"]):
    # Create Confusion Matrix Display
    ConfusionMatrixDisplay.from_estimator(
        estimator=model,         # Your trained logistic regression model
        X=X,                     # The features (input data)
        y=y,                     # The true labels
        display_labels= display_labels,  # Display labels for each class
        cmap="viridis",          # Color map for the matrix
        colorbar=True            # Display a colorbar alongside the matrix
        )
    # Set custom axis labels
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    # Print Confusion Matrix
    plt.title("Confusion Matrix Display for Logistic Regression Model")
    plt.show()

Confusion_matrix_create(model, X, y, ["PD", "PRCR"])


