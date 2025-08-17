# RecipeRanker
# Recipe Popularity Predictor

An end-to-end *machine learning web app* built with *Streamlit* that predicts how popular a recipe will be, based on features like reputation, ratings, engagement, and nutrition.  

The model classifies recipes into three categories:
- *Typical*: Score = 100 (majority of recipes)  
- *Popular*: Score 101–253 (higher than average)  
- *Viral*: Score >253 (top 10% rare recipes)  

---

## Features
- *ML Pipeline*: Custom feature engineering + preprocessing + Random Forest classifier  
- *Explainability*: Feature importance table for each prediction  
- *Interactive UI*: Built with Streamlit for easy recipe input  
- *Visualization*: Probability breakdown chart + confusion matrix from validation set  
- *Deployment Ready*: Configured for Streamlit Cloud / Render hosting  

---

## Tech Stack
- *Python 3.11+*  
- *Libraries*:  
  - [Streamlit](https://streamlit.io/) – Web app UI  
  - [scikit-learn](https://scikit-learn.org/) – ML pipeline + RandomForestClassifier  
  - [pandas / numpy](https://pandas.pydata.org/) – Data handling  
  - [matplotlib](https://matplotlib.org/) – Charts  
  - [joblib](https://joblib.readthedocs.io/) – Model persistence  

---

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/recipe-popularity-predictor.git
   cd recipe-popularity-predictor
2. Create a virtual environment:
   -python -m venv venv
   -source venv/bin/activate   # On Mac/Linux
   -venv\Scripts\activate      # On Windows

3. Install dependencies:<br>
   -pip install -r requirements.txt<br>

4. Run the Streamlit app:<br>
   -streamlit run app.py<br>
5. Live Demo<br>

