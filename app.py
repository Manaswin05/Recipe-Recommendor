from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import pickle
import os

app = Flask(__name__)

# Load CSV data
data = pd.read_csv('abc.csv')

class RecipeRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            max_features=500
        )
        self.ingredient_matrix = None
        self.recipes_df = None
        
    def fit(self, recipes_df):
        """Train the model on recipe data"""
        self.recipes_df = recipes_df
        
        # Combine ingredients into text for TF-IDF
        ingredient_texts = recipes_df['ingredients'].apply(
            lambda x: ' '.join([ing.strip().lower() for ing in str(x).split(',')])
        )
        
        # Create TF-IDF matrix
        self.ingredient_matrix = self.vectorizer.fit_transform(ingredient_texts)
        
        print(f"Model trained on {len(recipes_df)} recipes")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
    def predict(self, user_ingredients, top_n=10, difficulty_filter=None, cuisine_filter=None):
        """Recommend recipes based on user ingredients"""
        # Convert user ingredients to same format
        user_text = ' '.join([ing.strip().lower() for ing in user_ingredients])
        
        # Transform user input using trained vectorizer
        user_vector = self.vectorizer.transform([user_text])
        
        # Calculate cosine similarity between user input and all recipes
        similarities = cosine_similarity(user_vector, self.ingredient_matrix).flatten()
        
        # Get recipe indices sorted by similarity
        recipe_indices = similarities.argsort()[::-1]
        
        # Filter and collect recommendations
        recommendations = []
        for idx in recipe_indices:
            if len(recommendations) >= top_n:
                break
                
            recipe = self.recipes_df.iloc[idx]
            similarity_score = similarities[idx]
            
            # Skip if no similarity
            if similarity_score == 0:
                continue
            
            # Apply filters
            if difficulty_filter and str(recipe.get('difficulty', '')).lower() != difficulty_filter.lower():
                continue
            if cuisine_filter and str(recipe.get('cuisine', '')).lower() != cuisine_filter.lower():
                continue
            
            # Calculate additional metrics
            recipe_ingredients = [ing.strip().lower() for ing in str(recipe['ingredients']).split(',')]
            user_set = set([ing.lower() for ing in user_ingredients])
            recipe_set = set(recipe_ingredients)
            
            exact_matches = len(user_set.intersection(recipe_set))
            missing_ingredients = len(recipe_set - user_set)
            
            recommendations.append({
                'recipe': recipe,
                'similarity_score': float(similarity_score),
                'exact_matches': exact_matches,
                'missing_ingredients': missing_ingredients
            })
        
        return recommendations
    
    def save_model(self, filepath='recipe_model.pkl'):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'ingredient_matrix': self.ingredient_matrix,
                'recipes_df': self.recipes_df
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='recipe_model.pkl'):
        """Load trained model"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                self.vectorizer = model_data['vectorizer']
                self.ingredient_matrix = model_data['ingredient_matrix']
                self.recipes_df = model_data['recipes_df']
            print(f"Model loaded from {filepath}")
            return True
        return False

# Initialize and train the recommender
recommender = RecipeRecommender()

# Try to load existing model, otherwise train new one
if not recommender.load_model():
    print("Training new model...")
    recommender.fit(data)
    recommender.save_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_recipe', methods=['POST'])
def get_recipe():
    try:
        # Get parameters from request
        ingredients_input = request.json.get('ingredients', '').strip().lower()
        difficulty_filter = request.json.get('difficulty', None)
        cuisine_filter = request.json.get('cuisine', None)
        top_n = request.json.get('top_n', 10)
        
        print(f"Received request - Ingredients: {ingredients_input}")
        
        if not ingredients_input:
            return jsonify({'error': 'Please enter some ingredients'}), 400
        
        # Parse user ingredients
        user_ingredients = [ing.strip() for ing in ingredients_input.split(',')]
        
        # Get recommendations using ML model
        recommendations = recommender.predict(
            user_ingredients, 
            top_n=top_n,
            difficulty_filter=difficulty_filter,
            cuisine_filter=cuisine_filter
        )
        
        if not recommendations:
            return jsonify({'error': 'No recipes found matching your ingredients. Try different ingredients!'}), 404
        
        # Format response
        result_recipes = []
        for rec in recommendations:
            recipe = rec['recipe']
            
            # Split ingredients and instructions
            ingredients_list = [ing.strip() for ing in str(recipe['ingredients']).split(',')]
            instructions_list = [step.strip() for step in str(recipe['instructions']).split('.') if step.strip()]
            
            result_recipes.append({
                'title': str(recipe['name']),
                'ingredients': ingredients_list,
                'instructions': instructions_list,
                'time': f"{recipe.get('prep_time', 15) + recipe.get('cook_time', 15)} mins",
                'cuisine': str(recipe.get('cuisine', 'Indian')),
                'prepTime': f"{recipe.get('prep_time', 15)} mins",
                'cookTime': f"{recipe.get('cook_time', 15)} mins",
                'difficulty': str(recipe.get('difficulty', 'Medium')),
                'matchScore': round(rec['similarity_score'] * 100, 1),
                'exactMatches': rec['exact_matches'],
                'missingIngredients': rec['missing_ingredients']
            })
        
        print(f"Returning {len(result_recipes)} recipes")
        print(f"Top recipe: {result_recipes[0]['title']} (ML Score: {result_recipes[0]['matchScore']}%)")
        
        return jsonify(result_recipes)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    """Retrain the ML model with current data"""
    try:
        global recommender
        recommender = RecipeRecommender()
        recommender.fit(data)
        recommender.save_model()
        return jsonify({'message': 'Model retrained successfully', 'recipes': len(data)})
    except Exception as e:
        return jsonify({'error': f'Failed to retrain: {str(e)}'}), 500

@app.route('/get_all_recipes', methods=['GET'])
def get_all_recipes():
    """Get all available recipes"""
    try:
        all_recipes = []
        
        for idx, row in data.iterrows():
            ingredients_list = [ing.strip() for ing in str(row['ingredients']).split(',')]
            instructions_list = [step.strip() for step in str(row['instructions']).split('.') if step.strip()]
            
            recipe = {
                'title': str(row['name']),
                'ingredients': ingredients_list,
                'instructions': instructions_list,
                'time': f"{row.get('prep_time', 15) + row.get('cook_time', 15)} mins",
                'cuisine': str(row.get('cuisine', 'Indian')),
                'prepTime': f"{row.get('prep_time', 15)} mins",
                'cookTime': f"{row.get('cook_time', 15)} mins",
                'difficulty': str(row.get('difficulty', 'Medium'))
            }
            all_recipes.append(recipe)
        
        return jsonify(all_recipes)
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/get_popular_ingredients', methods=['GET'])
def get_popular_ingredients():
    """Get most common ingredients across all recipes"""
    try:
        all_ingredients = []
        
        for idx, row in data.iterrows():
            ingredients = [ing.strip().lower() for ing in str(row['ingredients']).split(',')]
            all_ingredients.extend(ingredients)
        
        # Count frequency
        ingredient_counts = Counter(all_ingredients)
        popular = [{'ingredient': ing, 'count': count} 
                   for ing, count in ingredient_counts.most_common(15)]
        
        return jsonify(popular)
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/get_cuisines', methods=['GET'])
def get_cuisines():
    """Get list of available cuisines"""
    try:
        cuisines = data['cuisine'].unique().tolist()
        return jsonify(cuisines)
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/get_recipe_stats', methods=['GET'])
def get_recipe_stats():
    """Get statistics about the recipe database"""
    try:
        stats = {
            'total_recipes': len(data),
            'cuisines': data['cuisine'].nunique(),
            'avg_prep_time': int(data['prep_time'].mean()),
            'avg_cook_time': int(data['cook_time'].mean()),
            'difficulty_distribution': data['difficulty'].value_counts().to_dict(),
            'model_vocabulary_size': len(recommender.vectorizer.vocabulary_)
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/get_model_info', methods=['GET'])
def get_model_info():
    """Get information about the ML model"""
    try:
        info = {
            'algorithm': 'TF-IDF + Cosine Similarity',
            'vectorizer': 'TfidfVectorizer',
            'vocabulary_size': len(recommender.vectorizer.vocabulary_),
            'trained_on': len(recommender.recipes_df),
            'ngram_range': '(1, 2)',
            'max_features': 500
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
