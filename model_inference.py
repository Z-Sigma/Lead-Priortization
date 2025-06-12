"""
Model inference module for lead prioritization
"""
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sklearn.preprocessing import MinMaxScaler
from rl_environment import LeadPrioritizationEnv


class LeadPrioritizationInference:
    """Handle model loading and inference for lead prioritization"""
    
    def __init__(self, model_path, pipeline_path):
        self.model_path = model_path
        self.pipeline_path = pipeline_path
        self.model = None
        self.pipeline = None
        self.relevance_scaler = MinMaxScaler()
        
    def load_model_and_pipeline(self):
        """Load the trained SAC model and data pipeline"""
        print("Loading data pipeline...")
        from data_pipeline import LeadDataPipeline
        self.pipeline = LeadDataPipeline.load(self.pipeline_path)
        
        print("Loading SAC model...")
        # Create dummy environment for model loading
        dummy_features = np.random.randn(1, 433).astype(np.float32)  # Adjust shape as needed
        dummy_relevance = np.array([0.5])
        dummy_env = LeadPrioritizationEnv(dummy_features, dummy_relevance)
        monitored_env = Monitor(dummy_env)
        vec_env = DummyVecEnv([lambda: monitored_env])
        
        self.model = SAC.load(self.model_path, env=vec_env)
        print("✅ Model and pipeline loaded successfully")
        
    def preprocess_data(self, df):
        """Preprocess raw data using the loaded pipeline"""
        if self.pipeline is None:
            raise ValueError("Pipeline not loaded. Call load_model_and_pipeline() first.")
            
        # Transform data using pipeline
        X, df_clean = self.pipeline.transform(df)
        
        # Normalize relevance scores separately for environment
        df_clean['relevance_score_norm'] = self.relevance_scaler.fit_transform(
            df_clean[['relevance_score']]
        )
        
        return X, df_clean
        
    def predict_priorities(self, X, df_clean):
        """Generate priority predictions for leads"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_pipeline() first.")
            
        # Create environment for inference
        env = LeadPrioritizationEnv(
            X.astype(np.float32), 
            df_clean['relevance_score_norm'].values
        )
        monitored_env = Monitor(env)
        vec_env = DummyVecEnv([lambda: monitored_env])
        
        # Generate predictions
        prioritized_leads = []
        obs = vec_env.reset()
        
        for i in range(len(df_clean)):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _ = vec_env.step(action)
            
            action_value = float(action[0][0])
            relevance = df_clean['relevance_score_norm'].iloc[i]
            original_relevance = df_clean['relevance_score'].iloc[i]
            domain_name = df_clean['domain'].iloc[i]
            site = df_clean['lead_site'].iloc[i]
            summary = df_clean['summary'].iloc[i]
            
            prioritized_leads.append({
                'index': i,
                'priority_score': action_value,
                'normalized_relevance': relevance,
                'original_relevance': original_relevance,
                'domain': domain_name,
                'lead_site': site,
                'summary': summary
            })
            
            if done:
                break
                
        return prioritized_leads
        
    def get_top_leads(self, prioritized_leads, top_n=20):
        """Get top N prioritized leads sorted by priority score"""
        # Sort by priority score (high to low)
        sorted_leads = sorted(prioritized_leads, key=lambda x: x['priority_score'], reverse=True)
        return sorted_leads[:top_n]
        
    def categorize_leads(self, prioritized_leads):
        """Categorize leads based on priority scores"""
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for lead in prioritized_leads:
            priority = lead['priority_score']
            if priority >= 0.8:
                high_priority.append(lead)
            elif priority >= 0.5:
                medium_priority.append(lead)
            else:
                low_priority.append(lead)
                
        return {
            'high_priority': high_priority,
            'medium_priority': medium_priority,
            'low_priority': low_priority
        }
        
    def generate_summary_report(self, prioritized_leads):
        """Generate a summary report of the prioritization results"""
        categorized = self.categorize_leads(prioritized_leads)
        
        total_leads = len(prioritized_leads)
        high_count = len(categorized['high_priority'])
        medium_count = len(categorized['medium_priority'])
        low_count = len(categorized['low_priority'])
        
        avg_priority = np.mean([lead['priority_score'] for lead in prioritized_leads])
        avg_relevance = np.mean([lead['original_relevance'] for lead in prioritized_leads])
        
        report = {
            'total_leads': total_leads,
            'high_priority_count': high_count,
            'medium_priority_count': medium_count,
            'low_priority_count': low_count,
            'high_priority_percentage': (high_count / total_leads) * 100,
            'medium_priority_percentage': (medium_count / total_leads) * 100,
            'low_priority_percentage': (low_count / total_leads) * 100,
            'average_priority_score': avg_priority,
            'average_relevance_score': avg_relevance,
            'categorized_leads': categorized
        }
        
        return report