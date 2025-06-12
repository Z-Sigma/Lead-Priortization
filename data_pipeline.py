"""
Data preprocessing pipeline for lead prioritization
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from sentence_transformers import SentenceTransformer
import joblib
from feature_extractors import LeadSiteFeatureExtractor, DomainFeatureExtractor, DomainLeadSiteAttention


class LeadDataPipeline:
    """Complete data preprocessing pipeline for lead data"""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.lead_site_extractor = LeadSiteFeatureExtractor()
        self.domain_extractor = DomainFeatureExtractor()
        self.relevance_scaler = MinMaxScaler()
        self.lead_site_scaler = StandardScaler()
        self.domain_scaler = StandardScaler()
        self.sentence_transformer = SentenceTransformer(model_name)
        self.attention_model = None
        self.is_fitted = False

        # Store feature dimensions for attention model
        self.domain_dim = 5
        self.lead_site_dim = 22
        self.proj_dim = 24

    def _clean_data(self, df):
        """Clean and prepare data"""
        # Create copy and drop unnecessary columns
        df_clean = df.copy()
        columns_to_drop = ['response', 'portfolio', 'sale_price', 'reserve']
        df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')

        # Remove null values
        df_clean = df_clean.dropna()

        return df_clean

    def fit(self, df):
        """Fit the pipeline on training data"""
        print("Fitting pipeline...")

        # Clean data
        df_clean = self._clean_data(df)

        # Fit relevance scaler
        self.relevance_scaler.fit(df_clean[['relevance_score']])

        # Extract and fit lead_site features
        lead_site_features = np.array([
            self.lead_site_extractor.extract_features(site)
            for site in df_clean['lead_site']
        ])
        self.lead_site_scaler.fit(lead_site_features)

        # Extract and fit domain features
        domain_features = np.array([
            self.domain_extractor.extract_features(domain)
            for domain in df_clean['domain']
        ])
        self.domain_scaler.fit(domain_features)

        # Initialize and fit attention model
        self.attention_model = DomainLeadSiteAttention(
            self.domain_dim, self.lead_site_dim, self.proj_dim
        )

        # Dummy forward pass to initialize parameters properly
        domain_t = torch.tensor(self.domain_scaler.transform(domain_features), dtype=torch.float32)
        lead_site_t = torch.tensor(self.lead_site_scaler.transform(lead_site_features), dtype=torch.float32)
        _, _ = self.attention_model(domain_t, lead_site_t)

        self.is_fitted = True
        print("Pipeline fitted successfully!")

        return self

    def transform(self, df):
        """Transform data using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform. Call fit() first.")

        print("Transforming data...")

        # Clean data
        df_clean = self._clean_data(df)

        # 1. Normalize relevance score
        relevance_norm = self.relevance_scaler.transform(df_clean[['relevance_score']])

        # 2. Extract and normalize lead_site features
        lead_site_features = np.array([
            self.lead_site_extractor.extract_features(site)
            for site in df_clean['lead_site']
        ])
        lead_site_normalized = self.lead_site_scaler.transform(lead_site_features)

        # 3. Extract and normalize domain features
        domain_features = np.array([
            self.domain_extractor.extract_features(domain)
            for domain in df_clean['domain']
        ])
        domain_normalized = self.domain_scaler.transform(domain_features)

        # 4. Generate summary embeddings
        print("Generating summary embeddings...")
        summary_embeddings = self.sentence_transformer.encode(
            df_clean['summary'].tolist(),
            show_progress_bar=True
        )

        # 5. Apply attention mechanism
        domain_t = torch.tensor(domain_normalized, dtype=torch.float32)
        lead_site_t = torch.tensor(lead_site_normalized, dtype=torch.float32)

        with torch.no_grad():
            attended_vec, _ = self.attention_model(domain_t, lead_site_t)
            attended_vec_np = attended_vec.numpy()

        # 6. Combine all features into final X matrix
        X = np.hstack([
            relevance_norm,           # relevance score (normalized)
            summary_embeddings,       # summary embeddings
            attended_vec_np,         # attended domain+lead_site features
        ])

        print(f"Final feature matrix shape: {X.shape}")

        return X.astype(np.float32), df_clean

    def fit_transform(self, df):
        """Fit pipeline and transform data in one step"""
        return self.fit(df).transform(df)

    def save(self, filepath):
        """Save the fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving.")

        pipeline_data = {
            'relevance_scaler': self.relevance_scaler,
            'lead_site_scaler': self.lead_site_scaler,
            'domain_scaler': self.domain_scaler,
            'attention_model_state': self.attention_model.state_dict(),
            'sentence_transformer_name': self.sentence_transformer._modules['0'].auto_model.name_or_path,
            'is_fitted': self.is_fitted,
            'domain_dim': self.domain_dim,
            'lead_site_dim': self.lead_site_dim,
            'proj_dim': self.proj_dim
        }

        joblib.dump(pipeline_data, filepath)
        print(f"Pipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load a fitted pipeline"""
        pipeline_data = joblib.load(filepath)

        # Create new instance
        pipeline = cls(model_name=pipeline_data['sentence_transformer_name'])

        # Load fitted components
        pipeline.relevance_scaler = pipeline_data['relevance_scaler']
        pipeline.lead_site_scaler = pipeline_data['lead_site_scaler']
        pipeline.domain_scaler = pipeline_data['domain_scaler']
        pipeline.is_fitted = pipeline_data['is_fitted']
        pipeline.domain_dim = pipeline_data['domain_dim']
        pipeline.lead_site_dim = pipeline_data['lead_site_dim']
        pipeline.proj_dim = pipeline_data['proj_dim']

        # Recreate and load attention model
        pipeline.attention_model = DomainLeadSiteAttention(
            pipeline.domain_dim, pipeline.lead_site_dim, pipeline.proj_dim
        )
        pipeline.attention_model.load_state_dict(pipeline_data['attention_model_state'])

        print(f"Pipeline loaded from {filepath}")
        return pipeline