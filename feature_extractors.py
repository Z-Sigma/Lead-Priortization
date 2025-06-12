"""
Feature extraction modules for lead prioritization
"""
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeadSiteFeatureExtractor:
    """Feature extractor for lead_site column"""

    def __init__(self):
        self.keywords = [
            'site', 'web', 'portal', 'online', 'app', 'shop', 'store', 'blog', 'service',
            'cloud', 'api', 'dashboard', 'mobile', 'login', 'account', 'admin', 'support',
            'help', 'pay', 'secure', 'tech', 'data', 'market', 'sales', 'user', 'client'
        ]
        self.tlds = ['.com', '.net', '.org', '.io', '.tech', '.co', '.biz', '.info', '.app']

    def extract_features(self, lead_site: str) -> np.ndarray:
        """Extract features from a single lead_site string"""
        features = []
        length = len(lead_site)
        features.append(length)

        # Digits
        digit_count = sum(c.isdigit() for c in lead_site)
        features.append(digit_count)
        features.append(digit_count / length if length > 0 else 0)

        # Special chars (excluding '.' and '-')
        special_char_count = len(re.findall(r'[^a-zA-Z0-9.-]', lead_site))
        features.append(special_char_count)
        features.append(special_char_count / length if length > 0 else 0)

        # Hyphens count
        hyphen_count = lead_site.count('-')
        features.append(hyphen_count)

        # Uppercase letters count and ratio
        uppercase_count = sum(c.isupper() for c in lead_site)
        features.append(uppercase_count)
        features.append(uppercase_count / length if length > 0 else 0)

        # Keyword count (total occurrences of all keywords)
        keyword_occurrences = sum(lead_site.lower().count(k) for k in self.keywords)
        features.append(keyword_occurrences)

        # Binary flag if any keyword present
        features.append(1 if keyword_occurrences > 0 else 0)

        # Count dots (subdomains)
        dot_count = lead_site.count('.')
        features.append(dot_count)

        # Longest segment length between dots
        segments = lead_site.split('.')
        longest_segment = max(len(seg) for seg in segments) if segments else 0
        features.append(longest_segment)

        # Ends with common TLD (one-hot for each)
        for tld in self.tlds:
            features.append(1 if lead_site.endswith(tld) else 0)

        # Looks like URL (contains http, https, www)
        features.append(1 if any(x in lead_site.lower() for x in ['http', 'https', 'www']) else 0)

        return np.array(features, dtype=np.float32)


class DomainFeatureExtractor:
    """Feature extractor for domain column"""

    def extract_features(self, domain: str) -> np.ndarray:
        """Extract features from domain column"""
        features = []
        features.append(len(domain))  # domain length
        features.append(1 if domain.endswith('.ai') else 0)  # .ai TLD

        tech_keywords = ['tech', 'ai', 'ml', 'data', 'cloud', 'app', 'io', 'dev']
        features.append(1 if any(k in domain.lower() for k in tech_keywords) else 0)
        features.append(1 if any(char.isdigit() for char in domain) else 0)  # has digits
        features.append(1 if re.search(r'[^a-zA-Z0-9.-]', domain) else 0)  # special chars

        return np.array(features, dtype=np.float32)


class DomainLeadSiteAttention(nn.Module):
    """Attention mechanism for combining domain and lead_site features"""

    def __init__(self, domain_dim, lead_site_dim, proj_dim=24):
        super().__init__()
        self.proj_dim = proj_dim
        self.domain_proj = nn.Linear(domain_dim, proj_dim)
        self.lead_site_proj = nn.Linear(lead_site_dim, proj_dim)
        self.query = nn.Parameter(torch.randn(proj_dim))

    def forward(self, domain_features, lead_site_features):
        domain_p = self.domain_proj(domain_features)
        lead_site_p = self.lead_site_proj(lead_site_features)

        combined = torch.stack([domain_p, lead_site_p], dim=1)

        batch_size = domain_features.size(0)
        query = self.query.expand(batch_size, -1).unsqueeze(2)

        attn_scores = torch.bmm(combined, query).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)

        attended_vector = torch.sum(combined * attn_weights.unsqueeze(-1), dim=1)

        return attended_vector, attn_weights