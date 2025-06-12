"""
Output formatting and saving utilities for lead prioritization results
"""
import pandas as pd
import json
from datetime import datetime
import os


class OutputFormatter:
    """Format and save prioritization results in various formats"""
    
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def save_detailed_results_csv(self, prioritized_leads, filename=None):
        """Save detailed results to CSV file"""
        if filename is None:
            filename = f"lead_prioritization_detailed_{self.timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to DataFrame for easy CSV export
        df_results = pd.DataFrame(prioritized_leads)
        df_results = df_results.sort_values('priority_score', ascending=False).reset_index(drop=True)
        
        # Add rank column
        df_results['rank'] = range(1, len(df_results) + 1)
        
        # Reorder columns for better readability
        columns_order = [
            'rank', 'index', 'priority_score', 'original_relevance', 
            'normalized_relevance', 'domain', 'lead_site', 'summary'
        ]
        df_results = df_results[columns_order]
        
        df_results.to_csv(filepath, index=False)
        print(f"Detailed results saved to: {filepath}")
        return filepath
    
    def save_top_leads_csv(self, top_leads, filename=None):
        """Save top prioritized leads to CSV"""
        if filename is None:
            filename = f"top_prioritized_leads_{self.timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to DataFrame
        df_top = pd.DataFrame(top_leads)
        df_top['rank'] = range(1, len(df_top) + 1)
        
        # Select key columns for top leads
        columns_order = [
            'rank', 'priority_score', 'original_relevance', 
            'domain', 'lead_site', 'summary'
        ]
        df_top = df_top[columns_order]
        
        df_top.to_csv(filepath, index=False)
        print(f"Top leads saved to: {filepath}")
        return filepath
    
    def save_summary_report_json(self, summary_report, filename=None):
        """Save summary report to JSON file"""
        if filename is None:
            filename = f"prioritization_summary_{self.timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare report for JSON serialization
        json_report = summary_report.copy()
        
        # Convert categorized leads to basic info only for JSON
        for category in ['high_priority', 'medium_priority', 'low_priority']:
            json_report['categorized_leads'][category] = [
                {
                    'domain': lead['domain'],
                    'priority_score': lead['priority_score'],
                    'original_relevance': lead['original_relevance']
                }
                for lead in json_report['categorized_leads'][category]
            ]
        
        with open(filepath, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        print(f"Summary report saved to: {filepath}")
        return filepath
    
    def print_summary_report(self, summary_report):
        """Print a formatted summary report to console"""
        print("\n" + "="*60)
        print("LEAD PRIORITIZATION SUMMARY REPORT")
        print("="*60)
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Leads Processed: {summary_report['total_leads']}")
        print(f"   Average Priority Score: {summary_report['average_priority_score']:.3f}")
        print(f"   Average Relevance Score: {summary_report['average_relevance_score']:.3f}")
        
        print(f"\n🎯 PRIORITY DISTRIBUTION:")
        print(f"   High Priority (≥0.8): {summary_report['high_priority_count']} ({summary_report['high_priority_percentage']:.1f}%)")
        print(f"   Medium Priority (0.5-0.8): {summary_report['medium_priority_count']} ({summary_report['medium_priority_percentage']:.1f}%)")
        print(f"   Low Priority (<0.5): {summary_report['low_priority_count']} ({summary_report['low_priority_percentage']:.1f}%)")
    
    def print_top_leads(self, top_leads, show_count=10):
        """Print top prioritized leads to console"""
        print(f"\n🔝 TOP {show_count} PRIORITIZED LEADS:")
        print("-" * 100)
        print(f"{'Rank':<5} {'Priority':<9} {'Relevance':<10} {'Domain':<25} {'Lead Site':<30}")
        print("-" * 100)
        
        for i, lead in enumerate(top_leads[:show_count], 1):
            domain = lead['domain'][:24] + "..." if len(lead['domain']) > 24 else lead['domain']
            site = lead['lead_site'][:29] + "..." if len(lead['lead_site']) > 29 else lead['lead_site']
            
            print(f"{i:<5} {lead['priority_score']:<9.3f} {lead['original_relevance']:<10.3f} {domain:<25} {site:<30}")
    
    def save_all_outputs(self, prioritized_leads, summary_report, top_leads):
        """Save all output formats at once"""
        print(f"\n💾 Saving all outputs to '{self.output_dir}' directory...")
        
        files_saved = []
        
        # Save detailed results
        files_saved.append(self.save_detailed_results_csv(prioritized_leads))
        
        # Save top leads
        files_saved.append(self.save_top_leads_csv(top_leads))
        
        # Save summary report
        files_saved.append(self.save_summary_report_json(summary_report))
        
        print(f"\n✅ All outputs saved successfully!")
        print("Files created:")
        for file in files_saved:
            print(f"   - {file}")
        
        return files_saved
    
    def export_for_crm(self, top_leads, filename=None, format='csv'):
        """Export top leads in CRM-friendly format"""
        if filename is None:
            filename = f"crm_export_{self.timestamp}.{format}"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare CRM-friendly data
        crm_data = []
        for i, lead in enumerate(top_leads, 1):
            crm_record = {
                'Lead_Rank': i,
                'Priority_Score': round(lead['priority_score'], 3),
                'Relevance_Score': round(lead['original_relevance'], 3),
                'Company_Domain': lead['domain'],
                'Lead_Source': lead['lead_site'],
                'Description': lead['summary'][:200] + "..." if len(lead['summary']) > 200 else lead['summary'],
                'Priority_Category': 'High' if lead['priority_score'] >= 0.8 else 'Medium' if lead['priority_score'] >= 0.5 else 'Low',
                'Export_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            crm_data.append(crm_record)
        
        if format.lower() == 'csv':
            df_crm = pd.DataFrame(crm_data)
            df_crm.to_csv(filepath, index=False)
        elif format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(crm_data, f, indent=2)
        
        print(f"CRM export saved to: {filepath}")
        return filepath
        