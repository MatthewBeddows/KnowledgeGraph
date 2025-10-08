"""
KNOWLEDGE GRAPH DEBUGGER
Diagnoses issues with semantic classification and relationship learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


class KnowledgeGraphDebugger:
    """Debug semantic knowledge graph issues"""
    
    def __init__(self, df, target_var='target'):
        self.df = df
        self.target_var = target_var
        
    def run_full_diagnosis(self):
        """Run complete diagnostic suite"""
        print("="*70)
        print("KNOWLEDGE GRAPH DIAGNOSTIC REPORT")
        print("="*70)
        
        self.check_data_quality()
        self.check_target_variable()
        self.check_correlations()
        self.check_temporal_structure()
        self.check_variable_names()
        self.test_semantic_classification()
        self.visualize_key_relationships()
        
        print("\n" + "="*70)
        print("DIAGNOSIS COMPLETE - See recommendations above")
        print("="*70)
    
    def check_data_quality(self):
        """Check overall data quality"""
        print("\n[1/7] DATA QUALITY CHECK")
        print("-"*70)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")
        print(f"Total time span: {(self.df.index.max() - self.df.index.min()).days} days")
        
        # Check for duplicates
        dupe_dates = self.df.index.duplicated().sum()
        print(f"\nDuplicate timestamps: {dupe_dates}")
        
        # Missing data
        print("\nMissing data by column (top 10):")
        missing = self.df.isnull().sum().sort_values(ascending=False)
        missing_pct = (missing / len(self.df) * 100).round(2)
        for col in missing.head(10).index:
            print(f"  {col:30s}: {missing[col]:6d} ({missing_pct[col]:5.1f}%)")
        
        # Check if data is sorted
        is_sorted = self.df.index.is_monotonic_increasing
        print(f"\nData sorted by date: {'✓ YES' if is_sorted else '✗ NO (THIS IS A PROBLEM!)'}")
        
        if not is_sorted:
            print("  ⚠️  WARNING: Data must be sorted by date for temporal analysis!")
            print("  Fix: df = df.sort_index()")
    
    def check_target_variable(self):
        """Deep dive on target variable"""
        print("\n[2/7] TARGET VARIABLE ANALYSIS")
        print("-"*70)
        
        if self.target_var not in self.df.columns:
            print(f"❌ ERROR: Target variable '{self.target_var}' not found!")
            print(f"Available columns: {list(self.df.columns)}")
            return
        
        target = self.df[self.target_var]
        
        print(f"Target variable: '{self.target_var}'")
        print(f"  Total values: {len(target)}")
        print(f"  Non-null: {target.notna().sum()} ({target.notna().sum()/len(target)*100:.1f}%)")
        print(f"  Null: {target.isna().sum()} ({target.isna().sum()/len(target)*100:.1f}%)")
        
        if target.notna().sum() == 0:
            print("  ❌ ERROR: Target has NO non-null values!")
            return
        
        clean_target = target.dropna()
        
        print(f"\nStatistics:")
        print(f"  Mean: {clean_target.mean():.4f}")
        print(f"  Median: {clean_target.median():.4f}")
        print(f"  Std: {clean_target.std():.4f}")
        print(f"  Min: {clean_target.min():.4f}")
        print(f"  Max: {clean_target.max():.4f}")
        print(f"  Range: {clean_target.max() - clean_target.min():.4f}")
        
        # Check variance
        if clean_target.std() == 0:
            print("  ❌ ERROR: Target has zero variance! All values are the same.")
        elif clean_target.std() < 0.01:
            print("  ⚠️  WARNING: Target has very low variance. Check if this is correct.")
        
        # Check for constants
        unique_vals = clean_target.nunique()
        print(f"  Unique values: {unique_vals}")
        
        if unique_vals < 10:
            print(f"  Value distribution:")
            for val, count in clean_target.value_counts().head(10).items():
                print(f"    {val}: {count} times ({count/len(clean_target)*100:.1f}%)")
        
        # Distribution
        print(f"\nDistribution:")
        print(f"  Skewness: {clean_target.skew():.4f}")
        print(f"  Kurtosis: {clean_target.kurt():.4f}")
        
        # Check by field if applicable
        if 'lookupEncoded' in self.df.columns:
            print(f"\nTarget by field:")
            field_stats = self.df.groupby('lookupEncoded')[self.target_var].agg([
                'count', 'mean', 'std', 'min', 'max'
            ])
            print(f"  Fields with target data: {(field_stats['count'] > 0).sum()}")
            print(f"  Avg values per field: {field_stats['count'].mean():.1f}")
            print(f"\n  Sample fields:")
            print(field_stats.head(5).to_string())
    
    def check_correlations(self):
        """Check correlations with target"""
        print("\n[3/7] CORRELATION ANALYSIS")
        print("-"*70)
        
        if self.target_var not in self.df.columns:
            print("Skipping - target not found")
            return
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c != self.target_var 
                       and c not in ['lookupEncoded', 'FarmEncoded']]
        
        print(f"Analyzing {len(feature_cols)} numeric features")
        
        # Calculate correlations
        correlations = {}
        for col in feature_cols:
            valid_mask = self.df[[col, self.target_var]].notna().all(axis=1)
            n_valid = valid_mask.sum()
            
            if n_valid < 10:
                continue
            
            try:
                pearson = np.corrcoef(
                    self.df.loc[valid_mask, col],
                    self.df.loc[valid_mask, self.target_var]
                )[0, 1]
                
                spearman, _ = spearmanr(
                    self.df.loc[valid_mask, col],
                    self.df.loc[valid_mask, self.target_var]
                )
                
                if not np.isnan(pearson):
                    correlations[col] = {
                        'pearson': pearson,
                        'spearman': spearman,
                        'n_samples': n_valid
                    }
            except:
                continue
        
        if not correlations:
            print("❌ ERROR: Could not calculate any correlations!")
            print("Possible causes:")
            print("  - Target variable has no overlap with features")
            print("  - All features are constant")
            print("  - Data type issues")
            return
        
        # Sort by absolute correlation
        sorted_corrs = sorted(correlations.items(), 
                            key=lambda x: abs(x[1]['pearson']), 
                            reverse=True)
        
        print(f"\nTop 20 correlations with '{self.target_var}':")
        print(f"{'Variable':<35s} {'Pearson':>8s} {'Spearman':>8s} {'N':>8s}")
        print("-"*70)
        
        for col, corr_data in sorted_corrs[:20]:
            print(f"{col:<35s} {corr_data['pearson']:>8.3f} {corr_data['spearman']:>8.3f} "
                  f"{corr_data['n_samples']:>8d}")
        
        # Check thresholds
        print(f"\nCorrelation strength breakdown:")
        strong = sum(1 for _, c in correlations.items() if abs(c['pearson']) > 0.7)
        moderate = sum(1 for _, c in correlations.items() if 0.3 < abs(c['pearson']) <= 0.7)
        weak = sum(1 for _, c in correlations.items() if 0.1 < abs(c['pearson']) <= 0.3)
        very_weak = sum(1 for _, c in correlations.items() if abs(c['pearson']) <= 0.1)
        
        print(f"  Strong (|r| > 0.7):     {strong:3d}")
        print(f"  Moderate (0.3-0.7):     {moderate:3d}")
        print(f"  Weak (0.1-0.3):         {weak:3d}")
        print(f"  Very weak (< 0.1):      {very_weak:3d}")
        
        if strong == 0 and moderate == 0:
            print("\n⚠️  WARNING: No strong or moderate correlations found!")
            print("Recommendations:")
            print("  1. Lower min_correlation threshold to 0.1 or 0.15")
            print("  2. Check if target variable is correctly specified")
            print("  3. Consider feature engineering (lags, rolling means, etc.)")
        
        return correlations
    
    def check_temporal_structure(self):
        """Check temporal structure of data"""
        print("\n[4/7] TEMPORAL STRUCTURE")
        print("-"*70)
        
        # Check if we have field information
        if 'lookupEncoded' in self.df.columns:
            fields = self.df['lookupEncoded'].unique()
            print(f"Number of fields/plots: {len(fields)}")
            
            # Sample a few fields
            sample_fields = fields[:5]
            print(f"\nTemporal structure for sample fields:")
            
            for field in sample_fields:
                field_data = self.df[self.df['lookupEncoded'] == field]
                print(f"\n  Field {int(field)}:")
                print(f"    Timestamps: {len(field_data)}")
                print(f"    Date range: {field_data.index.min()} to {field_data.index.max()}")
                print(f"    Span: {(field_data.index.max() - field_data.index.min()).days} days")
                
                # Check gaps
                if len(field_data) > 1:
                    time_diffs = field_data.index.to_series().diff().dt.days
                    print(f"    Avg gap: {time_diffs.mean():.1f} days")
                    print(f"    Max gap: {time_diffs.max():.0f} days")
                    
                # Check target availability
                target_available = field_data[self.target_var].notna().sum()
                print(f"    Target values: {target_available} ({target_available/len(field_data)*100:.1f}%)")
        else:
            print("No field information (lookupEncoded) found")
            print(f"Total timestamps: {len(self.df)}")
            
        # Check temporal autocorrelation
        if self.target_var in self.df.columns:
            target = self.df[self.target_var].dropna()
            if len(target) > 10:
                # Calculate lag-1 autocorrelation
                lag1_corr = target.autocorr(lag=1)
                print(f"\nTarget autocorrelation (lag=1): {lag1_corr:.3f}")
                
                if abs(lag1_corr) < 0.1:
                    print("  ⚠️  Low autocorrelation - data may not have strong temporal patterns")
    
    def check_variable_names(self):
        """Analyze variable naming patterns"""
        print("\n[5/7] VARIABLE NAME ANALYSIS")
        print("-"*70)
        
        columns = self.df.columns.tolist()
        print(f"Total columns: {len(columns)}")
        
        # Categorize by name patterns
        categories = {
            'shifted/lagged': [],
            'cumulative': [],
            'encoded': [],
            'temperature': [],
            'moisture/water': [],
            'nutrient': [],
            'vegetation': [],
            'other': []
        }
        
        for col in columns:
            col_lower = col.lower()
            
            if 'shift' in col_lower or 'lag' in col_lower:
                categories['shifted/lagged'].append(col)
            elif 'cumulative' in col_lower or 'cum' in col_lower:
                categories['cumulative'].append(col)
            elif 'encoded' in col_lower:
                categories['encoded'].append(col)
            elif any(x in col_lower for x in ['temp', 'temperature', 'thermal']):
                categories['temperature'].append(col)
            elif any(x in col_lower for x in ['moisture', 'water', 'rain', 'humid']):
                categories['moisture/water'].append(col)
            elif any(x in col_lower for x in ['nitrogen', 'phosph', 'potass', 'nutrient', 'n', 'p', 'k']):
                categories['nutrient'].append(col)
            elif any(x in col_lower for x in ['ndvi', 'evi', 'vegetation', 'green']):
                categories['vegetation'].append(col)
            else:
                categories['other'].append(col)
        
        print("\nVariable categories by name:")
        for cat, vars in categories.items():
            if vars:
                print(f"  {cat:20s}: {len(vars):3d} variables")
                if len(vars) <= 5:
                    for v in vars:
                        print(f"    - {v}")
                else:
                    for v in vars[:3]:
                        print(f"    - {v}")
                    print(f"    ... and {len(vars)-3} more")
    
    def test_semantic_classification(self):
        """Test semantic classification on known variables"""
        print("\n[6/7] SEMANTIC CLASSIFICATION TEST")
        print("-"*70)
        
        # Test cases with expected results
        test_cases = [
            ('ambient_temp_celsius', 'Temperature'),
            ('temperature', 'Temperature'),
            ('soil_temp', 'Temperature'),
            ('rainfall_mm', 'Precipitation'),
            ('soil_moisture', 'Moisture'),
            ('nitrogen_ppm', 'Nutrient'),
            ('ndvi', 'Vegetation_Index'),
            ('yield_kg_ha', 'Yield'),
            ('tomato_harvest_kg_ha', 'Yield'),
            ('plant_height_cm', 'Growth_Metric'),
        ]
        
        print("Testing classification on standard variable names:\n")
        print(f"{'Variable Name':<30s} {'Expected':<20s} {'Got':<20s} {'Match':<6s}")
        print("-"*70)
        
        from main import VariableOntology
        ontology = VariableOntology()
        
        failures = 0
        for var_name, expected in test_cases:
            category, subcategory, confidence = ontology.classify_variable(var_name)
            match = "✓" if category == expected else "✗"
            if category != expected:
                failures += 1
            
            print(f"{var_name:<30s} {expected:<20s} {category:<20s} {match:<6s}")
        
        print(f"\nClassification accuracy: {(len(test_cases)-failures)/len(test_cases)*100:.1f}%")
        
        if failures > 0:
            print(f"\n⚠️  {failures} classification failures detected!")
            print("This explains why transfer learning isn't working properly.")
            print("\nRecommended fix: Adjust ontology patterns or scoring weights")
    
    def visualize_key_relationships(self):
        """Create diagnostic visualizations"""
        print("\n[7/7] CREATING DIAGNOSTIC VISUALIZATIONS")
        print("-"*70)
        
        if self.target_var not in self.df.columns:
            print("Skipping - target not found")
            return
        
        import os
        os.makedirs('outputs/diagnostics', exist_ok=True)
        
        # 1. Target distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        target_clean = self.df[self.target_var].dropna()
        
        # Histogram
        axes[0, 0].hist(target_clean, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title(f'Target Distribution: {self.target_var}')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[0, 1].boxplot(target_clean)
        axes[0, 1].set_title('Target Box Plot')
        axes[0, 1].set_ylabel('Value')
        
        # Time series
        axes[1, 0].plot(self.df.index, self.df[self.target_var], alpha=0.5, linewidth=0.5)
        axes[1, 0].set_title('Target Over Time')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Autocorrelation plot
        if len(target_clean) > 50:
            from pandas.plotting import autocorrelation_plot
            autocorrelation_plot(target_clean.iloc[:1000], ax=axes[1, 1])  # Sample for speed
            axes[1, 1].set_title('Target Autocorrelation')
        
        plt.tight_layout()
        plt.savefig('outputs/diagnostics/target_analysis.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: outputs/diagnostics/target_analysis.png")
        plt.close()
        
        # 2. Correlation heatmap
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in ['lookupEncoded', 'FarmEncoded']]
        
        # Sample columns if too many
        if len(feature_cols) > 20:
            # Get top correlated features
            correlations = {}
            for col in feature_cols:
                if col == self.target_var:
                    continue
                valid_mask = self.df[[col, self.target_var]].notna().all(axis=1)
                if valid_mask.sum() > 10:
                    try:
                        corr = np.corrcoef(
                            self.df.loc[valid_mask, col],
                            self.df.loc[valid_mask, self.target_var]
                        )[0, 1]
                        if not np.isnan(corr):
                            correlations[col] = abs(corr)
                    except:
                        pass
            
            top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:15]
            selected_cols = [col for col, _ in top_features] + [self.target_var]
        else:
            selected_cols = feature_cols
        
        corr_matrix = self.df[selected_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, square=True)
        plt.title(f'Correlation Matrix (Top Features with {self.target_var})')
        plt.tight_layout()
        plt.savefig('outputs/diagnostics/correlation_heatmap.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: outputs/diagnostics/correlation_heatmap.png")
        plt.close()
        
        print("\n✓ Diagnostic visualizations complete!")


def main():
    """Run debugger"""
    print("="*70)
    print("KNOWLEDGE GRAPH DEBUGGER")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('full_dataset.csv')
    
    # Parse dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    elif all(col in df.columns for col in ['year', 'month', 'dayofmonth']):
        df['date'] = pd.to_datetime(df[['year', 'month', 'dayofmonth']].rename(
            columns={'dayofmonth': 'day'}))
        df = df.set_index('date')
        df = df.drop(columns=['year', 'month', 'dayofmonth', 'dayofyear', 
                             'weekofyear', 'quarter', 'dayofweek'], errors='ignore')
    
    # Sort by date
    df = df.sort_index()
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Run diagnostics
    debugger = KnowledgeGraphDebugger(df, target_var='target')
    debugger.run_full_diagnosis()
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review outputs/diagnostics/ for visualizations")
    print("2. Check correlation analysis above")
    print("3. Adjust min_correlation threshold based on findings")
    print("4. Fix semantic classification if needed")
    print("5. Re-run knowledge graph builder with adjusted parameters")


if __name__ == "__main__":
    main()